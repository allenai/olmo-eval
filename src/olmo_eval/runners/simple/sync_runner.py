"""Synchronous evaluation runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from olmo_eval.core.configs import expand_tasks, get_model_config
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.core.logging import get_logger
from olmo_eval.inference import InferenceProvider, ProviderType, create_provider
from olmo_eval.runners.base import BaseEvalRunner
from olmo_eval.runners.constants import ValidationError
from olmo_eval.runners.mixins import RunnerResultsMixin, S3Config
from olmo_eval.runners.utils import (
    TaskResult,
    compute_suite_aggregations,
    compute_task_hash,
    generate_experiment_id,
    run_task_impl,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = get_logger(__name__)


@dataclass
class SyncEvalRunner(RunnerResultsMixin, BaseEvalRunner):
    """Orchestrates synchronous evaluation runs across tasks."""

    model_name: str = ""
    task_specs: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    provider_override: str | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    # vLLM config
    attention_backend: str | None = None  # e.g., "FLASHINFER", "FLASH_ATTN"

    # Per-task overrides from inline spec (e.g., task::temperature=0.6)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Model overrides from inline spec (e.g., model::tokenizer=..., model::load_format=...)
    model_overrides: dict[str, Any] = field(default_factory=dict)

    # S3 upload configuration (optional)
    s3_config: S3Config | None = None

    # Experiment name for database storage
    experiment_name: str | None = None

    # Experiment group for grouping related experiments
    experiment_group: str | None = None

    # Model alias (short name used as model_name in DB, original path stored as model_path)
    alias: str | None = None

    # Output persistence options
    save_predictions: bool = True
    save_requests: bool = True

    def validate(self) -> None:
        """Validate all inputs before running.

        Raises:
            ValidationError: If any task specs are invalid.
        """
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print the resolved configuration without running."""
        table = Table(title="Run Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        model_config = get_model_config(self.model_name, **self.model_overrides)
        provider_str = self.provider_override or model_config.provider

        table.add_row("Model", model_config.model)
        if model_config.tokenizer:
            table.add_row("Tokenizer", model_config.tokenizer)
        table.add_row("Provider", provider_str)
        table.add_row("Output Dir", self.output_dir)

        console.print(table)

        # Show expanded tasks
        expanded = expand_tasks(self.task_specs)
        task_table = Table(title="Tasks to Run")
        task_table.add_column("Task Spec", style="cyan")

        for spec in expanded:
            task_table.add_row(spec)

        console.print(task_table)

    def run(self) -> dict[str, Any]:
        """Execute the evaluation run."""
        from olmo_eval.evals.tasks import AgentTask, get_task

        model_config = get_model_config(self.model_name, **self.model_overrides)

        # Determine provider (model_config.provider is a string)
        provider_str = self.provider_override or model_config.provider
        provider_type = ProviderType(provider_str)

        # Expand tasks first
        expanded_tasks = expand_tasks(self.task_specs)

        # Check for agent tasks - they must use AgentEvalRunner
        agent_tasks = [spec for spec in expanded_tasks if isinstance(get_task(spec), AgentTask)]
        if agent_tasks:
            raise ValidationError(
                f"Agent tasks found: {', '.join(agent_tasks)}. Use AgentEvalRunner for agent tasks."
            )

        # vLLM requires 'spawn' multiprocessing start method
        if provider_type == ProviderType.VLLM:
            current = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
            if current != "spawn":
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                if current:
                    logger.info(
                        f"Overriding VLLM_WORKER_MULTIPROC_METHOD from '{current}' to 'spawn'"
                    )

        console.print(f"[bold]Initializing {provider_type.value} provider...[/bold]")
        extra_kwargs = dict(model_config.extra_args)
        if self.attention_backend:
            extra_kwargs["attention_backend"] = self.attention_backend
        if model_config.max_model_len:
            extra_kwargs["max_model_len"] = model_config.max_model_len
        # Per-model vLLM loading options from inline spec (e.g., model::load_format=runai_streamer)
        if self.model_overrides.get("load_format"):
            extra_kwargs["load_format"] = self.model_overrides["load_format"]
        if self.model_overrides.get("extra_loader_config"):
            extra_kwargs["model_loader_extra_config"] = self.model_overrides["extra_loader_config"]
        provider = create_provider(
            provider_type,
            model_config.model,
            tokenizer=model_config.tokenizer,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            dtype=model_config.dtype,
            **extra_kwargs,
        )

        from olmo_eval.runners.mixins import get_model_display_name

        model_alias = self.model_overrides.get("alias")
        display_model_name = get_model_display_name(model_config.model, model_alias)

        # Build model_config dict from ModelConfig, adding runner-specific fields
        model_config_dict = model_config.to_dict()
        model_config_dict["attention_backend"] = self.attention_backend

        results: dict[str, Any] = {
            "model": display_model_name,
            "model_path": model_config.model,  # Original full path
            "provider": provider_type.value,
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
            "model_config": model_config_dict,
        }

        for spec in expanded_tasks:
            console.print(f"\n[bold blue]Running {spec}...[/bold blue]")
            task_result = self._run_task(spec, provider)
            task_data = task_result.to_dict(include_predictions=True)

            # Compute task hash from config and add to task_data
            task_hash = compute_task_hash(task_result.config)
            if task_hash:
                task_data["task_hash"] = task_hash

            results["tasks"][spec] = task_data

            # Write predictions to JSONL
            if self.save_predictions and task_result.predictions:
                self._write_predictions(
                    display_model_name, spec, task_result.predictions, task_hash
                )

            # Write requests to JSONL (with hash now that we have the config)
            if self.save_requests and task_result.requests:
                self._write_requests(display_model_name, spec, task_result.requests, task_hash)

            # Log metrics (for Beaker job details)
            if task_result.metrics:
                logger.info(f"** Task metrics for {spec}: **")
                for metric, value in task_result.metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                    console.print(f"  {metric}: {value:.4f}")

        # Compute suite aggregations
        suite_aggs = compute_suite_aggregations(self.task_specs, results["tasks"])
        if suite_aggs:
            results["suites"] = suite_aggs

        # Log summary of all scores
        self._log_summary(results)

        # Compute experiment_id and model_hash upfront for both metrics.json and storage
        from olmo_eval.core.types import compute_model_hash

        experiment_id = generate_experiment_id()
        model_hash = compute_model_hash(results.get("model_config", {}))

        # Write metrics.json for Beaker (with experiment identification fields)
        self._write_metrics_json(
            results,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_hash=model_hash,
        )

        s3_location: str | None = None

        # Upload to S3 first if configured (so we have s3_location for storage)
        if self.s3_config and model_hash:
            s3_location = self._upload_to_s3(
                model_name=results["model"],
                model_hash=model_hash,
                experiment_id=experiment_id,
            )

        # Save to storage backends with all context
        self._save_results(
            results,
            experiment_id=experiment_id,
            model_hash=model_hash,
            s3_location=s3_location,
        )

        return results

    def _run_task(self, spec: str, provider: InferenceProvider) -> TaskResult:
        """Run a single task and return results."""
        # Build overrides from per-task inline overrides
        overrides, sampling_overrides = self._build_task_overrides(spec)

        # Standard task - use shared task execution logic
        result = run_task_impl(
            spec=spec,
            provider=provider,
            overrides=overrides or None,
            progress_callback=lambda msg: console.print(f"  {msg}"),
            sampling_overrides=sampling_overrides or None,
        )

        # Check for errors
        if result.error:
            raise RuntimeError(f"Task {spec} failed: {result.error}")

        return result

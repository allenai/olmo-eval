"""Evaluation runner orchestrator."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from olmo_eval.backends import Backend, BackendType, create_backend
from olmo_eval.core import expand_tasks, get_model_config
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.runners.constants import SAMPLING_KEYS, TASKCONFIG_KEYS, ValidationError
from olmo_eval.runners.mixins import RunnerResultsMixin, S3Config
from olmo_eval.runners.utils import (
    TaskResult,
    compute_suite_aggregations,
    compute_task_hash,
    generate_experiment_id,
    run_task_impl,
    write_predictions_jsonl,
    write_requests_jsonl,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


@dataclass
class SyncEvalRunner(RunnerResultsMixin):
    """Orchestrates synchronous evaluation runs across tasks."""

    model_name: str
    task_specs: list[str]
    output_dir: str = BEAKER_RESULT_DIR
    num_shots_override: int | None = None
    limit_override: int | None = None
    temperature: float | None = None
    backend_override: str | None = None
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
        backend_str = self.backend_override or model_config.backend

        table.add_row("Model", model_config.model)
        if model_config.tokenizer:
            table.add_row("Tokenizer", model_config.tokenizer)
        table.add_row("Backend", backend_str)
        table.add_row("Output Dir", self.output_dir)

        if self.num_shots_override is not None:
            table.add_row("Num Shots Override", str(self.num_shots_override))
        if self.limit_override is not None:
            table.add_row("Limit Override", str(self.limit_override))

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
        model_config = get_model_config(self.model_name, **self.model_overrides)

        # Determine backend (model_config.backend is a string)
        backend_str = self.backend_override or model_config.backend
        backend_type = BackendType(backend_str)

        # vLLM requires 'spawn' multiprocessing start method
        if backend_type == BackendType.VLLM:
            current = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
            if current != "spawn":
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
                if current:
                    logger.info(
                        f"Overriding VLLM_WORKER_MULTIPROC_METHOD from '{current}' to 'spawn'"
                    )

        console.print(f"[bold]Initializing {backend_type.value} backend...[/bold]")
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
        backend = create_backend(
            backend_type,
            model_config.model,
            tokenizer=model_config.tokenizer,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            dtype=model_config.dtype,
            **extra_kwargs,
        )

        expanded_tasks = expand_tasks(self.task_specs)

        from olmo_eval.runners.mixins import get_model_display_name

        model_alias = self.model_overrides.get("alias")
        display_model_name = get_model_display_name(model_config.model, model_alias)

        results: dict[str, Any] = {
            "model": display_model_name,
            "model_path": model_config.model,  # Original full path
            "backend": backend_type.value,
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
            # Store model config details for metrics.json
            "model_config": {
                "model": model_config.model,
                "tokenizer": model_config.tokenizer,
                "backend": backend_type.value,
                "dtype": model_config.dtype,
                "revision": model_config.revision,
                "attention_backend": self.attention_backend,
            },
        }

        for spec in expanded_tasks:
            console.print(f"\n[bold blue]Running {spec}...[/bold blue]")
            task_result = self._run_task(spec, backend)
            task_data: dict[str, Any] = {
                "config": task_result.config,
                "num_instances": task_result.num_instances,
                "metrics": task_result.metrics,
                "duration_seconds": task_result.duration_seconds,
            }
            if task_result.primary_metric:
                task_data["primary_metric"] = task_result.primary_metric
            if task_result.predictions:
                task_data["predictions"] = task_result.predictions

            # Compute task hash from config and add to task_data
            task_hash = compute_task_hash(task_result.config)
            if task_hash:
                task_data["task_hash"] = task_hash

            results["tasks"][spec] = task_data

            # Write predictions to JSONL
            if task_result.predictions:
                self._write_predictions(
                    display_model_name, spec, task_result.predictions, task_hash
                )

            # Write requests to JSONL (with hash now that we have the config)
            if task_result.requests:
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

    def _run_task(self, spec: str, backend: Backend) -> TaskResult:
        """Run a single task and return results."""
        # Build overrides from instance settings (global CLI overrides)
        overrides: dict[str, Any] = {}
        sampling_overrides: dict[str, Any] = {}

        if self.num_shots_override is not None:
            overrides["num_fewshot"] = self.num_shots_override
        if self.limit_override is not None:
            overrides["limit"] = self.limit_override
        if self.temperature is not None:
            sampling_overrides["temperature"] = self.temperature

        # Apply per-task overrides from spec (highest priority)
        task_specific_overrides = self.task_overrides.get(spec, {})
        for key, value in task_specific_overrides.items():
            if key in TASKCONFIG_KEYS:
                overrides[key] = value
            elif key in SAMPLING_KEYS:
                sampling_overrides[key] = value

        # Use shared task execution logic
        result = run_task_impl(
            spec=spec,
            backend=backend,
            overrides=overrides or None,
            progress_callback=lambda msg: console.print(f"  {msg}"),
            sampling_overrides=sampling_overrides or None,
        )

        # Check for errors
        if result.error:
            raise RuntimeError(f"Task {spec} failed: {result.error}")

        return result

    def _write_predictions(
        self, model_name: str, spec: str, predictions: list[dict], task_hash: str | None = None
    ) -> None:
        """Write per-instance predictions to JSONL."""
        write_predictions_jsonl(self.output_dir, spec, predictions, model_name, task_hash=task_hash)

    def _write_requests(
        self, model_name: str, spec: str, requests: list[dict], task_hash: str | None = None
    ) -> None:
        """Write per-instance requests to JSONL (oe-eval compatible format)."""
        write_requests_jsonl(self.output_dir, spec, requests, model_name, task_hash=task_hash)

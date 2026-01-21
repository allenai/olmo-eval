"""Evaluation runner orchestrator."""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from olmo_eval.backends import Backend, BackendType, create_backend
from olmo_eval.core import expand_tasks, get_model_config
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.evals.suites import suite_exists
from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
from olmo_eval.evals.tasks.core.registry import parse_task_spec
from olmo_eval.runners.utils import (
    TaskResult,
    compute_suite_aggregations,
    get_primary_metric,
    run_task_impl,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation of runner inputs fails."""

    pass


# Keys that apply to TaskConfig
TASKCONFIG_KEYS = {"num_fewshot", "limit", "fewshot_seed"}

# Keys that apply to SamplingParams
SAMPLING_KEYS = {"temperature", "max_tokens", "top_p", "top_k", "num_samples"}


@dataclass
class EvalRunner:
    """Orchestrates evaluation runs across tasks."""

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

    # Model overrides from inline spec (e.g., model::tokenizer=...)
    model_overrides: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate all inputs before running.

        Raises:
            ValidationError: If any task specs are invalid.
        """
        errors: list[str] = []

        # Validate task specs
        available_tasks = set(list_tasks())
        regimes_by_task = list_regimes()
        variants_by_task = list_variants()

        for spec in self.task_specs:
            # Check if it's a suite
            if suite_exists(spec):
                continue

            # Parse task_name[:variant1[:variant2...]][::key=value,...] format
            # Note: regimes are now treated as variants
            task_name, variants, _overrides = parse_task_spec(spec)

            if task_name not in available_tasks:
                errors.append(f"Unknown task or suite: '{spec}'")
                continue

            # Validate each variant/regime exists (check both registries)
            task_variants = set(variants_by_task.get(task_name, []))
            task_regimes = set(regimes_by_task.get(task_name, []))
            all_valid_variants = task_variants | task_regimes

            for variant in variants:
                if variant not in all_valid_variants:
                    available_list = sorted(all_valid_variants)
                    if available_list:
                        errors.append(
                            f"Unknown variant/regime '{variant}' for task '{task_name}'. "
                            f"Available: {', '.join(available_list)}"
                        )
                    else:
                        errors.append(
                            f"Unknown variant/regime '{variant}' for task '{task_name}'. "
                            f"This task has no registered variants or regimes."
                        )

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

        console.print(f"[bold]Initializing {backend_type.value} backend...[/bold]")
        if model_config.tokenizer:
            console.print(f"[dim]Tokenizer: {model_config.tokenizer}[/dim]")
        extra_kwargs = dict(model_config.extra_args)
        if self.attention_backend:
            extra_kwargs["attention_backend"] = self.attention_backend
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
        results: dict[str, Any] = {
            "model": model_config.model,
            "backend": backend_type.value,
            "timestamp": datetime.now().isoformat(),
            "tasks": {},
        }

        # TODO(undfined): This is starting with the naive approach. Add intelligent
        # scheduling and possibly asynchronous runner.
        for spec in expanded_tasks:
            console.print(f"\n[bold blue]Running {spec}...[/bold blue]")
            task_result = self._run_task(spec, backend)
            results["tasks"][spec] = {
                "config": task_result.config,
                "num_instances": task_result.num_instances,
                "metrics": task_result.metrics,
            }

            # Write predictions to JSONL
            if task_result.predictions:
                self._write_predictions(spec, task_result.predictions)

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
        self._save_results(results)

        # Write metrics.json for Beaker
        self._write_metrics_json(results)

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

    def _log_summary(self, results: dict[str, Any]) -> None:
        """Log summary of all task scores."""
        logger.info("Summary of primary scores:")
        for task_name, task_data in results["tasks"].items():
            metrics = task_data.get("metrics", {})
            primary = get_primary_metric(metrics)
            if primary:
                metric_name, score = primary
                logger.info(f"  {task_name}: {score:.4f} ({metric_name})")

        for suite_name, suite_data in results.get("suites", {}).items():
            metrics = suite_data.get("metrics", {})
            primary = get_primary_metric(metrics)
            if primary:
                metric_name, score = primary
                logger.info(f"  {suite_name}: {score:.4f} ({metric_name})")

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results to all configured storage backends."""
        if self.storages:
            from olmo_eval.storage.base import convert_runner_results

            run_id = str(uuid.uuid4())
            eval_result = convert_runner_results(results, run_id)
            for storage in self.storages:
                storage.save(eval_result)
                backend_name = type(storage).__name__
                logger.info(f"Results saved to {backend_name} (run_id: {run_id})")
                console.print(f"[green]Results saved to {backend_name} (run_id: {run_id})[/green]")
        else:
            logger.info("No storage backend configured - results logged above only")

    def _write_metrics_json(self, results: dict[str, Any]) -> None:
        """Write metrics.json for Beaker display."""
        metrics_file = os.path.join(self.output_dir, "metrics.json")

        # Build simplified metrics structure
        tasks_list = [
            {
                "task": task_name,
                "metrics": task_data.get("metrics", {}),
                "num_instances": task_data.get("num_instances", 0),
            }
            for task_name, task_data in results.get("tasks", {}).items()
        ]

        # Build summary with primary metric for each task
        summary: dict[str, dict[str, Any]] = {}
        for task_name, task_data in results.get("tasks", {}).items():
            metrics = task_data.get("metrics", {})
            primary = get_primary_metric(metrics)
            if primary:
                metric_name, score = primary
                summary[task_name] = {"metric": metric_name, "score": score}

        # Build suite summaries if present
        suites_list = []
        if "suites" in results:
            for suite_name, suite_data in results["suites"].items():
                suites_list.append(
                    {
                        "suite": suite_name,
                        "metrics": suite_data.get("metrics", {}),
                        "num_tasks": suite_data.get("num_tasks", 0),
                        "aggregation": suite_data.get("aggregation", "mean"),
                    }
                )
                metrics = suite_data.get("metrics", {})
                primary = get_primary_metric(metrics)
                if primary:
                    metric_name, score = primary
                    summary[suite_name] = {"metric": metric_name, "score": score}

        metrics_output = {
            "model": results.get("model", ""),
            "backend": results.get("backend", ""),
            "timestamp": results.get("timestamp", ""),
            "tasks": tasks_list,
            "suites": suites_list,
            "summary": summary,
            "errors": results.get("errors", []),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics_output, f, indent=2)

        logger.info(f"Metrics written to {metrics_file}")
        console.print(f"[green]Metrics written to {metrics_file}[/green]")

    def _write_predictions(self, spec: str, predictions: list[dict]) -> None:
        """Write per-instance predictions to JSONL.

        Args:
            spec: Task specification string (used for filename)
            predictions: List of prediction dicts to write
        """
        pred_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        # Sanitize spec for filename: arc_challenge:bpb::olmes -> arc_challenge_bpb__olmes
        filename = spec.replace(":", "_").replace("/", "_") + ".jsonl"
        filepath = os.path.join(pred_dir, filename)

        with open(filepath, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

        logger.info(f"Predictions written to {filepath}")

"""Shared mixin classes for evaluation runners.

This module provides mixin classes that add common functionality to runners.
Core logic has been extracted to dedicated modules:
- models: Dataclasses for configuration and output
- formatting: Model name sanitization and S3 path building
- storage: S3 upload and storage backend operations
- metrics: Metrics building and writing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmo_eval.common.console import console
from olmo_eval.common.logging import get_logger
from olmo_eval.runners.common.models import (
    MetricsOutput,
    ModelMetadata,
    S3Config,
    ScoreSummary,
    TaskMetricsEntry,
)
from olmo_eval.runners.common.types import TaskResult
from olmo_eval.runners.io.formatting import (  # noqa: F401
    build_s3_prefix,
    get_model_display_name,
    sanitize_model_name,
)
from olmo_eval.runners.io.storage import save_results, upload_to_s3
from olmo_eval.runners.io.writers import write_predictions_jsonl, write_requests_jsonl
from olmo_eval.runners.processing.metrics import (
    build_multi_model_metrics,
    build_single_model_metrics,
    log_summary,
    write_metrics_json,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

logger = get_logger("runners.mixins")


__all__ = [
    "S3Config",
    "sanitize_model_name",
    "get_model_display_name",
    "build_s3_prefix",
    "ModelMetadata",
    "TaskMetricsEntry",
    "ScoreSummary",
    "MetricsOutput",
    "RunnerResultsMixin",
]


class RunnerResultsMixin:
    """Shared results-writing functionality for runners."""

    output_dir: str
    storages: list[StorageBackend]
    task_specs: list[str]

    # Optional S3 upload configuration
    s3_config: S3Config | None

    # Per-task overrides
    task_overrides: dict[str, dict[str, Any]]

    def _validate_task_specs(self) -> list[str]:
        """Validate task specs and return list of errors.

        Checks that:
        1. All tasks/suites exist
        2. All variants are valid
        3. All tasks have metrics configured

        Returns:
            List of error messages (empty if all specs are valid).
        """
        from olmo_eval.common.configs import expand_tasks, validate_task_metrics
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks.common import list_tasks, list_variants

        errors: list[str] = []
        available_tasks = set(list_tasks())
        variants_by_task = list_variants()

        for spec in self.task_specs:
            if suite_exists(spec):
                continue

            # Parse task_name[:variant1[:variant2...]] format
            # Try progressively shorter prefixes to handle tasks with colons in names
            parts = spec.split(":")
            task_name = None
            variants: list[str] = []
            for i in range(len(parts), 0, -1):
                candidate = ":".join(parts[:i])
                if candidate in available_tasks:
                    task_name = candidate
                    variants = [v for v in parts[i:] if v]
                    break

            if task_name is None:
                errors.append(f"Unknown task or suite: '{spec}'")
                continue

            # Validate each variant exists.
            task_variants = set(variants_by_task.get(task_name, []))

            for variant in variants:
                if not variant:
                    continue
                if variant not in task_variants:
                    available_list = sorted(task_variants)
                    if available_list:
                        errors.append(
                            f"Unknown variant '{variant}' for task '{task_name}'. "
                            f"Available: {', '.join(available_list)}"
                        )
                    else:
                        errors.append(
                            f"Unknown variant '{variant}' for task '{task_name}'. "
                            "This task has no registered variants."
                        )

        # If we have errors so far, return early (can't validate metrics on invalid tasks)
        if errors:
            return errors

        # Check for tasks without metrics configured
        expanded_tasks = expand_tasks(self.task_specs)
        _with_metrics, without_metrics = validate_task_metrics(expanded_tasks)

        for spec in without_metrics:
            errors.append(
                f"Task '{spec}' has no metrics configured. "
                f"Use a variant with metrics (e.g., '{spec}:bpb') or register metrics for the task."
            )

        return errors

    def _save_results(
        self,
        results: dict[str, Any],
        experiment_id: str | None = None,
        model_hash: str | None = None,
        s3_location: str | None = None,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> None:
        """Save results to all configured storage backends.

        Thin wrapper around storage.save_results() with runner-specific context.
        """
        s3_cfg = getattr(self, "s3_config", None)
        runner_experiment_name = getattr(self, "experiment_name", None)
        runner_experiment_group = getattr(self, "experiment_group", None)

        save_results(
            results=results,
            storages=self.storages,
            s3_config=s3_cfg,
            experiment_id=experiment_id,
            model_hash=model_hash,
            s3_location=s3_location,
            experiment_name=runner_experiment_name,
            experiment_group=runner_experiment_group,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

    def _upload_to_s3(
        self,
        model_name: str,
        model_hash: str,
        experiment_id: str,
    ) -> str | None:
        """Upload evaluation output to S3.

        Thin wrapper around storage.upload_to_s3().
        """
        s3_config = getattr(self, "s3_config", None)
        if not s3_config:
            logger.debug("S3 upload not configured; skipping.")
            return None

        return upload_to_s3(
            output_dir=self.output_dir,
            s3_config=s3_config,
            model_name=model_name,
            model_hash=model_hash,
            experiment_id=experiment_id,
        )

    def _log_summary(self, results: dict[str, Any], multi_model: bool = False) -> None:
        """Log summary of all task scores.

        Thin wrapper around metrics.log_summary().
        """
        log_summary(results, multi_model=multi_model)

    def _write_metrics_json(
        self,
        results: dict[str, Any],
        multi_model: bool = False,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        model_hash: str | None = None,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> None:
        """Write metrics.json.

        Thin wrapper around metrics.write_metrics_json().
        """
        write_metrics_json(
            output_dir=self.output_dir,
            results=results,
            multi_model=multi_model,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

    def _build_single_model_metrics(
        self,
        results: dict[str, Any],
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        model_hash: str | None = None,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> MetricsOutput:
        """Build metrics output for single-model format.

        Thin wrapper around metrics.build_single_model_metrics().
        """
        return build_single_model_metrics(
            results,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

    def _build_multi_model_metrics(
        self,
        results: dict[str, Any],
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> MetricsOutput:
        """Build metrics output for multi-model format.

        Thin wrapper around metrics.build_multi_model_metrics().
        """
        return build_multi_model_metrics(
            results,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

    def _report_task_completion(self, model_name: str, result: TaskResult) -> None:
        """Report when a task completes."""
        label = f"{model_name}:{result.spec}"
        if result.error:
            console.print(f"  [red]✗[/red] {label} (ERROR: {result.error})")
        else:
            console.print(
                f"  [green]✓[/green] {label} ({result.num_instances} instances, "
                f"{result.duration_seconds:.1f}s)"
            )

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

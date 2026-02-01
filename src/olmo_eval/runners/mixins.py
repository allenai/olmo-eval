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

from rich.console import Console

from olmo_eval.core.logging import get_logger

# Re-export for backward compatibility
from olmo_eval.runners.formatting import (  # noqa: F401
    build_s3_prefix,
    get_model_display_name,
    sanitize_model_name,
)
from olmo_eval.runners.metrics import (
    build_multi_model_metrics,
    build_single_model_metrics,
    log_summary,
    write_metrics_json,
)
from olmo_eval.runners.models import (
    MetricsOutput,
    ModelConfig,
    S3Config,
    ScoreSummary,
    TaskMetricsEntry,
)
from olmo_eval.runners.storage import save_results, upload_to_s3
from olmo_eval.runners.types import TaskResult
from olmo_eval.runners.writers import write_predictions_jsonl, write_requests_jsonl

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = get_logger("runners.mixins")


# Re-export dataclasses for backward compatibility
__all__ = [
    "S3Config",
    "sanitize_model_name",
    "get_model_display_name",
    "build_s3_prefix",
    "ModelConfig",
    "TaskMetricsEntry",
    "ScoreSummary",
    "MetricsOutput",
    "RunnerResultsMixin",
    "AsyncRunnerMixin",
]


class RunnerResultsMixin:
    """Shared results-writing functionality for runners."""

    output_dir: str
    storages: list[StorageBackend]
    task_specs: list[str]

    # Optional S3 upload configuration
    s3_config: S3Config | None

    # Per-task overrides from inline spec (e.g., task::temperature=0.6)
    task_overrides: dict[str, dict[str, Any]]

    def _build_task_overrides(self, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build task and sampling overrides for a given task spec.

        Returns:
            Tuple of (task_overrides, sampling_overrides)
        """
        from olmo_eval.core.types import SamplingParams
        from olmo_eval.evals.tasks.core.base import TaskConfig

        task_overrides: dict[str, Any] = {}
        sampling_overrides: dict[str, Any] = {}

        # Apply per-task inline overrides
        per_task = getattr(self, "task_overrides", {}).get(spec, {})
        for key, value in per_task.items():
            if key in TaskConfig.OVERRIDE_KEYS:
                task_overrides[key] = value
            elif key in SamplingParams.OVERRIDE_KEYS:
                sampling_overrides[key] = value

        return task_overrides, sampling_overrides

    def _validate_task_specs(self) -> list[str]:
        """Validate task specs and return list of errors.

        Returns:
            List of error messages (empty if all specs are valid).
        """
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
        from olmo_eval.evals.tasks.core.registry import parse_task_spec

        errors: list[str] = []
        available_tasks = set(list_tasks())
        regimes_by_task = list_regimes()
        variants_by_task = list_variants()

        for spec in self.task_specs:
            if suite_exists(spec):
                continue

            # Parse task_name[:variant1[:variant2...]][::key=value,...] format
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
        """Report when a (model, task) pair completes."""
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


class AsyncRunnerMixin(RunnerResultsMixin):
    """Additional shared functionality for async runners."""

    model_names: list[str]
    task_specs: list[str]
    num_workers: int | None
    gpus_per_worker: int

    # Subclasses should override these for print_config display
    _mode_name: str = "Async"
    _mode_description: str = "Async"

    def validate(self) -> None:
        """Validate configuration."""
        from olmo_eval.runners.constants import ValidationError

        if not self.model_names:
            raise ValidationError("model_names is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs using shared helper
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print configuration."""
        from rich.table import Table

        from olmo_eval.core.configs import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        models_str = ", ".join(self.model_names)
        table.add_row("Models", models_str)
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)
        table.add_row("Workers", str(self.num_workers or "auto-detect"))
        table.add_row("GPUs per Worker", str(self.gpus_per_worker))

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        total_pairs = len(self.model_names) * len(expanded)
        console.print(f"\n[bold]Models:[/bold] {len(self.model_names)}")
        console.print(f"[bold]Tasks:[/bold] {len(expanded)}")
        console.print(f"[bold]Total (model, task) pairs:[/bold] {total_pairs}")
        for spec in expanded:
            console.print(f"  - {spec}")

    def _get_num_workers(self) -> int:
        """Get number of workers based on available GPUs."""
        if self.num_workers is not None:
            return self.num_workers

        # Auto-detect GPUs
        try:
            import torch  # type: ignore[import-not-found]

            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                return 1  # Fallback to single worker for CPU
            return max(1, num_gpus // self.gpus_per_worker)
        except ImportError:
            return 1  # Fallback to single worker if torch unavailable

    def _get_total_gpus(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch  # type: ignore[import-not-found]

            return torch.cuda.device_count()
        except ImportError:
            return 0

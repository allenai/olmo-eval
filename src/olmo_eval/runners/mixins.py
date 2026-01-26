"""Shared mixin classes for evaluation runners."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.runners.utils import TaskResult, get_primary_metric

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


class RunnerResultsMixin:
    """Shared results-writing functionality for runners."""

    output_dir: str
    storages: list[StorageBackend]

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results to all configured storage backends."""
        if self.storages:
            from olmo_eval.storage.base import convert_runner_results

            experiment_id = str(uuid.uuid4())
            eval_result = convert_runner_results(results, experiment_id)
            for storage in self.storages:
                storage.save(eval_result)
                backend_name = type(storage).__name__
                logger.info(f"Results saved to {backend_name} (experiment_id: {experiment_id})")
                msg = f"Results saved to {backend_name} (experiment_id: {experiment_id})"
                console.print(f"[green]{msg}[/green]")
        else:
            logger.info("No storage backend configured - results logged above only")

    def _log_summary(self, results: dict[str, Any], multi_model: bool = False) -> None:
        """Log summary of all task scores.

        Args:
            results: Results dictionary
            multi_model: If True, iterate results["models"][model]["tasks"],
                        otherwise iterate results["tasks"] directly
        """
        logger.info("Summary of primary scores:")

        if multi_model:
            for model_name, model_data in results.get("models", {}).items():
                logger.info(f"  {model_name}:")
                for task_name, task_data in model_data.get("tasks", {}).items():
                    metrics = task_data.get("metrics", {})
                    preferred = task_data.get("primary_metric")
                    primary = get_primary_metric(metrics, preferred)
                    if primary:
                        metric_name, score = primary
                        logger.info(f"    {task_name}: {score:.4f} ({metric_name})")

                for suite_name, suite_data in model_data.get("suites", {}).items():
                    metrics = suite_data.get("metrics", {})
                    primary = get_primary_metric(metrics)
                    if primary:
                        metric_name, score = primary
                        logger.info(f"    {suite_name}: {score:.4f} ({metric_name})")
        else:
            for task_name, task_data in results["tasks"].items():
                metrics = task_data.get("metrics", {})
                preferred = task_data.get("primary_metric")
                primary = get_primary_metric(metrics, preferred)
                if primary:
                    metric_name, score = primary
                    logger.info(f"  {task_name}: {score:.4f} ({metric_name})")

            for suite_name, suite_data in results.get("suites", {}).items():
                metrics = suite_data.get("metrics", {})
                primary = get_primary_metric(metrics)
                if primary:
                    metric_name, score = primary
                    logger.info(f"  {suite_name}: {score:.4f} ({metric_name})")

    def _write_metrics_json(self, results: dict[str, Any], multi_model: bool = False) -> None:
        """Write metrics.json for Beaker display.

        Args:
            results: Results dictionary
            multi_model: If True, use multi-model format with results["models"],
                        otherwise use single-model format with results["tasks"]
        """
        metrics_file = os.path.join(self.output_dir, "metrics.json")

        if multi_model:
            metrics_output = self._build_multi_model_metrics(results)
        else:
            metrics_output = self._build_single_model_metrics(results)

        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics_output, f, indent=2)

        logger.info(f"Metrics written to {metrics_file}")
        console.print(f"[green]Metrics written to {metrics_file}[/green]")

    def _build_single_model_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Build metrics output for single-model format."""
        # Build config section from stored model config
        model_cfg = results.get("_model_config", {})
        config: dict[str, Any] = {
            "model": model_cfg.get("model", results.get("model", "")),
            "backend": model_cfg.get("backend", results.get("backend", "")),
            "dtype": model_cfg.get("dtype", "auto"),
        }
        # Only include optional fields if they have values
        if model_cfg.get("tokenizer"):
            config["tokenizer"] = model_cfg["tokenizer"]
        if model_cfg.get("revision"):
            config["revision"] = model_cfg["revision"]
        if model_cfg.get("attention_backend"):
            config["attention_backend"] = model_cfg["attention_backend"]

        # Build enhanced task entries
        tasks_list = []
        for task_name, task_data in results.get("tasks", {}).items():
            task_entry: dict[str, Any] = {
                "task": task_name,
                "metrics": task_data.get("metrics", {}),
                "num_instances": task_data.get("num_instances", 0),
            }
            if task_data.get("primary_metric"):
                task_entry["primary_metric"] = task_data["primary_metric"]
            if task_data.get("config"):
                task_entry["config"] = task_data["config"]
            if task_data.get("duration_seconds"):
                task_entry["duration_seconds"] = task_data["duration_seconds"]
            tasks_list.append(task_entry)

        # Build summary with primary metric for each task
        summary: dict[str, dict[str, Any]] = {}
        for task_name, task_data in results.get("tasks", {}).items():
            metrics = task_data.get("metrics", {})
            preferred = task_data.get("primary_metric")
            primary = get_primary_metric(metrics, preferred)
            if primary:
                metric_name, score = primary
                summary[task_name] = {"metric": metric_name, "score": score}

        # Add suite summaries to summary (without separate suites list)
        if "suites" in results:
            for suite_name, suite_data in results["suites"].items():
                metrics = suite_data.get("metrics", {})
                primary = get_primary_metric(metrics)
                if primary:
                    metric_name, score = primary
                    summary[suite_name] = {"metric": metric_name, "score": score}

        return {
            "timestamp": results.get("timestamp", ""),
            "config": config,
            "tasks": tasks_list,
            "summary": summary,
            "errors": results.get("errors", []),
        }

    def _build_multi_model_metrics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Build metrics output for multi-model format."""
        # Build config section with per-model details
        models_config: dict[str, dict[str, Any]] = {}
        for model_name, model_data in results.get("models", {}).items():
            model_cfg = model_data.get("_model_config", {})
            cfg: dict[str, Any] = {
                "model": model_cfg.get("model", model_data.get("model", "")),
                "backend": model_cfg.get("backend", model_data.get("backend", "")),
                "dtype": model_cfg.get("dtype", "auto"),
            }
            # Only include optional fields if they have values
            if model_cfg.get("tokenizer"):
                cfg["tokenizer"] = model_cfg["tokenizer"]
            if model_cfg.get("revision"):
                cfg["revision"] = model_cfg["revision"]
            if model_cfg.get("attention_backend"):
                cfg["attention_backend"] = model_cfg["attention_backend"]
            models_config[model_name] = cfg

        config: dict[str, Any] = {"models": models_config}

        # Build enhanced task entries - flatten (model, task) pairs
        tasks_list = []
        for model_name, model_data in results.get("models", {}).items():
            for task_name, task_data in model_data.get("tasks", {}).items():
                task_entry: dict[str, Any] = {
                    "model": model_name,
                    "task": task_name,
                    "metrics": task_data.get("metrics", {}),
                    "num_instances": task_data.get("num_instances", 0),
                }
                if task_data.get("primary_metric"):
                    task_entry["primary_metric"] = task_data["primary_metric"]
                if task_data.get("config"):
                    task_entry["config"] = task_data["config"]
                if task_data.get("duration_seconds"):
                    task_entry["duration_seconds"] = task_data["duration_seconds"]
                tasks_list.append(task_entry)

        # Build summary with primary metric for each (model, task) pair
        summary: dict[str, dict[str, dict[str, Any]]] = {}
        for model_name, model_data in results.get("models", {}).items():
            summary[model_name] = {}
            for task_name, task_data in model_data.get("tasks", {}).items():
                metrics = task_data.get("metrics", {})
                preferred = task_data.get("primary_metric")
                primary = get_primary_metric(metrics, preferred)
                if primary:
                    metric_name, score = primary
                    summary[model_name][task_name] = {"metric": metric_name, "score": score}

            # Add suite summaries to this model's summary
            if "suites" in model_data:
                for suite_name, suite_data in model_data["suites"].items():
                    metrics = suite_data.get("metrics", {})
                    primary = get_primary_metric(metrics)
                    if primary:
                        metric_name, score = primary
                        summary[model_name][suite_name] = {"metric": metric_name, "score": score}

        return {
            "timestamp": results.get("timestamp", ""),
            "config": config,
            "tasks": tasks_list,
            "summary": summary,
            "errors": results.get("errors", []),
        }

    def _report_task_completion(self, model_name: str, result: TaskResult) -> None:
        """Report when a (model, task) pair completes."""
        label = f"{model_name}:{result.spec}"
        if result.error:
            console.print(f"  [red]x[/red] {label} (ERROR: {result.error})")
        else:
            console.print(
                f"  [green]v[/green] {label} ({result.num_instances} instances, "
                f"{result.duration_seconds:.1f}s)"
            )


class AsyncRunnerMixin(RunnerResultsMixin):
    """Additional shared functionality for async runners."""

    model_names: list[str]
    task_specs: list[str]
    num_workers: int | None
    gpus_per_worker: int
    num_shots_override: int | None
    limit_override: int | None

    # Subclasses should override these for print_config display
    _mode_name: str = "Async"
    _mode_description: str = "Async"

    def validate(self) -> None:
        """Validate configuration."""
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
        from olmo_eval.evals.tasks.core.registry import parse_task_spec
        from olmo_eval.runners.synchronous import ValidationError

        if not self.model_names:
            raise ValidationError("model_names is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs
        errors: list[str] = []
        available_tasks = set(list_tasks())
        regimes_by_task = list_regimes()
        variants_by_task = list_variants()

        for spec in self.task_specs:
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
        """Print configuration."""
        from rich.table import Table

        from olmo_eval.core import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        models_str = ", ".join(self.model_names)
        table.add_row("Models", models_str)
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)
        table.add_row("Workers", str(self.num_workers or "auto-detect"))
        table.add_row("GPUs per Worker", str(self.gpus_per_worker))

        if self.num_shots_override is not None:
            table.add_row("Num Shots Override", str(self.num_shots_override))
        if self.limit_override is not None:
            table.add_row("Limit Override", str(self.limit_override))

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

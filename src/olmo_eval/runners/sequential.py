"""Evaluation runner orchestrator."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from olmo_eval.backends import Backend, BackendType, create_backend
from olmo_eval.core import expand_tasks, get_model_config
from olmo_eval.evals.suites import suite_exists
from olmo_eval.evals.tasks import list_tasks
from olmo_eval.evals.tasks.registry import list_regimes
from olmo_eval.runners.utils import run_task_impl

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation of runner inputs fails."""

    pass


@dataclass
class EvalRunner:
    """Orchestrates evaluation runs across tasks."""

    model_name: str
    task_specs: list[str]
    output_dir: str = "./results"
    num_shots_override: int | None = None
    limit_override: int | None = None
    backend_override: str | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    def validate(self) -> None:
        """Validate all inputs before running.

        Raises:
            ValidationError: If any task specs are invalid.
        """
        errors: list[str] = []

        # Validate task specs
        available_tasks = set(list_tasks())
        regimes_by_task = list_regimes()

        for spec in self.task_specs:
            # Check if it's a suite
            if suite_exists(spec):
                continue

            # Parse task::regime format
            task_name, _, regime = spec.partition("::")

            if task_name not in available_tasks:
                errors.append(f"Unknown task or suite: '{spec}'")
                continue

            # If regime specified, validate it exists
            if regime:
                task_regimes = regimes_by_task.get(task_name, [])
                if regime not in task_regimes:
                    if task_regimes:
                        errors.append(
                            f"Unknown regime '{regime}' for task '{task_name}'. "
                            f"Available: {', '.join(task_regimes)}"
                        )
                    else:
                        errors.append(
                            f"Unknown regime '{regime}' for task '{task_name}'. "
                            f"This task has no registered regimes."
                        )

        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print the resolved configuration without running."""
        table = Table(title="Run Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        model_config = get_model_config(self.model_name)
        backend_str = self.backend_override or model_config.backend

        table.add_row("Model", model_config.model)
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
        model_config = get_model_config(self.model_name)

        # Determine backend (model_config.backend is a string)
        backend_str = self.backend_override or model_config.backend
        backend_type = BackendType(backend_str)

        console.print(f"[bold]Initializing {backend_type.value} backend...[/bold]")
        backend = create_backend(
            backend_type,
            model_config.model,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
            dtype=model_config.dtype,
            **model_config.extra_args,
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
            task_results = self._run_task(spec, backend)
            results["tasks"][spec] = task_results

            # Log metrics (for Beaker job details)
            if "metrics" in task_results:
                logger.info(f"** Task metrics for {spec}: **")
                for metric, value in task_results["metrics"].items():
                    logger.info(f"  {metric}: {value:.4f}")
                    console.print(f"  {metric}: {value:.4f}")

        # Log summary of all scores
        self._log_summary(results)
        self._save_results(results)
        return results

    def _run_task(self, spec: str, backend: Backend) -> dict[str, Any]:
        """Run a single task and return results."""
        # Build overrides from instance settings
        overrides = {}
        if self.num_shots_override is not None:
            overrides["num_fewshot"] = self.num_shots_override
        if self.limit_override is not None:
            overrides["limit"] = self.limit_override

        # Use shared task execution logic
        result = run_task_impl(
            spec=spec,
            backend=backend,
            overrides=overrides or None,
            progress_callback=lambda msg: console.print(f"  {msg}"),
        )

        # Check for errors
        if result.error:
            raise RuntimeError(f"Task {spec} failed: {result.error}")

        return {
            "config": result.config,
            "num_instances": result.num_instances,
            "metrics": result.metrics,
        }

    def _log_summary(self, results: dict[str, Any]) -> None:
        """Log summary of all task scores."""
        logger.info("Summary of primary scores:")
        for task_name, task_data in results["tasks"].items():
            metrics = task_data.get("metrics", {})
            if metrics:
                # Use first metric as primary score
                primary_score = next(iter(metrics.values()))
                logger.info(f"  {task_name}: {primary_score:.4f}")

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

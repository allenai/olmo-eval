"""Evaluation runner orchestrator."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table

from olmo_eval.backends import Backend, BackendType, create_backend
from olmo_eval.core import Response, expand_tasks, get_model_config
from olmo_eval.evals.suites import suite_exists
from olmo_eval.tasks import get_task, list_tasks
from olmo_eval.tasks.registry import list_regimes

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()


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
    storage: StorageBackend | None = None

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

            # Print metrics
            if "metrics" in task_results:
                for metric, value in task_results["metrics"].items():
                    console.print(f"  {metric}: {value:.4f}")

        self._save_results(results)
        return results

    def _run_task(self, spec: str, backend: Backend) -> dict[str, Any]:
        """Run a single task and return results."""
        task = get_task(spec)

        # Apply overrides
        if self.num_shots_override is not None or self.limit_override is not None:
            overrides = {}
            if self.num_shots_override is not None:
                overrides["num_fewshot"] = self.num_shots_override
            if self.limit_override is not None:
                overrides["limit"] = self.limit_override
            task.config = replace(task.config, **overrides)

        # Collect instances
        instances = list(task.instances)
        if task.config.limit:
            instances = instances[: task.config.limit]

        console.print(f"  Evaluating {len(instances)} instances...")

        # Format requests
        requests = [task.format_request(inst) for inst in instances]

        # Generate outputs
        outputs = backend.generate(requests)

        # Build responses
        responses = [
            Response(instance=inst, request=req, outputs=out)
            for inst, req, out in zip(instances, requests, outputs, strict=True)
        ]

        # Score and compute metrics
        scored = task.score_responses(responses)
        metrics = task.compute_metrics(scored)

        return {
            "config": {
                "name": task.config.name,
                "split": task.config.split.value,
                "num_fewshot": task.config.num_fewshot,
                "limit": task.config.limit,
            },
            "num_instances": len(instances),
            "metrics": metrics,
        }

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save results using storage backend or legacy file output."""
        if self.storage is not None:
            # Use the storage backend
            from olmo_eval.storage.file import convert_runner_results

            run_id = str(uuid.uuid4())
            eval_result = convert_runner_results(results, run_id)
            self.storage.save(eval_result)
            console.print(f"\n[green]Results saved (run_id: {run_id})[/green]")
        else:
            # Legacy behavior: save to JSON file directly
            output_path = Path(self.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_slug = self.model_name.replace("/", "_").replace(".", "-")
            filename = f"{model_slug}_{timestamp}.json"

            filepath = output_path / filename
            with open(filepath, "w") as f:
                json.dump(results, f, indent=2)

            console.print(f"\n[green]Results saved to {filepath}[/green]")

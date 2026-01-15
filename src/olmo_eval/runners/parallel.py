"""Async evaluation runner for parallel task execution."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.backends import BackendType, create_backend
from olmo_eval.core import expand_tasks, get_model_config
from olmo_eval.runners.sequential import ValidationError
from olmo_eval.runners.utils import TaskResult, run_task_impl

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


def worker_process(
    worker_id: int,
    gpu_ids: list[int],
    task_queue: mp.Queue[tuple[str, str, dict] | None],
    result_queue: mp.Queue[TaskResult],
    overrides: dict[str, Any],
) -> None:
    """Worker process that executes tasks from queue.

    Args:
        worker_id: Worker identifier
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        task_queue: Queue of (task_spec, model_name, backend_type) tuples
        result_queue: Queue to put results
        overrides: Task overrides (num_fewshot, limit)
    """
    # Set CUDA devices for this worker
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in gpu_ids)

    backend = None
    current_model = None

    while True:
        item = task_queue.get()
        if item is None:  # Poison pill
            break

        task_spec, model_name, backend_type_str = item

        # Load backend if needed (reuse if same model)
        if backend is None or current_model != model_name:
            if backend is not None:
                del backend

            backend_type = BackendType(backend_type_str)
            backend = create_backend(backend_type, model_name)
            current_model = model_name

        # Execute task
        result = run_task_impl(
            spec=task_spec,
            backend=backend,
            overrides=overrides or None,
        )
        result_queue.put(result)


@dataclass
class AsyncEvalRunner:
    """Async evaluation runner with parallel task execution.

    Runs tasks in parallel across multiple GPUs/compute instances using a simple
    queue-based approach. Tasks are pulled from a queue by worker processes.
    """

    model_name: str
    task_specs: list[str]
    output_dir: str = "./results"
    num_shots_override: int | None = None
    limit_override: int | None = None
    backend_override: str | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    # Async-specific config
    num_workers: int | None = None  # Number of workers (default: num GPUs)
    gpus_per_worker: int = 1  # Number of GPUs each worker uses

    def validate(self) -> None:
        """Validate configuration."""
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks import list_tasks
        from olmo_eval.evals.tasks.registry import list_regimes

        if not self.model_name:
            raise ValidationError("model_name is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs
        errors: list[str] = []
        available_tasks = set(list_tasks())
        regimes_by_task = list_regimes()

        for spec in self.task_specs:
            if suite_exists(spec):
                continue

            task_name, _, regime = spec.partition("::")

            if task_name not in available_tasks:
                errors.append(f"Unknown task or suite: '{spec}'")
                continue

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
        """Print configuration."""
        from rich.table import Table

        table = Table(title="Run Configuration (Async Mode)")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        model_config = get_model_config(self.model_name)
        backend_str = self.backend_override or model_config.backend

        table.add_row("Model", model_config.model)
        table.add_row("Backend", backend_str)
        table.add_row("Mode", "Async (Parallel Execution)")
        table.add_row("Output Dir", self.output_dir)
        table.add_row("Workers", str(self.num_workers or "auto-detect"))
        table.add_row("GPUs per Worker", str(self.gpus_per_worker))

        if self.num_shots_override is not None:
            table.add_row("Num Shots Override", str(self.num_shots_override))
        if self.limit_override is not None:
            table.add_row("Limit Override", str(self.limit_override))

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        console.print(f"\n[bold]Tasks to run:[/bold] {len(expanded)}")
        for spec in expanded:
            console.print(f"  • {spec}")

    def _get_num_workers(self) -> int:
        """Get number of workers."""
        if self.num_workers is not None:
            return self.num_workers

        # Auto-detect GPUs
        try:
            import torch  # type: ignore[import-untyped]

            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                raise RuntimeError("No GPUs detected. Specify --num-workers explicitly.")
            return num_gpus // self.gpus_per_worker
        except ImportError:
            raise RuntimeError("torch not available. Specify --num-workers explicitly.") from None

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations asynchronously."""
        # Get model config
        model_config = get_model_config(self.model_name)
        if self.backend_override:
            model_config.backend = self.backend_override

        backend_type = BackendType(model_config.backend)

        # Expand tasks
        expanded_tasks = expand_tasks(self.task_specs)

        # Determine number of workers
        num_workers = self._get_num_workers()

        console.print(f"[bold]Model:[/bold] {model_config.model}")
        console.print(f"[bold]Backend:[/bold] {backend_type.value}")
        console.print(f"[bold]Tasks:[/bold] {len(expanded_tasks)}")
        console.print(f"[bold]Workers:[/bold] {num_workers}")
        console.print(f"[bold]GPUs per worker:[/bold] {self.gpus_per_worker}")

        # Create queues
        ctx = mp.get_context("spawn")
        task_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()

        # Build overrides
        overrides = {}
        if self.num_shots_override is not None:
            overrides["num_fewshot"] = self.num_shots_override
        if self.limit_override is not None:
            overrides["limit"] = self.limit_override

        # Enqueue all tasks
        for task_spec in expanded_tasks:
            task_queue.put((task_spec, self.model_name, backend_type.value))

        # Add poison pills
        for _ in range(num_workers):
            task_queue.put(None)

        # Assign GPUs to workers
        try:
            import torch  # type: ignore[import-untyped]

            total_gpus = torch.cuda.device_count()
        except ImportError:
            total_gpus = 0

        # Spawn workers
        workers = []
        for i in range(num_workers):
            # Assign GPUs to this worker
            if total_gpus > 0:
                start_gpu = i * self.gpus_per_worker
                gpu_ids = list(range(start_gpu, min(start_gpu + self.gpus_per_worker, total_gpus)))
            else:
                gpu_ids = []

            process = ctx.Process(
                target=worker_process,
                args=(i, gpu_ids, task_queue, result_queue, overrides),
            )
            process.start()
            workers.append(process)

        console.print("[bold green]Workers started, processing tasks...[/bold green]")

        # Collect results
        results = []
        for _ in range(len(expanded_tasks)):
            result = await asyncio.get_event_loop().run_in_executor(None, result_queue.get)
            results.append(result)

            status = "red" if result.error else "green"
            msg = f"ERROR: {result.error}" if result.error else f"{result.num_instances} instances"
            console.print(f"  [{status}]✓[/] {result.spec} ({msg})")

            # Log metrics (for Beaker job details)
            if result.error is None and result.metrics:
                logger.info(f"** Task metrics for {result.spec}: **")
                for metric, value in result.metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")

        # Wait for workers
        for process in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()

        # Check for errors
        errors = [r for r in results if r.error]
        if errors:
            console.print(f"\n[bold red]Errors:[/bold red] {len(errors)} tasks failed")
            for error_result in errors:
                console.print(f"  - {error_result.spec}: {error_result.error}")

        # Aggregate results
        results_dict = {
            "model": model_config.model,
            "backend": backend_type.value,
            "timestamp": datetime.now().isoformat(),
            "tasks": {
                r.spec: {
                    "config": r.config,
                    "num_instances": r.num_instances,
                    "metrics": r.metrics,
                }
                for r in results
                if r.error is None
            },
            "errors": [{"spec": r.spec, "error": r.error} for r in results if r.error is not None],
        }

        # Log summary of all scores
        self._log_summary(results_dict)

        # Save results
        self._save_results(results_dict)

        return results_dict

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

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

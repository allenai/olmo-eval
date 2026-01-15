"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import queue
import random
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.backends import BackendType, create_backend
from olmo_eval.core import Instance, LMOutput, LMRequest, Response, expand_tasks, get_model_config
from olmo_eval.evals.tasks import get_task
from olmo_eval.evals.tasks.base import Task
from olmo_eval.runners.sequential import ValidationError
from olmo_eval.runners.utils import TaskResult

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data structures for instance-level queuing
# -----------------------------------------------------------------------------


@dataclass
class QueueItem:
    """Single instance ready for generation."""

    task_id: str  # Task spec string
    instance_idx: int  # Index within task's instance list
    instance: Instance
    request: LMRequest  # Pre-formatted request
    attempt: int = 0  # Retry attempt number


@dataclass
class TaskTracker:
    """Tracks completion state for a single task."""

    spec: str
    task: Task | None  # None if task prep failed
    total_instances: int
    completed_count: int = 0
    responses: dict[int, Response] = field(default_factory=dict)
    error: str | None = None
    start_time: float = field(default_factory=time.time)

    def is_complete(self) -> bool:
        """Check if task is complete (all instances done or error occurred)."""
        return self.completed_count >= self.total_instances or self.error is not None

    def add_response(self, idx: int, response: Response) -> bool:
        """Add a response. Returns True if task is now complete."""
        self.responses[idx] = response
        self.completed_count += 1
        return self.is_complete()


@dataclass
class ResultItem:
    """Result for a single instance from the worker."""

    task_id: str
    instance_idx: int
    instance: Instance
    request: LMRequest
    outputs: list[LMOutput]
    error: str | None = None
    attempt: int = 0


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def prepare_task_items(
    spec: str,
    overrides: dict[str, Any] | None,
) -> tuple[Task, list[QueueItem]]:
    """Prepare a task and its queue items.

    Args:
        spec: Task specification string
        overrides: Optional config overrides (num_fewshot, limit)

    Returns:
        Tuple of (Task instance for scoring, list of QueueItems)
    """
    task = get_task(spec)

    if overrides:
        task.config = replace(task.config, **overrides)

    instances = list(task.instances)
    if task.config.limit:
        instances = instances[: task.config.limit]

    items = [
        QueueItem(
            task_id=spec,
            instance_idx=idx,
            instance=inst,
            request=task.format_request(inst),
        )
        for idx, inst in enumerate(instances)
    ]

    return task, items


def finalize_task(tracker: TaskTracker) -> TaskResult:
    """Score responses and compute metrics for a completed task.

    Args:
        tracker: TaskTracker with all responses collected

    Returns:
        TaskResult with metrics or error
    """
    duration = time.time() - tracker.start_time

    if tracker.error:
        return TaskResult(
            spec=tracker.spec,
            config={},
            num_instances=0,
            metrics={},
            error=tracker.error,
            duration_seconds=duration,
        )

    if tracker.task is None:
        return TaskResult(
            spec=tracker.spec,
            config={},
            num_instances=0,
            metrics={},
            error="Task not initialized",
            duration_seconds=duration,
        )

    # Reconstruct responses in original order
    responses = [tracker.responses[i] for i in range(tracker.total_instances)]

    # Score and compute metrics
    scored = tracker.task.score_responses(responses)
    metrics = tracker.task.compute_metrics(scored)

    return TaskResult(
        spec=tracker.spec,
        config={
            "name": tracker.task.config.name,
            "split": tracker.task.config.split.value,
            "num_fewshot": tracker.task.config.num_fewshot,
            "limit": tracker.task.config.limit,
        },
        num_instances=tracker.total_instances,
        metrics=metrics,
        duration_seconds=duration,
    )


# -----------------------------------------------------------------------------
# Worker process
# -----------------------------------------------------------------------------


def _process_batch(
    batch: list[QueueItem],
    backend: Any,
    result_queue: mp.Queue,
) -> None:
    """Process a batch of instances through the backend.

    Args:
        batch: List of QueueItems to process
        backend: Backend instance
        result_queue: Queue to put results
    """
    requests = [item.request for item in batch]

    try:
        outputs_list = backend.generate(requests)

        for item, outputs in zip(batch, outputs_list, strict=True):
            result_queue.put(
                ResultItem(
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=item.request,
                    outputs=outputs,
                    error=None,
                    attempt=item.attempt,
                )
            )
    except Exception as e:
        # On batch failure, report error for all items
        for item in batch:
            result_queue.put(
                ResultItem(
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=item.request,
                    outputs=[],
                    error=str(e),
                    attempt=item.attempt,
                )
            )


def instance_worker_process(
    gpu_ids: list[int],
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    backend_type_str: str,
    batch_size: int = 32,
    batch_timeout: float = 0.1,
) -> None:
    """Worker that processes instances in batches.

    Args:
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        instance_queue: Queue of QueueItems (None = poison pill)
        result_queue: Queue to put ResultItems
        model_name: Model name for backend
        backend_type_str: Backend type string
        batch_size: Maximum batch size
        batch_timeout: Seconds to wait for more items when batching
    """
    if gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    backend_type = BackendType(backend_type_str)
    backend = create_backend(backend_type, model_name)

    while True:
        # Collect batch
        batch: list[QueueItem] = []
        deadline = time.time() + batch_timeout

        while len(batch) < batch_size:
            try:
                remaining = max(0.001, deadline - time.time())
                # Block indefinitely for first item, then use timeout
                timeout = None if not batch else remaining
                item = instance_queue.get(timeout=timeout)
            except queue.Empty:
                break

            if item is None:  # Poison pill
                # Process remaining batch, then exit
                if batch:
                    _process_batch(batch, backend, result_queue)
                return

            batch.append(item)

        if batch:
            _process_batch(batch, backend, result_queue)


# -----------------------------------------------------------------------------
# AsyncEvalRunner
# -----------------------------------------------------------------------------


@dataclass
class AsyncEvalRunner:
    """Async evaluation runner with instance-level queuing.

    Uses a unified queue where instances from all tasks are mixed together,
    enabling better GPU utilization and early completion reporting.
    """

    model_name: str
    task_specs: list[str]
    output_dir: str = "./results"
    num_shots_override: int | None = None
    limit_override: int | None = None
    backend_override: str | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    # Instance queue config
    batch_size: int = 32
    max_retries: int = 3

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
        table.add_row("Mode", "Async (Instance Queue)")
        table.add_row("Output Dir", self.output_dir)
        table.add_row("Batch Size", str(self.batch_size))
        table.add_row("Max Retries", str(self.max_retries))

        if self.num_shots_override is not None:
            table.add_row("Num Shots Override", str(self.num_shots_override))
        if self.limit_override is not None:
            table.add_row("Limit Override", str(self.limit_override))

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        console.print(f"\n[bold]Tasks to run:[/bold] {len(expanded)}")
        for spec in expanded:
            console.print(f"  - {spec}")

    def _get_gpu_ids(self) -> list[int]:
        """Get available GPU IDs."""
        try:
            import torch

            num_gpus = torch.cuda.device_count()
            return list(range(num_gpus))
        except ImportError:
            return []

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using instance-level queuing."""
        # Get model config
        model_config = get_model_config(self.model_name)
        if self.backend_override:
            model_config.backend = self.backend_override

        backend_type = BackendType(model_config.backend)
        expanded_tasks = expand_tasks(self.task_specs)

        # Build overrides
        overrides: dict[str, Any] = {}
        if self.num_shots_override is not None:
            overrides["num_fewshot"] = self.num_shots_override
        if self.limit_override is not None:
            overrides["limit"] = self.limit_override

        # Prepare all tasks and collect items
        trackers: dict[str, TaskTracker] = {}
        all_items: list[QueueItem] = []

        console.print(f"[bold]Model:[/bold] {model_config.model}")
        console.print(f"[bold]Backend:[/bold] {backend_type.value}")
        console.print(f"[bold]Tasks:[/bold] {len(expanded_tasks)}")
        console.print("[bold]Preparing tasks...[/bold]")

        for spec in expanded_tasks:
            try:
                task, items = prepare_task_items(spec, overrides or None)
                trackers[spec] = TaskTracker(
                    spec=spec,
                    task=task,
                    total_instances=len(items),
                )
                all_items.extend(items)
                console.print(f"  - {spec}: {len(items)} instances")
            except Exception as e:
                trackers[spec] = TaskTracker(
                    spec=spec,
                    task=None,
                    total_instances=0,
                    error=str(e),
                )
                console.print(f"  [red]- {spec}: ERROR - {e}[/red]")

        total_instances = len(all_items)
        console.print(f"\n[bold]Total instances:[/bold] {total_instances}")

        # Shuffle for better mixing across tasks
        random.shuffle(all_items)

        # Create queues
        ctx = mp.get_context("spawn")
        instance_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()

        # Enqueue all items
        for item in all_items:
            instance_queue.put(item)
        instance_queue.put(None)  # Poison pill

        # Start worker
        gpu_ids = self._get_gpu_ids()
        worker = ctx.Process(
            target=instance_worker_process,
            args=(
                gpu_ids,
                instance_queue,
                result_queue,
                self.model_name,
                backend_type.value,
                self.batch_size,
            ),
        )
        worker.start()

        console.print("[bold green]Worker started, processing instances...[/bold green]")

        # Track results
        results: list[TaskResult] = []
        completed_tasks = 0
        total_tasks = len(expanded_tasks)

        # Pre-add error tasks to results
        for _spec, tracker in trackers.items():
            if tracker.error:
                task_result = finalize_task(tracker)
                results.append(task_result)
                completed_tasks += 1
                self._report_task_completion(task_result)

        # Track pending retries
        pending_instances = total_instances

        processed = 0
        while completed_tasks < total_tasks and pending_instances > 0:
            result_item: ResultItem = await asyncio.get_event_loop().run_in_executor(
                None, result_queue.get
            )
            processed += 1

            tracker = trackers[result_item.task_id]

            # Skip if task already failed
            if tracker.error:
                pending_instances -= 1
                continue

            if result_item.error:
                # Instance error - retry or fail task
                if result_item.attempt < self.max_retries:
                    # Re-enqueue with incremented attempt
                    retry_item = QueueItem(
                        task_id=result_item.task_id,
                        instance_idx=result_item.instance_idx,
                        instance=result_item.instance,
                        request=result_item.request,
                        attempt=result_item.attempt + 1,
                    )
                    instance_queue.put(retry_item)
                    logger.warning(
                        f"Retrying instance {result_item.instance_idx} of {result_item.task_id} "
                        f"(attempt {result_item.attempt + 1}/{self.max_retries})"
                    )
                else:
                    # Retries exhausted - fail task
                    tracker.error = (
                        f"Instance {result_item.instance_idx} failed after "
                        f"{self.max_retries} retries: {result_item.error}"
                    )
                    pending_instances -= 1
                    if tracker.is_complete():
                        task_result = finalize_task(tracker)
                        results.append(task_result)
                        completed_tasks += 1
                        self._report_task_completion(task_result)
            else:
                # Success - add response
                response = Response(
                    instance=result_item.instance,
                    request=result_item.request,
                    outputs=result_item.outputs,
                )

                is_complete = tracker.add_response(result_item.instance_idx, response)
                pending_instances -= 1

                if is_complete:
                    task_result = finalize_task(tracker)
                    results.append(task_result)
                    completed_tasks += 1
                    self._report_task_completion(task_result)

            # Progress update
            if processed % 100 == 0:
                console.print(
                    f"  Processed {processed} results, "
                    f"{completed_tasks}/{total_tasks} tasks complete"
                )

        # Send poison pill for any remaining retries and cleanup
        instance_queue.put(None)

        # Wait for worker
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()
            worker.join()

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

    def _report_task_completion(self, result: TaskResult) -> None:
        """Report when a task completes."""
        if result.error:
            console.print(f"  [red]x[/red] {result.spec} (ERROR: {result.error})")
        else:
            console.print(
                f"  [green]v[/green] {result.spec} ({result.num_instances} instances, "
                f"{result.duration_seconds:.1f}s)"
            )
            # Log metrics
            if result.metrics:
                logger.info(f"** Task metrics for {result.spec}: **")
                for metric, value in result.metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")

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

"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.backends import BackendType, create_backend
from olmo_eval.core import (
    Instance,
    LMOutput,
    LMRequest,
    Response,
    SamplingParams,
    expand_tasks,
    get_model_config,
)
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.evals.tasks import Task, get_task
from olmo_eval.runners.synchronous import ValidationError
from olmo_eval.runners.utils import (
    TaskResult,
    build_predictions,
    compute_suite_aggregations,
    get_primary_metric,
    serialize_sampling_params,
    write_predictions_jsonl,
    write_requests_jsonl,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = logging.getLogger(__name__)


# Keys that apply to TaskConfig
TASKCONFIG_KEYS = {"num_fewshot", "limit", "fewshot_seed"}

# Keys that apply to SamplingParams
SAMPLING_KEYS = {"temperature", "max_tokens", "top_p", "top_k", "num_samples"}


# -----------------------------------------------------------------------------
# Data structures for instance-level queuing
# -----------------------------------------------------------------------------


@dataclass
class QueueItem:
    """Single instance ready for generation."""

    model_name: str  # Which model this is for
    task_id: str  # Task spec string
    instance_idx: int  # Index within task's instance list
    instance: Instance
    request: LMRequest  # Pre-formatted request
    sampling_params: SamplingParams | None = None
    attempt: int = 0  # Retry attempt number


@dataclass
class TaskTracker:
    """Tracks completion state for a single (model, task) pair."""

    model_name: str  # Which model this is for
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

    model_name: str  # Which model produced this result
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


def build_requests_from_items(
    items: list[QueueItem],
    task_name: str,
) -> list[dict]:
    """Build request objects from QueueItems for early writing.

    Args:
        items: List of QueueItems with instance and request data
        task_name: Name of the task

    Returns:
        List of request dicts suitable for JSONL output
    """
    from olmo_eval.runners.utils import build_requests

    if not items:
        return []

    instances = [item.instance for item in items]
    requests = [item.request for item in items]
    sampling_params = items[0].sampling_params if items else None

    return build_requests(instances, requests, task_name, sampling_params)


def prepare_task_items(
    spec: str,
    model_name: str,
    overrides: dict[str, Any] | None,
    temperature: float | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> tuple[Task, list[QueueItem]]:
    """Prepare a task and its queue items.

    Args:
        spec: Task specification string
        model_name: Model name this task is for
        overrides: Optional config overrides (num_fewshot, limit, fewshot_seed)
        temperature: Optional temperature for sampling (deprecated, use sampling_overrides)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)

    Returns:
        Tuple of (Task instance for scoring, list of QueueItems)
    """
    task = get_task(spec)

    if overrides:
        task.config = replace(task.config, **overrides)

    # Build sampling params from overrides
    # Priority: sampling_overrides > temperature > task default
    existing_params = task.config.sampling_params or SamplingParams()

    # Apply legacy temperature parameter (deprecated)
    if temperature is not None:
        existing_params = replace(existing_params, temperature=temperature)

    # Apply sampling_overrides (highest priority)
    if sampling_overrides:
        for key, value in sampling_overrides.items():
            if hasattr(existing_params, key):
                existing_params = replace(existing_params, **{key: value})

    # Always update task config with final sampling params (so finalize_task captures them)
    task.config = replace(task.config, sampling_params=existing_params)

    instances = list(task.instances)
    if task.config.limit:
        instances = instances[: task.config.limit]

    items = [
        QueueItem(
            model_name=model_name,
            task_id=spec,
            instance_idx=idx,
            instance=inst,
            request=task.format_request(inst),
            sampling_params=existing_params,
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

    # Build predictions for per-instance inspection
    predictions = build_predictions(scored)

    # Extract primary metric name from task config if specified
    primary_metric_name = None
    if tracker.task.config.primary_metric:
        primary_metric_name = tracker.task.config.primary_metric.name

    return TaskResult(
        spec=tracker.spec,
        config={
            "name": tracker.task.config.name,
            "split": tracker.task.config.split.value,
            "num_fewshot": tracker.task.config.num_fewshot,
            "fewshot_seed": tracker.task.config.fewshot_seed,
            "limit": tracker.task.config.limit,
            "sampling_params": serialize_sampling_params(tracker.task.config.sampling_params),
        },
        num_instances=tracker.total_instances,
        metrics=metrics,
        duration_seconds=duration,
        predictions=predictions,
        primary_metric=primary_metric_name,
    )


# -----------------------------------------------------------------------------
# Worker health monitoring
# -----------------------------------------------------------------------------


def check_workers_alive(
    workers: list[mp.process.BaseProcess],
    result_queue: mp.Queue,
    timeout: float = 0.1,
) -> None:
    """Check if workers are alive and handle any fatal errors in the queue.

    Args:
        workers: List of worker processes
        result_queue: Queue to check for fatal error markers
        timeout: How long to wait for queue items

    Raises:
        RuntimeError: If all workers are dead or a fatal error is found
    """
    # Check for fatal errors in queue (non-blocking)
    try:
        while True:
            result_item = result_queue.get_nowait()
            if result_item.task_id == "__WORKER_FATAL__":
                console.print("\n[bold red]FATAL: Worker crashed![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
                # Terminate all workers
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                # Cancel queue join thread to allow clean process exit
                result_queue.cancel_join_thread()
                raise RuntimeError(f"Worker process crashed: {result_item.error}")
            else:
                # Put non-fatal item back (this is rare but handle it)
                result_queue.put(result_item)
                break
    except Exception:
        pass  # Queue empty, continue

    # Check if all workers are dead
    alive_count = sum(1 for w in workers if w.is_alive())
    if alive_count == 0:
        # All workers dead - check exit codes
        exit_codes = [w.exitcode for w in workers]
        if any(code != 0 and code is not None for code in exit_codes):
            raise RuntimeError(f"All workers died unexpectedly. Exit codes: {exit_codes}")


def wait_for_workers_ready(
    workers: list[mp.process.BaseProcess],
    result_queue: mp.Queue,
    startup_timeout: float = 30.0,
) -> None:
    """Wait briefly for workers to start and check for early failures.

    Args:
        workers: List of worker processes
        result_queue: Queue to check for fatal error markers
        startup_timeout: How long to wait for workers to stabilize

    Raises:
        RuntimeError: If workers fail during startup
    """
    # Give workers a moment to initialize and potentially fail
    start_time = time.time()
    check_interval = 0.5

    while time.time() - start_time < startup_timeout:
        time.sleep(check_interval)

        # Check for fatal errors
        try:
            result_item = result_queue.get_nowait()
            if result_item.task_id == "__WORKER_FATAL__":
                console.print("\n[bold red]FATAL: Worker failed during startup![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
                # Terminate all workers
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                # Cancel queue join thread to allow clean process exit
                result_queue.cancel_join_thread()
                raise RuntimeError(f"Worker failed during startup: {result_item.error}")
            else:
                # Put non-fatal item back
                result_queue.put(result_item)
        except Exception:
            pass  # Queue empty

        # Check if any worker died with non-zero exit code
        for worker in workers:
            if not worker.is_alive() and worker.exitcode is not None and worker.exitcode != 0:
                raise RuntimeError(f"Worker died during startup with exit code {worker.exitcode}")

        # If all workers are alive, we're good
        if all(w.is_alive() for w in workers):
            return

    # Final check
    check_workers_alive(workers, result_queue)


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
    from olmo_eval.core import RequestType

    requests = [item.request for item in batch]
    sampling_params = batch[0].sampling_params if batch else None

    try:
        # Use logprobs for LOGLIKELIHOOD requests (e.g., BPB tasks)
        if requests and requests[0].request_type == RequestType.LOGLIKELIHOOD:
            outputs_list = backend.logprobs(requests)
        else:
            outputs_list = backend.generate(requests, sampling_params)

        for item, outputs in zip(batch, outputs_list, strict=True):
            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
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
                    model_name=item.model_name,
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
    attention_backend: str | None = None,
    tokenizer: str | None = None,
    max_model_len: int | None = None,
    load_format: str | None = None,
    extra_loader_config: dict[str, Any] | None = None,
) -> None:
    """Worker that collects all items and processes them at once.

    Collects all items from the queue, then processes them in a single
    backend call for maximum throughput. vLLM handles internal batching.

    Args:
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        instance_queue: Queue of QueueItems (None = poison pill)
        result_queue: Queue to put ResultItems
        model_name: Model name for backend
        backend_type_str: Backend type string
        attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN")
        tokenizer: Tokenizer path/identifier, defaults to model if None
        max_model_len: Maximum model context length (overrides model's default)
        load_format: vLLM model loading format (e.g., "runai_streamer")
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true})
    """
    import sys

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        backend_type = BackendType(backend_type_str)
        # Pass tensor_parallel_size for vLLM to use all assigned GPUs
        engine_kwargs: dict[str, Any] = {"tensor_parallel_size": len(gpu_ids)} if gpu_ids else {}
        if attention_backend:
            engine_kwargs["attention_backend"] = attention_backend
        if max_model_len:
            engine_kwargs["max_model_len"] = max_model_len
        if load_format:
            engine_kwargs["load_format"] = load_format
        if extra_loader_config:
            engine_kwargs["model_loader_extra_config"] = extra_loader_config
        backend = create_backend(backend_type, model_name, tokenizer=tokenizer, **engine_kwargs)

        # Collect all items from queue
        items: list[QueueItem] = []
        while True:
            item = instance_queue.get()
            if item is None:  # Poison pill
                break
            items.append(item)

        # Process all items at once - vLLM handles internal batching
        if items:
            _process_batch(items, backend, result_queue)
    except Exception as e:
        logger.error(f"Worker process failed: {e}")
        # Put a fatal error marker in the result queue so main process knows we died
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id="__WORKER_FATAL__",
                instance_idx=-1,
                instance=None,  # type: ignore[arg-type]
                request=None,  # type: ignore[arg-type]
                outputs=[],
                error=f"Worker process crashed: {e}",
            )
        )
        sys.exit(1)


def streaming_worker_process(
    gpu_ids: list[int],
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    attention_backend: str | None = None,
    tokenizer: str | None = None,
    max_model_len: int | None = None,
    load_format: str | None = None,
    extra_loader_config: dict[str, Any] | None = None,
) -> None:
    """Worker using async streaming for true continuous batching.

    Unlike the batch worker, this worker uses vLLM's AsyncLLMEngine to:
    1. Add requests continuously as they arrive
    2. Stream results back as they complete
    3. Enable true continuous batching for optimal throughput

    Args:
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        instance_queue: Queue of QueueItems (None = poison pill)
        result_queue: Queue to put ResultItems
        model_name: Model name for backend
        attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN")
        tokenizer: Tokenizer path/identifier, defaults to model if None
        max_model_len: Maximum model context length (overrides model's default)
        load_format: vLLM model loading format (e.g., "runai_streamer")
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true})
    """
    import sys

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        # Run the async worker
        num_gpus = len(gpu_ids) if gpu_ids else 1
        asyncio.run(
            _streaming_worker_async(
                instance_queue,
                result_queue,
                model_name,
                num_gpus,
                attention_backend,
                tokenizer,
                max_model_len,
                load_format,
                extra_loader_config,
            )
        )
    except Exception as e:
        logger.error(f"Streaming worker process failed: {e}")
        # Put a fatal error marker in the result queue so main process knows we died
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id="__WORKER_FATAL__",
                instance_idx=-1,
                instance=None,  # type: ignore[arg-type]
                request=None,  # type: ignore[arg-type]
                outputs=[],
                error=f"Streaming worker process crashed: {e}",
            )
        )
        sys.exit(1)


async def _streaming_worker_async(
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    num_gpus: int = 1,
    attention_backend: str | None = None,
    tokenizer: str | None = None,
    max_model_len: int | None = None,
    load_format: str | None = None,
    extra_loader_config: dict[str, Any] | None = None,
) -> None:
    """Async implementation of streaming worker.

    Uses AsyncVLLMBackend for true continuous batching with streaming results.
    """
    from olmo_eval.backends.vllm import AsyncVLLMBackend

    engine_kwargs: dict[str, Any] = {"tensor_parallel_size": num_gpus}
    if max_model_len:
        engine_kwargs["max_model_len"] = max_model_len
    if load_format:
        engine_kwargs["load_format"] = load_format
    if extra_loader_config:
        engine_kwargs["model_loader_extra_config"] = extra_loader_config

    backend = AsyncVLLMBackend(
        model_name,
        tokenizer=tokenizer,
        attention_backend=attention_backend,
        **engine_kwargs,
    )

    # Track request_id -> QueueItem mapping
    pending_requests: dict[str, QueueItem] = {}
    request_counter = 0
    all_items_received = False
    items_added = 0
    results_received = 0

    async def add_requests() -> None:
        """Coroutine to continuously add requests from the queue."""
        nonlocal all_items_received, request_counter, items_added

        loop = asyncio.get_event_loop()
        while True:
            # Non-blocking get from queue
            item = await loop.run_in_executor(None, instance_queue.get)

            if item is None:  # Poison pill
                all_items_received = True
                break

            # Generate request ID and track it
            request_counter += 1
            request_id = f"req-{request_counter}"
            pending_requests[request_id] = item
            items_added += 1

            # Add to engine (non-blocking)
            try:
                await backend.add_request(request_id, item.request)
            except Exception as e:
                # Report error immediately
                result_queue.put(
                    ResultItem(
                        model_name=item.model_name,
                        task_id=item.task_id,
                        instance_idx=item.instance_idx,
                        instance=item.instance,
                        request=item.request,
                        outputs=[],
                        error=str(e),
                        attempt=item.attempt,
                    )
                )
                del pending_requests[request_id]

    async def collect_results() -> None:
        """Coroutine to collect results as they stream back."""
        nonlocal results_received

        while pending_requests or not all_items_received:
            # Wait a bit if no pending requests yet
            if not pending_requests:
                await asyncio.sleep(0.01)
                continue

            try:
                async for request_id, outputs in backend.stream_results():
                    if request_id not in pending_requests:
                        continue

                    item = pending_requests.pop(request_id)
                    results_received += 1

                    result_queue.put(
                        ResultItem(
                            model_name=item.model_name,
                            task_id=item.task_id,
                            instance_idx=item.instance_idx,
                            instance=item.instance,
                            request=item.request,
                            outputs=outputs,
                            error=None,
                            attempt=item.attempt,
                        )
                    )

                    # Check if we're done
                    if all_items_received and not pending_requests:
                        return

            except Exception as e:
                # Handle engine errors - fail all pending requests
                logger.error(f"Streaming engine error: {e}")
                for _req_id, item in list(pending_requests.items()):
                    result_queue.put(
                        ResultItem(
                            model_name=item.model_name,
                            task_id=item.task_id,
                            instance_idx=item.instance_idx,
                            instance=item.instance,
                            request=item.request,
                            outputs=[],
                            error=str(e),
                            attempt=item.attempt,
                        )
                    )
                pending_requests.clear()
                return

    # Run both coroutines concurrently
    await asyncio.gather(add_requests(), collect_results())

    # Shutdown engine
    await backend.shutdown()


# -----------------------------------------------------------------------------
# AsyncEvalRunner
# -----------------------------------------------------------------------------


@dataclass
class AsyncEvalRunner:
    """Async evaluation runner with instance-level queuing.

    Uses per-model queues where instances from all tasks are mixed together,
    enabling better GPU utilization and early completion reporting.
    Supports multiple models in a single run, producing results for each
    unique (model, task) pair.
    """

    model_names: list[str]
    task_specs: list[str]
    output_dir: str = BEAKER_RESULT_DIR
    num_shots_override: int | None = None
    limit_override: int | None = None
    temperature: float | None = None
    backend_override: str | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    # Multi-worker config
    num_workers: int | None = None  # Total workers (distributed across models)
    gpus_per_worker: int = 1  # Number of GPUs each worker uses

    # vLLM config
    attention_backend: str | None = None  # e.g., "FLASHINFER", "FLASH_ATTN"

    # Per-task overrides from inline spec (e.g., task::temperature=0.6)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-model overrides from inline spec (e.g., model::tokenizer=..., model::load_format=...)
    # Maps model name -> overrides dict
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
        from olmo_eval.evals.tasks.core.registry import parse_task_spec

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

        table = Table(title="Run Configuration (Async Mode)")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        # Show all models
        models_str = ", ".join(self.model_names)
        table.add_row("Models", models_str)
        table.add_row("Mode", "Async (All-at-once)")
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

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using instance-level queuing with multi-model support.

        Creates per-model instance queues and a shared result queue. Workers for each
        model process instances and report to the shared queue. Results are reported
        immediately when each (model, task) pair completes.
        """
        expanded_tasks = expand_tasks(self.task_specs)

        # Build global overrides from CLI args
        global_overrides: dict[str, Any] = {}
        global_sampling_overrides: dict[str, Any] = {}

        if self.num_shots_override is not None:
            global_overrides["num_fewshot"] = self.num_shots_override
        if self.limit_override is not None:
            global_overrides["limit"] = self.limit_override
        if self.temperature is not None:
            global_sampling_overrides["temperature"] = self.temperature

        # Prepare all (model, task) pairs
        trackers: dict[tuple[str, str], TaskTracker] = {}
        model_items: dict[str, list[QueueItem]] = {m: [] for m in self.model_names}
        model_configs: dict[str, Any] = {}

        console.print(f"[bold]Models:[/bold] {len(self.model_names)}")
        console.print(f"[bold]Tasks:[/bold] {len(expanded_tasks)}")
        total_pairs = len(self.model_names) * len(expanded_tasks)
        console.print(f"[bold]Total (model, task) pairs:[/bold] {total_pairs}")

        # Get model configs with per-model overrides
        for model_name in self.model_names:
            overrides = self.model_overrides.get(model_name, {})
            model_config = get_model_config(model_name, **overrides)
            if self.backend_override:
                model_config.backend = self.backend_override
            model_configs[model_name] = model_config

        # Prepare tasks in parallel
        console.print(f"[bold]Preparing {total_pairs} tasks...[/bold]")

        def prepare_one(
            model_name: str, spec: str
        ) -> tuple[str, str, TaskTracker, list[QueueItem]]:
            try:
                # Build overrides for this task
                # 1. Start with global CLI overrides
                overrides = dict(global_overrides)
                sampling_overrides = dict(global_sampling_overrides)

                # 2. Apply per-task overrides (highest priority)
                task_specific = self.task_overrides.get(spec, {})
                for key, value in task_specific.items():
                    if key in TASKCONFIG_KEYS:
                        overrides[key] = value
                    elif key in SAMPLING_KEYS:
                        sampling_overrides[key] = value

                task, items = prepare_task_items(
                    spec,
                    model_name,
                    overrides or None,
                    sampling_overrides=sampling_overrides or None,
                )
                tracker = TaskTracker(
                    model_name=model_name,
                    spec=spec,
                    task=task,
                    total_instances=len(items),
                )
                return (model_name, spec, tracker, items)
            except Exception as e:
                tracker = TaskTracker(
                    model_name=model_name,
                    spec=spec,
                    task=None,
                    total_instances=0,
                    error=str(e),
                )
                return (model_name, spec, tracker, [])

        with ThreadPoolExecutor(max_workers=min(32, total_pairs)) as executor:
            futures = {
                executor.submit(prepare_one, model_name, spec): (model_name, spec)
                for model_name in self.model_names
                for spec in expanded_tasks
            }
            for future in as_completed(futures):
                model_name, spec, tracker, items = future.result()
                key = (model_name, spec)
                trackers[key] = tracker
                model_items[model_name].extend(items)
                if tracker.error:
                    console.print(f"  [red]- {model_name}:{spec}: ERROR - {tracker.error}[/red]")
                else:
                    console.print(f"  - {model_name}:{spec}: {len(items)} instances")
                    # Write requests early - we know them upfront before generation
                    if items and tracker.task:
                        request_objects = build_requests_from_items(items, tracker.task.config.name)
                        self._write_requests(model_name, spec, request_objects)

        total_instances = sum(len(items) for items in model_items.values())
        console.print(f"[bold]Total instances:[/bold] {total_instances}")

        # Setup multiprocessing context
        ctx = mp.get_context("spawn")

        # Create per-model queues + shared result queue
        model_queues: dict[str, mp.Queue] = {m: ctx.Queue() for m in self.model_names}
        result_queue: mp.Queue = ctx.Queue()

        # Shuffle and enqueue items per model
        for model_name, items in model_items.items():
            random.shuffle(items)
            for item in items:
                model_queues[model_name].put(item)

        # GPU allocation across models
        total_gpus = self._get_total_gpus()
        total_workers = self._get_num_workers()

        # Distribute workers across models
        num_models = len(self.model_names)
        workers_per_model = max(1, total_workers // num_models)
        gpus_per_model = max(0, total_gpus // num_models) if total_gpus > 0 else 0

        console.print(f"[bold]Total workers:[/bold] {total_workers}")
        console.print(f"[bold]Workers per model:[/bold] {workers_per_model}")
        console.print(f"[bold]GPUs per model:[/bold] {gpus_per_model}")

        # Add poison pills for all models AFTER all items are enqueued
        # This ensures workers see all items before the termination signal
        for model_name in self.model_names:
            for _ in range(workers_per_model):
                model_queues[model_name].put(None)

        # Start workers for each model
        workers: list[mp.process.BaseProcess] = []
        gpu_offset = 0

        for model_name in self.model_names:
            model_config = model_configs[model_name]
            backend_type = BackendType(model_config.backend)

            # Get per-model vLLM loading options from model_overrides
            per_model_overrides = self.model_overrides.get(model_name, {})
            effective_load_format = per_model_overrides.get("load_format")
            effective_extra_loader_config = per_model_overrides.get("extra_loader_config")

            # Spawn workers for this model
            for i in range(workers_per_model):
                if total_gpus > 0:
                    start_gpu = gpu_offset + (i * self.gpus_per_worker)
                    end_gpu = min(start_gpu + self.gpus_per_worker, gpu_offset + gpus_per_model)
                    gpu_ids = list(range(start_gpu, end_gpu)) if start_gpu < end_gpu else []
                else:
                    gpu_ids = []

                worker = ctx.Process(
                    target=instance_worker_process,
                    args=(
                        gpu_ids,
                        model_queues[model_name],
                        result_queue,
                        model_config.model,
                        backend_type.value,
                        self.attention_backend,
                        model_config.tokenizer,
                        model_config.max_model_len,
                        effective_load_format,
                        effective_extra_loader_config,
                    ),
                )
                worker.start()
                workers.append(worker)

            gpu_offset += gpus_per_model

        total_workers_spawned = len(workers)
        console.print(
            f"[bold green]{total_workers_spawned} worker(s) started across "
            f"{num_models} model(s), processing instances...[/bold green]"
        )

        # Wait for workers to initialize and check for early failures
        console.print("[dim]Waiting for workers to initialize...[/dim]")
        wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)
        console.print("[dim]Workers initialized successfully[/dim]")

        # Track results - keyed by (model, task)
        results: dict[tuple[str, str], TaskResult] = {}
        completed_pairs = 0

        # Pre-add error tasks to results
        for key, tracker in trackers.items():
            if tracker.error:
                task_result = finalize_task(tracker)
                results[key] = task_result
                completed_pairs += 1
                self._report_task_completion(tracker.model_name, task_result)

        # Track pending instances
        pending_instances = total_instances

        processed = 0
        last_health_check = time.time()
        health_check_interval = 5.0  # Check worker health every 5 seconds

        while completed_pairs < total_pairs and pending_instances > 0:
            # Use timeout-based queue get to allow periodic health checks
            try:
                result_item: ResultItem = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: result_queue.get(timeout=1.0)
                )
            except Exception:
                # Queue timeout - check worker health
                if time.time() - last_health_check > health_check_interval:
                    check_workers_alive(workers, result_queue)
                    last_health_check = time.time()
                continue

            processed += 1

            # Check for fatal worker crash
            if result_item.task_id == "__WORKER_FATAL__":
                console.print("\n[bold red]FATAL: Worker crashed![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
                # Terminate all workers
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                # Cancel queue join threads to allow clean process exit
                for queue in list(model_queues.values()) + [result_queue]:
                    queue.cancel_join_thread()
                raise RuntimeError(f"Worker process crashed: {result_item.error}")

            key = (result_item.model_name, result_item.task_id)
            tracker = trackers[key]

            # Skip if this (model, task) already failed
            if tracker.error:
                pending_instances -= 1
                continue

            if result_item.error:
                # Instance error - fail this (model, task) pair
                tracker.error = f"Instance {result_item.instance_idx} failed: {result_item.error}"
                pending_instances -= 1
                if tracker.is_complete():
                    task_result = finalize_task(tracker)
                    results[key] = task_result
                    completed_pairs += 1
                    self._report_task_completion(tracker.model_name, task_result)
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
                    results[key] = task_result
                    completed_pairs += 1
                    self._report_task_completion(tracker.model_name, task_result)
                    # Write predictions to JSONL
                    if task_result.predictions:
                        self._write_predictions(
                            tracker.model_name, task_result.spec, task_result.predictions
                        )
                    # Note: Requests are written early during task preparation,
                    # so we don't need to write them again here

        # Wait for all workers
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()

        # Check for errors
        errors = [(k, r) for k, r in results.items() if r.error]
        if errors:
            console.print(
                f"\n[bold red]Errors:[/bold red] {len(errors)} (model, task) pairs failed"
            )
            for (model_name, spec), error_result in errors:
                console.print(f"  - {model_name}:{spec}: {error_result.error}")

        # Aggregate results - grouped by model
        results_dict: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "errors": [],
        }

        for model_name in self.model_names:
            model_config = model_configs[model_name]
            backend_type = BackendType(model_config.backend)

            model_results: dict[str, Any] = {
                "model": model_config.model,
                "backend": backend_type.value,
                "tasks": {},
            }

            for spec in expanded_tasks:
                key = (model_name, spec)
                if key in results:
                    task_result = results[key]
                    if task_result.error:
                        results_dict["errors"].append(
                            {
                                "model": model_name,
                                "spec": spec,
                                "error": task_result.error,
                            }
                        )
                    else:
                        task_data: dict[str, Any] = {
                            "config": task_result.config,
                            "num_instances": task_result.num_instances,
                            "metrics": task_result.metrics,
                            "duration_seconds": task_result.duration_seconds,
                        }
                        if task_result.primary_metric:
                            task_data["primary_metric"] = task_result.primary_metric
                        model_results["tasks"][spec] = task_data

            # Store model config details for metrics.json
            model_results["_model_config"] = {
                "model": model_config.model,
                "tokenizer": model_config.tokenizer,
                "backend": backend_type.value,
                "dtype": model_config.dtype,
                "revision": model_config.revision,
                "attention_backend": self.attention_backend,
            }

            results_dict["models"][model_name] = model_results

            # Compute suite aggregations for this model
            suite_aggs = compute_suite_aggregations(self.task_specs, model_results["tasks"])
            if suite_aggs:
                model_results["suites"] = suite_aggs

        # Log summary of all scores
        self._log_summary(results_dict)

        # Save results
        self._save_results(results_dict)

        # Write metrics.json for Beaker
        self._write_metrics_json(results_dict)

        return results_dict

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

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    def _log_summary(self, results: dict[str, Any]) -> None:
        """Log summary of all task scores."""
        logger.info("Summary of primary scores:")
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

        metrics_output = {
            "timestamp": results.get("timestamp", ""),
            "config": config,
            "tasks": tasks_list,
            "summary": summary,
            "errors": results.get("errors", []),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics_output, f, indent=2)

        logger.info(f"Metrics written to {metrics_file}")
        console.print(f"[green]Metrics written to {metrics_file}[/green]")

    def _write_predictions(self, model_name: str, spec: str, predictions: list[dict]) -> None:
        """Write per-instance predictions to JSONL."""
        write_predictions_jsonl(self.output_dir, spec, predictions, model_name)

    def _write_requests(self, model_name: str, spec: str, requests: list[dict]) -> None:
        """Write per-instance requests to JSONL (oe-eval compatible format)."""
        write_requests_jsonl(self.output_dir, spec, requests, model_name)


# -----------------------------------------------------------------------------
# StreamingEvalRunner
# -----------------------------------------------------------------------------


@dataclass
class StreamingEvalRunner:
    """Streaming evaluation runner with true continuous batching.

    Uses vLLM's AsyncLLMEngine for true streaming where:
    - Requests are added continuously as they're prepared
    - Results stream back as they complete
    - Tasks report completion immediately when all instances finish

    This provides maximum throughput and earliest possible completion reporting.
    Only supports vLLM backend.
    """

    model_names: list[str]
    task_specs: list[str]
    output_dir: str = BEAKER_RESULT_DIR
    num_shots_override: int | None = None
    limit_override: int | None = None
    temperature: float | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    # Multi-worker config
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # vLLM config
    attention_backend: str | None = None  # e.g., "FLASHINFER", "FLASH_ATTN"

    # Per-task overrides from inline spec (e.g., task::temperature=0.6)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-model overrides from inline spec (e.g., model::tokenizer=..., model::load_format=...)
    # Maps model name -> overrides dict
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration."""
        from olmo_eval.evals.suites import suite_exists
        from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants
        from olmo_eval.evals.tasks.core.registry import parse_task_spec

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

        table = Table(title="Run Configuration (Streaming Mode)")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        models_str = ", ".join(self.model_names)
        table.add_row("Models", models_str)
        table.add_row("Mode", "Streaming (AsyncLLMEngine)")
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

        try:
            import torch  # type: ignore[import-not-found]

            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                return 1
            return max(1, num_gpus // self.gpus_per_worker)
        except ImportError:
            return 1

    def _get_total_gpus(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch  # type: ignore[import-not-found]

            return torch.cuda.device_count()
        except ImportError:
            return 0

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using streaming continuous batching.

        Uses vLLM's AsyncLLMEngine for true streaming where requests are
        added continuously and results stream back as they complete.
        """
        expanded_tasks = expand_tasks(self.task_specs)

        # Build global overrides from CLI args
        global_overrides: dict[str, Any] = {}
        global_sampling_overrides: dict[str, Any] = {}

        if self.num_shots_override is not None:
            global_overrides["num_fewshot"] = self.num_shots_override
        if self.limit_override is not None:
            global_overrides["limit"] = self.limit_override
        if self.temperature is not None:
            global_sampling_overrides["temperature"] = self.temperature

        # Prepare all (model, task) pairs
        trackers: dict[tuple[str, str], TaskTracker] = {}
        model_items: dict[str, list[QueueItem]] = {m: [] for m in self.model_names}
        model_configs: dict[str, Any] = {}

        console.print(f"[bold]Models:[/bold] {len(self.model_names)}")
        console.print(f"[bold]Tasks:[/bold] {len(expanded_tasks)}")
        total_pairs = len(self.model_names) * len(expanded_tasks)
        console.print(f"[bold]Total (model, task) pairs:[/bold] {total_pairs}")

        # Get model configs with per-model overrides
        for model_name in self.model_names:
            overrides = self.model_overrides.get(model_name, {})
            model_config = get_model_config(model_name, **overrides)
            model_configs[model_name] = model_config

        # Prepare tasks in parallel
        console.print(f"[bold]Preparing {total_pairs} tasks...[/bold]")

        def prepare_one(
            model_name: str, spec: str
        ) -> tuple[str, str, TaskTracker, list[QueueItem]]:
            try:
                # Build overrides for this task
                # 1. Start with global CLI overrides
                overrides = dict(global_overrides)
                sampling_overrides = dict(global_sampling_overrides)

                # 2. Apply per-task overrides (highest priority)
                task_specific = self.task_overrides.get(spec, {})
                for key, value in task_specific.items():
                    if key in TASKCONFIG_KEYS:
                        overrides[key] = value
                    elif key in SAMPLING_KEYS:
                        sampling_overrides[key] = value

                task, items = prepare_task_items(
                    spec,
                    model_name,
                    overrides or None,
                    sampling_overrides=sampling_overrides or None,
                )
                tracker = TaskTracker(
                    model_name=model_name,
                    spec=spec,
                    task=task,
                    total_instances=len(items),
                )
                return (model_name, spec, tracker, items)
            except Exception as e:
                tracker = TaskTracker(
                    model_name=model_name,
                    spec=spec,
                    task=None,
                    total_instances=0,
                    error=str(e),
                )
                return (model_name, spec, tracker, [])

        with ThreadPoolExecutor(max_workers=min(32, total_pairs)) as executor:
            futures = {
                executor.submit(prepare_one, model_name, spec): (model_name, spec)
                for model_name in self.model_names
                for spec in expanded_tasks
            }
            for future in as_completed(futures):
                model_name, spec, tracker, items = future.result()
                key = (model_name, spec)
                trackers[key] = tracker
                model_items[model_name].extend(items)
                if tracker.error:
                    console.print(f"  [red]- {model_name}:{spec}: ERROR - {tracker.error}[/red]")
                else:
                    console.print(f"  - {model_name}:{spec}: {len(items)} instances")
                    # Write requests early - we know them upfront before generation
                    if items and tracker.task:
                        request_objects = build_requests_from_items(items, tracker.task.config.name)
                        self._write_requests(model_name, spec, request_objects)

        total_instances = sum(len(items) for items in model_items.values())
        console.print(f"[bold]Total instances:[/bold] {total_instances}")

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        model_queues: dict[str, mp.Queue] = {m: ctx.Queue() for m in self.model_names}
        result_queue: mp.Queue = ctx.Queue()

        # GPU allocation
        total_gpus = self._get_total_gpus()
        total_workers = self._get_num_workers()
        num_models = len(self.model_names)
        workers_per_model = max(1, total_workers // num_models)
        gpus_per_model = max(0, total_gpus // num_models) if total_gpus > 0 else 0

        console.print(f"[bold]Total workers:[/bold] {total_workers}")
        console.print(f"[bold]Workers per model:[/bold] {workers_per_model}")
        console.print(f"[bold]GPUs per model:[/bold] {gpus_per_model}")

        # Start streaming workers for each model
        workers: list[mp.process.BaseProcess] = []
        gpu_offset = 0

        for model_name in self.model_names:
            model_config = model_configs[model_name]

            # Get per-model vLLM loading options from model_overrides
            per_model_overrides = self.model_overrides.get(model_name, {})
            effective_load_format = per_model_overrides.get("load_format")
            effective_extra_loader_config = per_model_overrides.get("extra_loader_config")

            for i in range(workers_per_model):
                if total_gpus > 0:
                    start_gpu = gpu_offset + (i * self.gpus_per_worker)
                    end_gpu = min(start_gpu + self.gpus_per_worker, gpu_offset + gpus_per_model)
                    gpu_ids = list(range(start_gpu, end_gpu)) if start_gpu < end_gpu else []
                else:
                    gpu_ids = []

                worker = ctx.Process(
                    target=streaming_worker_process,
                    args=(
                        gpu_ids,
                        model_queues[model_name],
                        result_queue,
                        model_config.model,
                        self.attention_backend,
                        model_config.tokenizer,
                        model_config.max_model_len,
                        effective_load_format,
                        effective_extra_loader_config,
                    ),
                )
                worker.start()
                workers.append(worker)

            gpu_offset += gpus_per_model

        total_workers_spawned = len(workers)
        console.print(
            f"[bold green]{total_workers_spawned} streaming worker(s) started across "
            f"{num_models} model(s)[/bold green]"
        )

        # Wait for workers to initialize and check for early failures
        console.print("[dim]Waiting for workers to initialize...[/dim]")
        wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)
        console.print("[dim]Workers initialized successfully[/dim]")

        # Enqueue items - workers will start processing immediately
        for model_name, items in model_items.items():
            random.shuffle(items)
            for item in items:
                model_queues[model_name].put(item)

        # Send poison pills
        for model_name in self.model_names:
            for _ in range(workers_per_model):
                model_queues[model_name].put(None)

        # Track results
        results: dict[tuple[str, str], TaskResult] = {}
        completed_pairs = 0

        # Pre-add error tasks
        for key, tracker in trackers.items():
            if tracker.error:
                task_result = finalize_task(tracker)
                results[key] = task_result
                completed_pairs += 1
                self._report_task_completion(tracker.model_name, task_result)

        pending_instances = total_instances
        processed = 0
        last_health_check = time.time()
        health_check_interval = 5.0  # Check worker health every 5 seconds

        while completed_pairs < total_pairs and pending_instances > 0:
            # Use timeout-based queue get to allow periodic health checks
            try:
                result_item: ResultItem = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: result_queue.get(timeout=1.0)
                )
            except Exception:
                # Queue timeout - check worker health
                if time.time() - last_health_check > health_check_interval:
                    check_workers_alive(workers, result_queue)
                    last_health_check = time.time()
                continue

            processed += 1

            # Check for fatal worker crash
            if result_item.task_id == "__WORKER_FATAL__":
                console.print("\n[bold red]FATAL: Worker crashed![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
                # Terminate all workers
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                # Cancel queue join threads to allow clean process exit
                for queue in list(model_queues.values()) + [result_queue]:
                    queue.cancel_join_thread()
                raise RuntimeError(f"Worker process crashed: {result_item.error}")

            key = (result_item.model_name, result_item.task_id)
            tracker = trackers[key]

            if tracker.error:
                pending_instances -= 1
                continue

            if result_item.error:
                tracker.error = f"Instance {result_item.instance_idx} failed: {result_item.error}"
                pending_instances -= 1
                if tracker.is_complete():
                    task_result = finalize_task(tracker)
                    results[key] = task_result
                    completed_pairs += 1
                    self._report_task_completion(tracker.model_name, task_result)
            else:
                response = Response(
                    instance=result_item.instance,
                    request=result_item.request,
                    outputs=result_item.outputs,
                )

                is_complete = tracker.add_response(result_item.instance_idx, response)
                pending_instances -= 1

                if is_complete:
                    task_result = finalize_task(tracker)
                    results[key] = task_result
                    completed_pairs += 1
                    self._report_task_completion(tracker.model_name, task_result)
                    # Write predictions to JSONL
                    if task_result.predictions:
                        self._write_predictions(
                            tracker.model_name, task_result.spec, task_result.predictions
                        )
                    # Note: Requests are written early during task preparation,
                    # so we don't need to write them again here

        # Wait for workers
        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join()

        # Check for errors
        errors = [(k, r) for k, r in results.items() if r.error]
        if errors:
            console.print(
                f"\n[bold red]Errors:[/bold red] {len(errors)} (model, task) pairs failed"
            )
            for (model_name, spec), error_result in errors:
                console.print(f"  - {model_name}:{spec}: {error_result.error}")

        # Aggregate results
        results_dict: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "errors": [],
        }

        for model_name in self.model_names:
            model_config = model_configs[model_name]

            model_results: dict[str, Any] = {
                "model": model_config.model,
                "backend": "vllm",
                "tasks": {},
            }

            for spec in expanded_tasks:
                key = (model_name, spec)
                if key in results:
                    task_result = results[key]
                    if task_result.error:
                        results_dict["errors"].append(
                            {
                                "model": model_name,
                                "spec": spec,
                                "error": task_result.error,
                            }
                        )
                    else:
                        task_data: dict[str, Any] = {
                            "config": task_result.config,
                            "num_instances": task_result.num_instances,
                            "metrics": task_result.metrics,
                            "duration_seconds": task_result.duration_seconds,
                        }
                        if task_result.primary_metric:
                            task_data["primary_metric"] = task_result.primary_metric
                        model_results["tasks"][spec] = task_data

            # Store model config details for metrics.json
            model_results["_model_config"] = {
                "model": model_config.model,
                "tokenizer": model_config.tokenizer,
                "backend": "vllm",
                "dtype": model_config.dtype,
                "revision": model_config.revision,
                "attention_backend": self.attention_backend,
            }

            results_dict["models"][model_name] = model_results

            # Compute suite aggregations for this model
            suite_aggs = compute_suite_aggregations(self.task_specs, model_results["tasks"])
            if suite_aggs:
                model_results["suites"] = suite_aggs

        self._log_summary(results_dict)
        self._save_results(results_dict)
        self._write_metrics_json(results_dict)

        return results_dict

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

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    def _log_summary(self, results: dict[str, Any]) -> None:
        """Log summary of all task scores."""
        logger.info("Summary of primary scores:")
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

        metrics_output = {
            "timestamp": results.get("timestamp", ""),
            "config": config,
            "tasks": tasks_list,
            "summary": summary,
            "errors": results.get("errors", []),
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics_output, f, indent=2)

        logger.info(f"Metrics written to {metrics_file}")
        console.print(f"[green]Metrics written to {metrics_file}[/green]")

    def _write_predictions(self, model_name: str, spec: str, predictions: list[dict]) -> None:
        """Write per-instance predictions to JSONL."""
        write_predictions_jsonl(self.output_dir, spec, predictions, model_name)

    def _write_requests(self, model_name: str, spec: str, requests: list[dict]) -> None:
        """Write per-instance requests to JSONL (oe-eval compatible format)."""
        write_requests_jsonl(self.output_dir, spec, requests, model_name)

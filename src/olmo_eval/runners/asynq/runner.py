"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.cli.utils import console
from olmo_eval.common.configs import expand_tasks
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.logging import get_logger, get_worker_id
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq.monitoring import (
    terminate_workers,
    wait_for_init_times,
    wait_for_workers_ready,
)
from olmo_eval.runners.asynq.preparation import (
    build_requests_from_items,
    prepare_task_items,
)
from olmo_eval.runners.asynq.results import aggregate_results, process_results
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker
from olmo_eval.runners.asynq.workers import inference_worker, scoring_worker
from olmo_eval.runners.common.base import BaseEvalRunner
from olmo_eval.runners.common.mixins import RunnerResultsMixin
from olmo_eval.runners.common.models import S3Config
from olmo_eval.runners.processing.utils import compute_task_hash, generate_experiment_id
from olmo_eval.storage import StorageBackend

logger = get_logger(__name__)


@dataclass
class AsyncEvalRunner(RunnerResultsMixin, BaseEvalRunner):
    """Async evaluation runner with instance-level queuing.

    Uses a single model with instance-level queuing where instances from all
    tasks are mixed together, enabling better GPU utilization and early
    completion reporting.
    """

    # Harness configuration (includes provider, tools, system prompt)
    harness_config: HarnessConfig = field(default_factory=lambda: HarnessConfig(name="default"))

    # Task configuration
    task_specs: list[str] = field(default_factory=list)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Output configuration
    output_dir: str = BEAKER_RESULT_DIR
    storages: list[StorageBackend] = field(default_factory=list)

    # vLLM-specific configuration
    attention_backend: str | None = None

    # S3 upload configuration (optional)
    s3_config: S3Config | None = None

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None

    # Output persistence options
    save_predictions: bool = True
    save_requests: bool = True

    # Instance inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_request: bool = False

    # Configuration for print_config display
    _mode_name: str = "Async Mode"
    _mode_description: str = "Async (All-at-once)"

    @property
    def provider_config(self) -> ProviderConfig:
        """Get the provider config from harness config."""
        return self.harness_config.provider

    @property
    def model_name(self) -> str:
        """Get the model name from provider config."""
        return self.provider_config.model

    def validate(self) -> None:
        """Validate runner configuration."""
        from olmo_eval.runners.common.constants import ValidationError

        if not self.provider_config.model:
            raise ValidationError("provider_config.model is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def print_config(self) -> None:
        """Print runner configuration."""
        from rich.table import Table

        from olmo_eval.common.configs import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", self.model_name)
        table.add_row("Provider", self.provider_config.get_provider_name())
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)

        console.print(table)

        expanded = expand_tasks(self.task_specs)
        console.print(f"\n[bold]Tasks:[/bold] {len(expanded)}")
        for spec in expanded:
            console.print(f"  - {spec}")

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using instance-level queuing."""
        # Track experiment start time
        experiment_start = time.time()

        # Prepare tasks
        expanded_tasks, trackers, items = self._prepare_tasks()
        total_instances = len(items)
        logger.info(f"Total instances: {total_instances}")

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        item_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()
        scoring_queue: mp.Queue = ctx.Queue()
        scored_queue: mp.Queue = ctx.Queue()
        num_workers = self._get_num_workers()
        total_gpus = self._get_gpu_count()

        # Create shared dict for tracking worker init times
        manager = ctx.Manager()
        init_times = manager.dict()

        # Shuffle and enqueue items
        random.shuffle(items)
        for item in items:
            item_queue.put(item)

        # Add poison pills AFTER all items are enqueued
        for _ in range(num_workers):
            item_queue.put(None)

        # Start workers
        workers: list[mp.process.BaseProcess] = []
        scorer_proc: mp.process.BaseProcess | None = None

        try:
            # Start scoring worker first so it's ready to receive work
            scorer_proc = ctx.Process(
                target=scoring_worker,
                args=(scoring_queue, scored_queue, total_instances),
            )
            scorer_proc.start()

            workers = self._start_workers(
                ctx, num_workers, total_gpus, item_queue, result_queue, init_times
            )

            # Wait for workers to initialize
            logger.info("Waiting for workers to initialize...")
            wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)
            logger.info("Workers initialized successfully")

            # Wait for workers to report their init times
            provider_init_seconds = wait_for_init_times(init_times, num_workers)

            # Reset tracker start times now that workers are ready
            # This ensures task duration only measures actual processing time
            processing_start = time.time()
            for tracker in trackers.values():
                tracker.start_time = processing_start

            # Process results
            results = await self._process_results(
                trackers,
                result_queue,
                scoring_queue,
                scored_queue,
                workers,
                len(expanded_tasks),
                total_instances,
            )

            # Signal scoring worker to shutdown and wait
            scoring_queue.put(None)
            scorer_proc.join(timeout=30)
            if scorer_proc.is_alive():
                scorer_proc.terminate()
                scorer_proc.join()

            # Wait for all workers
            for worker in workers:
                worker.join(timeout=10)
                if worker.is_alive():
                    worker.terminate()
                    worker.join()

            # Compute experiment duration
            experiment_duration_seconds = time.time() - experiment_start

            # Aggregate and save results
            results_dict = self._aggregate_results(results, expanded_tasks)
            return self._finalize_and_save(
                results_dict,
                experiment_duration_seconds=experiment_duration_seconds,
                provider_init_seconds=provider_init_seconds,
            )
        finally:
            terminate_workers(workers)
            if scorer_proc and scorer_proc.is_alive():
                scorer_proc.terminate()
                scorer_proc.join(timeout=5)
            for q in [item_queue, result_queue, scoring_queue, scored_queue]:
                q.cancel_join_thread()
            manager.shutdown()

    def _prepare_tasks(
        self,
    ) -> tuple[list[str], dict[str, TaskTracker], list[QueueItem]]:
        """Prepare all tasks and return tracking data structures."""
        expanded_tasks = expand_tasks(self.task_specs)

        trackers: dict[str, TaskTracker] = {}
        items: list[QueueItem] = []

        def prepare_one(spec: str) -> tuple[str, TaskTracker, list[QueueItem]]:
            try:
                overrides, sampling_overrides = self._build_task_overrides(spec)
                task, task_items = prepare_task_items(
                    spec,
                    self.model_name,
                    overrides or None,
                    sampling_overrides=sampling_overrides or None,
                )
                tracker = TaskTracker(
                    model_name=self.model_name,
                    spec=spec,
                    task=task,
                    total_instances=len(task_items),
                )
                return (spec, tracker, task_items)
            except Exception as e:
                tracker = TaskTracker(
                    model_name=self.model_name,
                    spec=spec,
                    task=None,
                    total_instances=0,
                    error=str(e),
                )
                return (spec, tracker, [])

        with ThreadPoolExecutor(max_workers=min(32, len(expanded_tasks))) as executor:
            futures = {executor.submit(prepare_one, spec): spec for spec in expanded_tasks}
            for future in as_completed(futures):
                spec, tracker, task_items = future.result()
                trackers[spec] = tracker
                items.extend(task_items)
                if tracker.error:
                    logger.error(f"  {spec}: ERROR - {tracker.error}")
                else:
                    logger.info(f"  {spec}: {len(task_items)} instances")
                    if self.save_requests and task_items and tracker.task:
                        request_objects = build_requests_from_items(
                            task_items, tracker.task.config.name
                        )
                        task_hash = compute_task_hash(tracker.task.config.to_dict())
                        self._write_requests(self.model_name, spec, request_objects, task_hash)

        # Optionally inspect first instance of each task
        if (
            self.inspect_instance
            or self.inspect_formatted
            or self.inspect_tokens
            or self.inspect_request
        ):
            from olmo_eval.common.inspection import inspect_task_instances

            inspect_task_instances(
                trackers,
                self.provider_config,
                inspect_instance_flag=self.inspect_instance,
                inspect_formatted=self.inspect_formatted,
                inspect_tokens_flag=self.inspect_tokens,
                inspect_request_flag=self.inspect_request,
                console=console,
            )

        return expanded_tasks, trackers, items

    def _build_task_overrides(self, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build task and sampling overrides for a given task spec.

        Returns:
            Tuple of (task_overrides, sampling_overrides)
        """
        from dataclasses import fields

        from olmo_eval.common.types import SamplingParams
        from olmo_eval.evals.tasks.common.base import TaskConfig

        task_overrides: dict[str, Any] = {}
        sampling_overrides: dict[str, Any] = {}

        # Get field names from dataclasses
        task_fields = {f.name for f in fields(TaskConfig)}
        sampling_fields = {f.name for f in fields(SamplingParams)}

        # Apply per-task overrides
        per_task = self.task_overrides.get(spec, {})
        for key, value in per_task.items():
            if key in task_fields:
                task_overrides[key] = value
            elif key in sampling_fields:
                sampling_overrides[key] = value

        return task_overrides, sampling_overrides

    def _start_workers(
        self,
        ctx: Any,
        num_workers: int,
        total_gpus: int,
        item_queue: mp.Queue,
        result_queue: mp.Queue,
        init_times: Any,
    ) -> list[mp.process.BaseProcess]:
        """Start worker processes."""
        workers: list[mp.process.BaseProcess] = []
        harness_config_dict = self.harness_config.to_dict()

        for i in range(num_workers):
            worker_id = get_worker_id(self.provider_config.model, i)

            # All GPUs are assigned to the single worker for tensor parallelism
            gpu_ids = list(range(total_gpus)) if total_gpus > 0 else []

            worker = ctx.Process(
                target=inference_worker,
                args=(
                    worker_id,
                    gpu_ids,
                    item_queue,
                    result_queue,
                    harness_config_dict,
                    init_times,
                    self.output_dir,
                ),
            )
            worker.start()
            workers.append(worker)

        return workers

    def _get_gpu_count(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch  # type: ignore[import-not-found]

            return torch.cuda.device_count()
        except ImportError:
            return 0

    def _get_num_workers(self) -> int:
        """Get number of workers based on provider requirements.

        For single-GPU or CPU providers, returns 1.
        Multi-GPU tensor parallelism is handled within a single worker.
        """
        return 1

    async def _process_results(
        self,
        trackers: dict[str, TaskTracker],
        result_queue: mp.Queue,
        scoring_queue: mp.Queue,
        scored_queue: mp.Queue,
        workers: list[mp.process.BaseProcess],
        total_tasks: int,
        total_instances: int,
    ) -> dict[str, Any]:
        """Process results from workers with parallel instance-level scoring."""
        return await process_results(
            trackers=trackers,
            result_queue=result_queue,
            scoring_queue=scoring_queue,
            scored_queue=scored_queue,
            workers=workers,
            total_tasks=total_tasks,
            total_instances=total_instances,
            model_name=self.model_name,
            save_predictions=self.save_predictions,
            write_predictions_fn=self._write_predictions,
        )

    def _aggregate_results(
        self,
        results: dict[str, Any],
        expanded_tasks: list[str],
    ) -> dict[str, Any]:
        """Aggregate results and prepare final output."""
        return aggregate_results(
            results=results,
            expanded_tasks=expanded_tasks,
            task_specs=self.task_specs,
            provider_config=self.provider_config,
            attention_backend=self.attention_backend,
        )

    def _finalize_and_save(
        self,
        results_dict: dict[str, Any],
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log summary, write metrics, upload to S3, and save results."""
        from olmo_eval.common.types import compute_model_hash

        self._log_summary(results_dict)

        experiment_id = generate_experiment_id()
        model_hash = compute_model_hash(results_dict.get("model_config", {}))
        results_dict["_model_hash"] = model_hash

        self._write_metrics_json(
            results=results_dict,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        s3_location: str | None = None
        if self.s3_config and model_hash:
            s3_location = self._upload_to_s3(
                model_name=self.model_name,
                model_hash=model_hash,
                experiment_id=experiment_id,
            )

        results_dict["_experiment_id"] = experiment_id
        results_dict["_s3_location"] = s3_location

        self._save_results(
            results=results_dict,
            experiment_id=experiment_id,
            model_hash=model_hash,
            s3_location=s3_location,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        return results_dict

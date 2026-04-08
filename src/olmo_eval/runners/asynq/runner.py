"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from olmo_eval.common.configs import expand_tasks
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.logging import configure_worker_logging, get_logger
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq.monitoring import (
    terminate_workers,
    wait_for_init_times,
    wait_for_scorer_ready,
    wait_for_workers_ready,
)
from olmo_eval.runners.asynq.preparation import (
    build_requests_from_items,
    prepare_task_items,
)
from olmo_eval.runners.asynq.results import aggregate_results, process_results
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker
from olmo_eval.runners.asynq.workers import scoring_worker
from olmo_eval.runners.common.base import BaseEvalRunner
from olmo_eval.runners.common.mixins import RunnerResultsMixin
from olmo_eval.runners.common.models import S3Config
from olmo_eval.runners.processing.utils import compute_task_hash, generate_experiment_id
from olmo_eval.storage import StorageBackend

logger = get_logger(__name__)
runner_logger = configure_worker_logging("runner")
console = Console(force_terminal=True, width=120)


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

    # Shuffle seed for deterministic instance ordering (enables future checkpointing)
    shuffle_seed: int = 42

    # Instance inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
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

        # Generate experiment ID early so metrics can include it
        experiment_id = generate_experiment_id()

        # Prepare tasks
        expanded_tasks, trackers, items = self._prepare_tasks()
        total_instances = len(items)
        runner_logger.info(f"Total instances: {total_instances}")

        # Update harness metrics config with experiment metadata before starting workers
        self._update_metrics_config(experiment_id)

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        item_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()
        scoring_queue: mp.Queue = ctx.Queue()
        scored_queue: mp.Queue = ctx.Queue()
        total_gpus = self._get_gpu_count()

        # Create shared dict for tracking worker init times
        manager = ctx.Manager()
        init_times = manager.dict()

        # Shuffle with seed for deterministic ordering (enables future checkpointing)
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(items)
        for item in items:
            item_queue.put(item)

        # Start workers
        workers: list[mp.process.BaseProcess] = []
        scorer_proc: mp.process.BaseProcess | None = None
        inference_manager = None

        try:
            from olmo_eval.inference.gpu_planner import GPUPlanner
            from olmo_eval.inference.provider_manager import (
                ProviderManager,
                validate_inference_workers,
            )

            # Use GPU planner to allocate GPUs for main workers and auxiliary providers
            planner = GPUPlanner.from_harness_config(self.harness_config, total_gpus)
            gpu_plan = planner.plan()

            # Get main worker GPUs (flattened list from all worker allocations)
            main_gpu_ids: list[int] = []
            for alloc in gpu_plan.main_workers:
                main_gpu_ids.extend(alloc.gpu_ids)

            # Number of inference workers is determined by provider.num_instances
            num_inference_workers = self.harness_config.provider.num_instances

            # Validate configuration
            validate_inference_workers(num_inference_workers, len(main_gpu_ids))

            provider_manager = ProviderManager(
                harness_config=self.harness_config,
                num_inference_workers=num_inference_workers,
                gpu_ids=main_gpu_ids,
                item_queue=item_queue,
                result_queue=result_queue,
                output_dir=self.output_dir,
            )

            # Add poison pills (one per worker)
            provider_manager.add_poison_pills()

            # Start workers
            workers = provider_manager.start(ctx, total_instances, init_times)

            # Start auxiliary inference servers if configured
            registry_config: dict[str, list[dict[str, Any]]] | None = None
            if self.harness_config.auxiliary_providers:
                from olmo_eval.inference.manager import InferenceManager

                auxiliary_gpus = gpu_plan.get_auxiliary_gpus()

                inference_manager = InferenceManager(
                    configs=dict(self.harness_config.auxiliary_providers),
                    available_gpu_ids=auxiliary_gpus,
                )
                registry_config = inference_manager.start()
                runner_logger.info(f"Auxiliary providers ready: {list(registry_config.keys())}")

            # Prepare sandbox configs for scoring worker if configured
            sandbox_configs_list = None
            if self.harness_config.sandboxes:
                sandbox_configs_list = [s.to_dict() for s in self.harness_config.sandboxes]

            # Create ready event for scoring worker
            scorer_ready = ctx.Event()

            # Determine scoring concurrency
            if self.harness_config.scoring_concurrency:
                scoring_concurrency = self.harness_config.scoring_concurrency
            elif sandbox_configs_list is not None:
                # Match sandbox pool size (sum of instances across all configs)
                sandbox_pool_size = sum(cfg.get("instances", 1) for cfg in sandbox_configs_list)
                scoring_concurrency = max(1, sandbox_pool_size)
            else:
                # CPU-bound scoring - use available cores
                import os

                cpu_count = os.cpu_count() or 4
                scoring_concurrency = max(1, int(cpu_count * 0.75))

            runner_logger.info(f"Scoring concurrency: {scoring_concurrency}")

            scorer_id = "scorer-0"  # TODO: support multiple scorers
            scorer_proc = ctx.Process(
                target=scoring_worker,
                args=(
                    scorer_id,
                    scoring_queue,
                    scored_queue,
                    total_instances,
                    sandbox_configs_list,
                    scorer_ready,
                    scoring_concurrency,
                    registry_config,
                ),
            )
            scorer_proc.start()

            # Wait for workers to initialize
            runner_logger.info("Waiting for inference workers to initialize...")
            wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)

            # Now wait for scoring worker (runs in parallel with inference worker init)
            if sandbox_configs_list is not None:
                # Compute scorer startup timeout
                if self.harness_config.scorer_startup_timeout is not None:
                    scorer_timeout = self.harness_config.scorer_startup_timeout
                else:
                    # Derive from max sandbox startup_timeout + buffer
                    max_startup = max(
                        cfg.get("startup_timeout", 60.0) for cfg in sandbox_configs_list
                    )
                    scorer_timeout = max_startup + 60.0

                runner_logger.info(
                    f"Waiting for scoring worker to initialize (timeout={scorer_timeout}s)..."
                )
                wait_for_scorer_ready(
                    scorer_proc, scorer_ready, scored_queue, timeout=scorer_timeout
                )
                runner_logger.info("Scoring worker ready")

            # Wait for workers to report their init times (also checks for crashes)
            provider_init_seconds = wait_for_init_times(
                init_times, num_inference_workers, workers=workers, result_queue=result_queue
            )

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
                scorer_proc,
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

            # Optionally inspect first response of each task
            if self.inspect_response:
                from olmo_eval.common.inspection import inspect_response

                for spec, tracker in trackers.items():
                    if tracker.responses:
                        first_response = next(iter(tracker.responses.values()))
                        console.print()
                        inspect_response(
                            first_response,
                            console=console,
                            task_name=spec,
                        )
                        break  # Only show first task's first response

            # Compute experiment duration
            experiment_duration_seconds = time.time() - experiment_start

            # Aggregate and save results
            results_dict = self._aggregate_results(results, expanded_tasks)
            return self._finalize_and_save(
                results_dict,
                experiment_id=experiment_id,
                experiment_duration_seconds=experiment_duration_seconds,
                provider_init_seconds=provider_init_seconds,
            )
        finally:
            terminate_workers(workers)
            if scorer_proc and scorer_proc.is_alive():
                scorer_proc.terminate()
                scorer_proc.join(timeout=5)
            if inference_manager is not None:
                inference_manager.shutdown()
            for q in [item_queue, result_queue, scoring_queue, scored_queue]:
                q.cancel_join_thread()
            manager.shutdown()

    def _prepare_tasks(
        self,
    ) -> tuple[list[str], dict[str, TaskTracker], list[QueueItem]]:
        """Prepare all tasks and return tracking data structures."""
        expanded_tasks = expand_tasks(self.task_specs)

        if not expanded_tasks:
            raise ValueError("No tasks to run after expansion. Check task_specs configuration.")

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

        from rich.table import Table

        # Collect prepared tasks in parallel, but accumulate results for deterministic ordering
        prepared_results: dict[str, tuple[TaskTracker, list[QueueItem]]] = {}
        with ThreadPoolExecutor(max_workers=min(32, len(expanded_tasks))) as executor:
            futures = {executor.submit(prepare_one, spec): spec for spec in expanded_tasks}
            for future in as_completed(futures):
                spec, tracker, task_items = future.result()
                trackers[spec] = tracker
                prepared_results[spec] = (tracker, task_items)

        # Add items in deterministic task spec order (not completion order)
        # This ensures shuffle produces identical results across runs
        for spec in expanded_tasks:
            tracker, task_items = prepared_results[spec]
            items.extend(task_items)
            if not tracker.error and self.save_requests and task_items and tracker.task:
                request_objects = build_requests_from_items(task_items, tracker.task.config.name)
                task_hash = compute_task_hash(tracker.task.config.to_dict())
                self._write_requests(self.model_name, spec, request_objects, task_hash)

        # Print task preparation summary table
        table = Table(title="Tasks")
        table.add_column("Task", style="cyan")
        table.add_column("Instances", justify="right")
        table.add_column("Status")

        for spec in expanded_tasks:
            tracker = trackers[spec]
            if tracker.error:
                table.add_row(spec, "-", f"[red]ERROR: {tracker.error}[/red]")
            else:
                table.add_row(spec, str(tracker.total_instances), "[green]Ready[/green]")

        console.print(table)

        # Fail fast if any tasks failed to prepare
        failed_tasks = [spec for spec, tracker in trackers.items() if tracker.error]
        if failed_tasks:
            error_details = "\n".join(
                f"  - {spec}: {trackers[spec].error}" for spec in failed_tasks
            )
            raise RuntimeError(f"Failed to prepare {len(failed_tasks)} task(s):\n{error_details}")

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

    def _get_gpu_count(self) -> int:
        """Get total number of available GPUs."""
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            return 0

    def _update_metrics_config(self, experiment_id: str) -> None:
        """Update harness metrics config with experiment metadata.

        This must be called before starting workers so the serialized config
        includes all metadata for metrics reporting.
        """
        from olmo_eval.common.types import compute_model_hash

        if self.harness_config.metrics is None or not self.harness_config.metrics.enabled:
            return

        # Compute model hash from provider config
        model_hash = compute_model_hash(self.harness_config.provider.to_dict())

        # Update metrics config with all available metadata
        updated_metrics = self.harness_config.metrics.with_metadata(
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_name=self.model_name,
            model_hash=model_hash,
        )

        # Replace harness config with updated metrics
        self.harness_config = self.harness_config.with_metrics(updated_metrics)

    async def _process_results(
        self,
        trackers: dict[str, TaskTracker],
        result_queue: mp.Queue,
        scoring_queue: mp.Queue,
        scored_queue: mp.Queue,
        workers: list[mp.process.BaseProcess],
        scorer_proc: mp.process.BaseProcess,
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
            scorer_proc=scorer_proc,
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
            harness_config=self.harness_config,
        )

    def _finalize_and_save(
        self,
        results_dict: dict[str, Any],
        experiment_id: str,
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log summary, write metrics, upload to S3, and save results."""
        from olmo_eval.common.types import compute_model_hash

        self._log_summary(results_dict)

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

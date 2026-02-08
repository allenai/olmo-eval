"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rich.console import Console

from olmo_eval.core.configs import expand_tasks
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.core.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.core.logging import get_logger, get_worker_id
from olmo_eval.core.types import Response
from olmo_eval.inference import ProviderType
from olmo_eval.runners.base import BaseEvalRunner
from olmo_eval.runners.mixins import S3Config
from olmo_eval.runners.simple.helpers import (
    check_workers_alive,
    terminate_workers,
    wait_for_workers_ready,
)
from olmo_eval.runners.simple.queue import (
    QueueItem,
    ResultItem,
    TaskTracker,
    build_requests_from_items,
    finalize_task,
    prepare_task_items,
)
from olmo_eval.runners.simple.workers import instance_worker_process
from olmo_eval.runners.utils import (
    compute_suite_aggregations,
    compute_task_hash,
    generate_experiment_id,
)
from olmo_eval.storage import StorageBackend

console = Console()
logger = get_logger(__name__)


@dataclass
class AsyncEvalRunner(BaseEvalRunner):
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

    # Worker configuration
    num_workers: int | None = None
    gpus_per_worker: int = 1

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
        from olmo_eval.runners.constants import ValidationError

        if not self.provider_config.model:
            raise ValidationError("provider_config.model is required")

        if not self.task_specs:
            raise ValidationError("task_specs is required")

        # Validate task specs
        errors = self._validate_task_specs()
        if errors:
            raise ValidationError("\n".join(errors))

    def _validate_task_specs(self) -> list[str]:
        """Validate task specs and return list of errors.

        Includes validation of task names and variants/regimes.
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

            # Parse task_name[:variant1[:variant2...]] format
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

    def print_config(self) -> None:
        """Print runner configuration."""
        from rich.table import Table

        from olmo_eval.core.configs import expand_tasks

        table = Table(title=f"Run Configuration ({self._mode_name})")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", self.model_name)
        table.add_row("Provider", self.provider_config.get_provider_name())
        table.add_row("Mode", self._mode_description)
        table.add_row("Output Dir", self.output_dir)
        table.add_row("Workers", str(self.num_workers or "auto-detect"))
        table.add_row("GPUs per Worker", str(self.gpus_per_worker))

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

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        instance_queue, result_queue, num_workers = self._setup_workers(items, ctx)
        total_gpus = self._get_total_gpus()

        # Create shared dict for tracking worker init times
        manager = ctx.Manager()
        init_times = manager.dict()

        # Shuffle and enqueue items
        random.shuffle(items)
        for item in items:
            instance_queue.put(item)

        # Add poison pills AFTER all items are enqueued
        for _ in range(num_workers):
            instance_queue.put(None)

        # Start workers
        workers: list[mp.process.BaseProcess] = []
        provider_type = ProviderType(self.provider_config.get_provider_name())

        try:
            for i in range(num_workers):
                worker_id = get_worker_id(self.provider_config.model, i)

                if total_gpus > 0:
                    start_gpu = i * self.gpus_per_worker
                    end_gpu = min(start_gpu + self.gpus_per_worker, total_gpus)
                    gpu_ids = list(range(start_gpu, end_gpu)) if start_gpu < end_gpu else []
                else:
                    gpu_ids = []

                worker = ctx.Process(
                    target=instance_worker_process,
                    args=(
                        worker_id,
                        gpu_ids,
                        instance_queue,
                        result_queue,
                        self.provider_config.model,
                        provider_type.value,
                        self.attention_backend,
                        self.provider_config.tokenizer,
                        self.provider_config.max_model_len,
                        self.provider_config.kwargs.get("load_format"),
                        self.provider_config.kwargs.get("extra_loader_config"),
                        self.provider_config.max_concurrency,
                        init_times,
                        self.harness_config.to_dict(),
                    ),
                )
                worker.start()
                workers.append(worker)

            logger.info(f"{len(workers)} worker(s) started, processing instances...")

            # Wait for workers to initialize
            logger.info("Waiting for workers to initialize...")
            wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)
            logger.info("Workers initialized successfully")

            # Capture init times from workers
            provider_init_seconds = dict(init_times)

            # Reset tracker start times now that workers are ready
            # This ensures task duration only measures actual processing time
            processing_start = time.time()
            for tracker in trackers.values():
                tracker.start_time = processing_start

            # Process results
            results = await self._process_results(
                trackers,
                result_queue,
                instance_queue,
                workers,
                len(expanded_tasks),
                total_instances,
            )

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
            for q in [instance_queue, result_queue]:
                q.cancel_join_thread()
            manager.shutdown()

    def _prepare_tasks(
        self,
    ) -> tuple[list[str], dict[str, TaskTracker], list[QueueItem]]:
        """Prepare all tasks and return tracking data structures."""
        expanded_tasks = expand_tasks(self.task_specs)

        trackers: dict[str, TaskTracker] = {}
        items: list[QueueItem] = []

        logger.info(f"Model: {self.model_name}")
        logger.info(f"Tasks: {len(expanded_tasks)}")

        # Prepare tasks in parallel
        logger.info(f"Preparing {len(expanded_tasks)} tasks...")

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
            self._inspect_tasks(trackers)

        return expanded_tasks, trackers, items

    def _build_task_overrides(self, spec: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build task and sampling overrides for a given task spec.

        Returns:
            Tuple of (task_overrides, sampling_overrides)
        """
        from dataclasses import fields

        from olmo_eval.core.types import SamplingParams
        from olmo_eval.evals.tasks.core.base import TaskConfig

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

    def _inspect_tasks(self, trackers: dict[str, TaskTracker]) -> None:
        """Inspect first instance of each unique task."""
        from olmo_eval.core.inspection import (
            format_with_chat_template,
            inspect_formatted_request,
            inspect_instance,
            inspect_request,
            inspect_tokens,
            load_tokenizer,
            tokenize_request,
        )

        tokenizer = None

        if self.inspect_formatted or self.inspect_tokens:
            tokenizer_name = self.provider_config.tokenizer or self.provider_config.model
            try:
                tokenizer = load_tokenizer(tokenizer_name)
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")

        for spec, tracker in trackers.items():
            if tracker.task and not tracker.error:
                first_instance = next(iter(tracker.task.instances), None)
                if first_instance:
                    native_id = first_instance.metadata.get("id", "0")

                    if self.inspect_instance:
                        console.print()
                        inspect_instance(
                            first_instance, console=console, task_name=spec, native_id=native_id
                        )

                    if self.inspect_request or (
                        tokenizer and (self.inspect_formatted or self.inspect_tokens)
                    ):
                        request = tracker.task.format_request(first_instance)

                        if self.inspect_request:
                            inspect_request(
                                request, console=console, task_name=spec, native_id=native_id
                            )

                        if tokenizer and self.inspect_formatted:
                            try:
                                formatted_prompt = format_with_chat_template(request, tokenizer)
                                inspect_formatted_request(
                                    formatted_prompt,
                                    console=console,
                                    task_name=spec,
                                    native_id=native_id,
                                )
                            except Exception as e:
                                logger.error(f"Error formatting request: {e}")

                        if tokenizer and self.inspect_tokens:
                            try:
                                tokens = tokenize_request(request, tokenizer)
                                inspect_tokens(
                                    tokens,
                                    tokenizer,
                                    console=console,
                                    task_name=spec,
                                    native_id=native_id,
                                )
                            except Exception as e:
                                logger.error(f"Error tokenizing request: {e}")

    def _setup_workers(
        self,
        items: list[QueueItem],
        ctx: Any,
    ) -> tuple[mp.Queue, mp.Queue, int]:
        """Setup queues and compute worker allocation."""
        total_instances = len(items)
        logger.info(f"Total instances: {total_instances}")

        instance_queue: mp.Queue = ctx.Queue()
        result_queue: mp.Queue = ctx.Queue()

        num_workers = self._get_num_workers()
        logger.info(f"Total workers: {num_workers}")

        return instance_queue, result_queue, num_workers

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

    async def _process_results(
        self,
        trackers: dict[str, TaskTracker],
        result_queue: mp.Queue,
        instance_queue: mp.Queue,
        workers: list[mp.process.BaseProcess],
        total_tasks: int,
        total_instances: int,
    ) -> dict[str, Any]:
        """Process results from workers."""
        from olmo_eval.runners.utils import TaskResult

        results: dict[str, TaskResult] = {}
        completed_tasks = 0

        # Pre-add error tasks to results
        for spec, tracker in trackers.items():
            if tracker.error:
                task_result = finalize_task(tracker)
                results[spec] = task_result
                completed_tasks += 1
                self._report_task_completion(self.model_name, task_result)

        pending_instances = total_instances
        last_health_check = time.time()
        health_check_interval = 5.0

        while completed_tasks < total_tasks and pending_instances > 0:
            try:
                result_item: ResultItem = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: result_queue.get(timeout=1.0)
                )
            except queue.Empty:
                if time.time() - last_health_check > health_check_interval:
                    check_workers_alive(workers, result_queue)
                    last_health_check = time.time()
                continue

            # Check for fatal worker crash
            if result_item.task_id == "__WORKER_FATAL__":
                logger.error("FATAL: Worker crashed!")
                logger.error(result_item.error)
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                for mp_queue in [instance_queue, result_queue]:
                    mp_queue.cancel_join_thread()
                raise RuntimeError(f"Worker process crashed: {result_item.error}")

            tracker = trackers[result_item.task_id]

            if tracker.error:
                pending_instances -= 1
                continue

            if result_item.error:
                tracker.error = f"Instance {result_item.instance_idx} failed: {result_item.error}"
                pending_instances -= 1
                if tracker.is_complete():
                    task_result = finalize_task(tracker)
                    results[result_item.task_id] = task_result
                    completed_tasks += 1
                    self._report_task_completion(self.model_name, task_result)
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
                    results[result_item.task_id] = task_result
                    completed_tasks += 1
                    self._report_task_completion(self.model_name, task_result)
                    if self.save_predictions and task_result.predictions:
                        task_hash = compute_task_hash(task_result.config)
                        self._write_predictions(
                            self.model_name, task_result.spec, task_result.predictions, task_hash
                        )

        return results

    def _report_task_completion(self, model_name: str, result: Any) -> None:
        """Report when a task completes."""
        label = f"{model_name}:{result.spec}"
        if result.error:
            logger.error(f"✗ {label} (ERROR: {result.error})")
        else:
            logger.info(
                f"✓ {label} ({result.num_instances} instances, {result.duration_seconds:.1f}s)"
            )

    def _aggregate_results(
        self,
        results: dict[str, Any],
        expanded_tasks: list[str],
    ) -> dict[str, Any]:
        """Aggregate results and prepare final output."""
        from olmo_eval.runners.mixins import get_model_display_name

        errors = [(spec, r) for spec, r in results.items() if r.error]
        if errors:
            logger.error(f"{len(errors)} tasks failed")
            for spec, error_result in errors:
                logger.error(f"  {spec}: {error_result.error}")

        try:
            provider_type = ProviderType(self.provider_config.get_provider_name())
            provider_str = provider_type.value
        except ValueError:
            provider_str = "vllm"

        display_model_name = get_model_display_name(
            self.provider_config.model, self.provider_config.alias
        )

        results_dict: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "model": display_model_name,
            "model_path": self.provider_config.model,
            "provider": provider_str,
            "tasks": {},
            "errors": [],
        }

        for spec in expanded_tasks:
            if spec in results:
                task_result = results[spec]
                if task_result.error:
                    results_dict["errors"].append({"spec": spec, "error": task_result.error})
                else:
                    task_data = task_result.to_dict(include_predictions=True)
                    task_hash = compute_task_hash(task_result.config)
                    if task_hash:
                        task_data["task_hash"] = task_hash
                    results_dict["tasks"][spec] = task_data

        model_config_dict = self.provider_config.to_dict()
        model_config_dict["attention_backend"] = self.attention_backend
        results_dict["model_config"] = model_config_dict

        suite_aggs = compute_suite_aggregations(self.task_specs, results_dict["tasks"])
        if suite_aggs:
            results_dict["suites"] = suite_aggs

        return results_dict

    def _finalize_and_save(
        self,
        results_dict: dict[str, Any],
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log summary, write metrics, upload to S3, and save results."""
        from olmo_eval.core.types import compute_model_hash
        from olmo_eval.runners.metrics import log_summary, write_metrics_json
        from olmo_eval.runners.storage import save_results, upload_to_s3

        log_summary(results_dict, multi_model=False)

        experiment_id = generate_experiment_id()
        model_hash = compute_model_hash(results_dict.get("model_config", {}))
        results_dict["_model_hash"] = model_hash

        write_metrics_json(
            output_dir=self.output_dir,
            results=results_dict,
            multi_model=False,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            model_hash=model_hash,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        s3_location: str | None = None
        if self.s3_config and model_hash:
            s3_location = upload_to_s3(
                output_dir=self.output_dir,
                s3_config=self.s3_config,
                model_name=self.model_name,
                model_hash=model_hash,
                experiment_id=experiment_id,
            )

        results_dict["_experiment_id"] = experiment_id
        results_dict["_s3_location"] = s3_location

        save_results(
            results=results_dict,
            storages=self.storages,
            s3_config=self.s3_config,
            experiment_id=experiment_id,
            model_hash=model_hash,
            s3_location=s3_location,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        return results_dict

    def _write_predictions(
        self, model_name: str, spec: str, predictions: list[dict], task_hash: str | None = None
    ) -> None:
        """Write per-instance predictions to JSONL."""
        from olmo_eval.runners.writers import write_predictions_jsonl

        write_predictions_jsonl(self.output_dir, spec, predictions, model_name, task_hash=task_hash)

    def _write_requests(
        self, model_name: str, spec: str, requests: list[dict], task_hash: str | None = None
    ) -> None:
        """Write per-instance requests to JSONL."""
        from olmo_eval.runners.writers import write_requests_jsonl

        write_requests_jsonl(self.output_dir, spec, requests, model_name, task_hash=task_hash)

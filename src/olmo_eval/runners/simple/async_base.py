"""Shared functionality for async evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import time
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.core.configs import expand_tasks, get_model_config
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import Response
from olmo_eval.runners.base import BaseEvalRunner
from olmo_eval.runners.mixins import AsyncRunnerMixin, S3Config
from olmo_eval.runners.simple.helpers import check_workers_alive
from olmo_eval.runners.simple.queue import (
    QueueItem,
    ResultItem,
    TaskTracker,
    build_requests_from_items,
    finalize_task,
    prepare_task_items,
)
from olmo_eval.runners.utils import (
    compute_suite_aggregations,
    compute_task_hash,
    generate_experiment_id,
)

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = get_logger(__name__)


@dataclass
class AsyncBaseRunner(AsyncRunnerMixin, BaseEvalRunner):
    """Base class for async evaluation runners.

    Provides common infrastructure for AsyncEvalRunner and StreamingEvalRunner.
    """

    model_names: list[str] = field(default_factory=list)
    task_specs: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    storages: list[StorageBackend] = field(default_factory=list)

    # Multi-worker config
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # vLLM config
    attention_backend: str | None = None

    # Per-task overrides from inline spec (e.g., task::temperature=0.6)
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-model overrides from inline spec
    model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # S3 upload configuration (optional)
    s3_config: S3Config | None = None

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None
    alias: str | None = None

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
    _mode_description: str = "Async"

    def _prepare_tasks(
        self,
    ) -> tuple[
        list[str],
        dict[tuple[str, str], TaskTracker],
        dict[str, list[QueueItem]],
        dict[str, Any],
    ]:
        """Prepare all tasks and return tracking data structures.

        Returns:
            Tuple of (expanded_tasks, trackers, model_items, model_configs)
        """
        expanded_tasks = expand_tasks(self.task_specs)

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
                overrides, sampling_overrides = self._build_task_overrides(spec)
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
                    # Write requests early
                    if self.save_requests and items and tracker.task:
                        request_objects = build_requests_from_items(items, tracker.task.config.name)
                        task_hash = compute_task_hash(tracker.task.config.to_dict())
                        self._write_requests(model_name, spec, request_objects, task_hash)

        # Optionally inspect first instance of each task
        if (
            self.inspect_instance
            or self.inspect_formatted
            or self.inspect_tokens
            or self.inspect_request
        ):
            from olmo_eval.core.inspection import (
                format_with_chat_template,
                inspect_formatted_request,
                inspect_instance,
                inspect_request,
                inspect_tokens,
                load_tokenizer,
                tokenize_request,
            )

            inspected_tasks: set[str] = set()
            tokenizer = None

            # Load tokenizer once for formatted/token inspection
            if self.inspect_formatted or self.inspect_tokens:
                # Use the first model's tokenizer config
                first_model = self.model_names[0] if self.model_names else None
                if first_model:
                    first_model_config = model_configs.get(first_model)
                    if first_model_config:
                        tokenizer_name = first_model_config.tokenizer or first_model_config.model
                        try:
                            tokenizer = load_tokenizer(tokenizer_name)
                        except Exception as e:
                            console.print(
                                f"[yellow]Warning:[/yellow] Could not load tokenizer: {e}"
                            )

            for key, tracker in trackers.items():
                spec = key[1]
                if spec not in inspected_tasks and tracker.task and not tracker.error:
                    first_instance = next(iter(tracker.task.instances), None)
                    if first_instance:
                        if self.inspect_instance:
                            console.print()
                            inspect_instance(
                                first_instance, console=console, task_name=spec, index=0
                            )

                        # Get request for inspection
                        if self.inspect_request or (
                            tokenizer and (self.inspect_formatted or self.inspect_tokens)
                        ):
                            request = tracker.task.format_request(first_instance)

                            if self.inspect_request:
                                inspect_request(
                                    request,
                                    console=console,
                                    title=f"[bold]Request[/bold] ({spec})",
                                )

                            if tokenizer and self.inspect_formatted:
                                try:
                                    formatted_prompt = format_with_chat_template(request, tokenizer)
                                    inspect_formatted_request(
                                        formatted_prompt,
                                        console=console,
                                        title=f"[bold]Formatted Prompt[/bold] ({spec})",
                                    )
                                except Exception as e:
                                    console.print(f"[red]Error formatting request:[/red] {e}")

                            if tokenizer and self.inspect_tokens:
                                try:
                                    tokens = tokenize_request(request, tokenizer)
                                    inspect_tokens(
                                        tokens,
                                        tokenizer,
                                        console=console,
                                        title=f"[bold]Token IDs[/bold] ({spec})",
                                    )
                                except Exception as e:
                                    console.print(f"[red]Error tokenizing request:[/red] {e}")

                        inspected_tasks.add(spec)

        return expanded_tasks, trackers, model_items, model_configs

    def _setup_workers(
        self,
        model_items: dict[str, list[QueueItem]],
        model_configs: dict[str, Any],
        ctx: Any,
    ) -> tuple[dict[str, mp.Queue], mp.Queue, int, int, int]:
        """Setup queues and compute worker allocation.

        Returns:
            Tuple of (model_queues, result_queue, total_workers, workers_per_model, gpus_per_model)
        """
        total_instances = sum(len(items) for items in model_items.values())
        console.print(f"[bold]Total instances:[/bold] {total_instances}")

        model_queues: dict[str, mp.Queue] = {m: ctx.Queue() for m in self.model_names}
        result_queue: mp.Queue = ctx.Queue()

        total_gpus = self._get_total_gpus()
        total_workers = self._get_num_workers()
        num_models = len(self.model_names)
        workers_per_model = max(1, total_workers // num_models)
        gpus_per_model = max(0, total_gpus // num_models) if total_gpus > 0 else 0

        console.print(f"[bold]Total workers:[/bold] {total_workers}")
        console.print(f"[bold]Workers per model:[/bold] {workers_per_model}")
        console.print(f"[bold]GPUs per model:[/bold] {gpus_per_model}")

        return model_queues, result_queue, total_workers, workers_per_model, gpus_per_model

    async def _process_results(
        self,
        trackers: dict[tuple[str, str], TaskTracker],
        result_queue: mp.Queue,
        model_queues: dict[str, mp.Queue],
        workers: list[mp.process.BaseProcess],
        total_pairs: int,
        total_instances: int,
    ) -> dict[tuple[str, str], Any]:
        """Process results from workers.

        Returns:
            Dict mapping (model_name, spec) to TaskResult
        """
        from olmo_eval.runners.utils import TaskResult

        results: dict[tuple[str, str], TaskResult] = {}
        completed_pairs = 0

        # Pre-add error tasks to results
        for key, tracker in trackers.items():
            if tracker.error:
                task_result = finalize_task(tracker)
                results[key] = task_result
                completed_pairs += 1
                self._report_task_completion(tracker.model_name, task_result)

        pending_instances = total_instances
        last_health_check = time.time()
        health_check_interval = 5.0

        while completed_pairs < total_pairs and pending_instances > 0:
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
                console.print("\n[bold red]FATAL: Worker crashed![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                for mp_queue in list(model_queues.values()) + [result_queue]:
                    mp_queue.cancel_join_thread()
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
                    if self.save_predictions and task_result.predictions:
                        task_hash = compute_task_hash(task_result.config)
                        self._write_predictions(
                            tracker.model_name, task_result.spec, task_result.predictions, task_hash
                        )

        return results

    def _aggregate_results(
        self,
        results: dict[tuple[str, str], Any],
        expanded_tasks: list[str],
        model_configs: dict[str, Any],
        provider_name: str,
    ) -> dict[str, Any]:
        """Aggregate results by model and prepare final output."""
        from olmo_eval.inference import ProviderType
        from olmo_eval.runners.mixins import get_model_display_name

        errors = [(k, r) for k, r in results.items() if r.error]
        if errors:
            console.print(
                f"\n[bold red]Errors:[/bold red] {len(errors)} (model, task) pairs failed"
            )
            for (model_name, spec), error_result in errors:
                console.print(f"  - {model_name}:{spec}: {error_result.error}")

        results_dict: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
            "errors": [],
        }

        for model_name in self.model_names:
            model_config = model_configs[model_name]
            try:
                provider_type = ProviderType(model_config.provider)
                provider_str = provider_type.value
            except ValueError:
                provider_str = provider_name

            model_alias = self.model_overrides.get(model_name, {}).get("alias")
            display_model_name = get_model_display_name(model_config.model, model_alias)

            model_results: dict[str, Any] = {
                "model": display_model_name,
                "model_path": model_config.model,
                "provider": provider_str,
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
                        task_data = task_result.to_dict(include_predictions=True)
                        task_hash = compute_task_hash(task_result.config)
                        if task_hash:
                            task_data["task_hash"] = task_hash
                        model_results["tasks"][spec] = task_data

            model_config_dict = model_config.to_dict()
            model_config_dict["attention_backend"] = self.attention_backend
            model_results["model_config"] = model_config_dict

            results_dict["models"][model_name] = model_results

            suite_aggs = compute_suite_aggregations(self.task_specs, model_results["tasks"])
            if suite_aggs:
                model_results["suites"] = suite_aggs

        return results_dict

    def _finalize_and_save(
        self,
        results_dict: dict[str, Any],
        experiment_duration_seconds: float | None = None,
        provider_init_seconds: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Log summary, write metrics, upload to S3, and save results."""
        from olmo_eval.core.types import compute_model_hash

        self._log_summary(results_dict, multi_model=True)

        experiment_id = generate_experiment_id()

        for model_data in results_dict.get("models", {}).values():
            model_hash = compute_model_hash(model_data.get("model_config", {}))
            model_data["_model_hash"] = model_hash

        self._write_metrics_json(
            results_dict,
            multi_model=True,
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        for model_name, model_data in results_dict.get("models", {}).items():
            model_hash = model_data.get("_model_hash")
            s3_location: str | None = None

            if self.s3_config and model_hash:
                s3_location = self._upload_to_s3(
                    model_name=model_name,
                    model_hash=model_hash,
                    experiment_id=experiment_id,
                )

            model_data["_experiment_id"] = experiment_id
            model_data["_s3_location"] = s3_location

        self._save_results(
            results_dict,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

        return results_dict

    @abstractmethod
    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations asynchronously. Subclasses must implement."""
        ...

    def run(self) -> dict[str, Any]:
        """Sync wrapper for async execution."""
        return asyncio.run(self.run_async())


__all__ = ["AsyncBaseRunner"]

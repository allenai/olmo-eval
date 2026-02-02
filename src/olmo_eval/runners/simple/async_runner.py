"""Async evaluation runner with instance-level queuing."""

from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from rich.console import Console

from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.core.literals import ProviderLiteral
from olmo_eval.core.logging import get_logger, get_worker_id
from olmo_eval.inference import ProviderType
from olmo_eval.runners.mixins import S3Config
from olmo_eval.runners.simple.async_base import AsyncBaseRunner
from olmo_eval.runners.simple.helpers import wait_for_workers_ready
from olmo_eval.runners.simple.workers import instance_worker_process

if TYPE_CHECKING:
    from olmo_eval.storage import StorageBackend

console = Console()
logger = get_logger(__name__)


@dataclass
class AsyncEvalRunner(AsyncBaseRunner):
    """Async evaluation runner with instance-level queuing.

    Uses per-model queues where instances from all tasks are mixed together,
    enabling better GPU utilization and early completion reporting.
    Supports multiple models in a single run, producing results for each
    unique (model, task) pair.
    """

    model_names: list[str] = field(default_factory=list)
    task_specs: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    provider_override: str | None = None
    storages: list[StorageBackend] = field(default_factory=list)

    # Multi-worker config
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # vLLM config
    attention_backend: str | None = None

    # Per-task overrides
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Per-model overrides
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

    # Configuration for print_config display
    _mode_name: str = "Async Mode"
    _mode_description: str = "Async (All-at-once)"

    async def run_async(self) -> dict[str, Any]:
        """Execute evaluations using instance-level queuing with multi-model support."""
        import time

        # Track experiment start time
        experiment_start = time.time()

        # Prepare tasks
        expanded_tasks, trackers, model_items, model_configs = self._prepare_tasks()

        # Apply provider override if specified
        for model_name in self.model_names:
            if self.provider_override:
                model_configs[model_name].provider = cast(ProviderLiteral, self.provider_override)

        total_pairs = len(self.model_names) * len(expanded_tasks)
        total_instances = sum(len(items) for items in model_items.values())

        # Setup multiprocessing
        ctx = mp.get_context("spawn")
        model_queues, result_queue, total_workers, workers_per_model, gpus_per_model = (
            self._setup_workers(model_items, model_configs, ctx)
        )
        total_gpus = self._get_total_gpus()

        # Create shared dict for tracking worker init times
        manager = ctx.Manager()
        init_times = manager.dict()  # DictProxy[str, float]

        # Shuffle and enqueue items per model
        for model_name, items in model_items.items():
            random.shuffle(items)
            for item in items:
                model_queues[model_name].put(item)

        # Add poison pills AFTER all items are enqueued
        for model_name in self.model_names:
            for _ in range(workers_per_model):
                model_queues[model_name].put(None)

        # Start workers for each model
        workers: list[mp.process.BaseProcess] = []
        gpu_offset = 0

        for model_name in self.model_names:
            model_config = model_configs[model_name]
            provider_type = ProviderType(model_config.provider)

            # Get per-model vLLM loading options
            per_model_overrides = self.model_overrides.get(model_name, {})
            effective_load_format = per_model_overrides.get("load_format")
            effective_extra_loader_config = per_model_overrides.get("extra_loader_config")

            for i in range(workers_per_model):
                worker_id = get_worker_id(model_config.model, i)

                if total_gpus > 0:
                    start_gpu = gpu_offset + (i * self.gpus_per_worker)
                    end_gpu = min(start_gpu + self.gpus_per_worker, gpu_offset + gpus_per_model)
                    gpu_ids = list(range(start_gpu, end_gpu)) if start_gpu < end_gpu else []
                else:
                    gpu_ids = []

                worker = ctx.Process(
                    target=instance_worker_process,
                    args=(
                        worker_id,
                        gpu_ids,
                        model_queues[model_name],
                        result_queue,
                        model_config.model,
                        provider_type.value,
                        self.attention_backend,
                        model_config.tokenizer,
                        model_config.max_model_len,
                        effective_load_format,
                        effective_extra_loader_config,
                        init_times,
                    ),
                )
                worker.start()
                workers.append(worker)

            gpu_offset += gpus_per_model

        console.print(
            f"[bold green]{len(workers)} worker(s) started across "
            f"{len(self.model_names)} model(s), processing instances...[/bold green]"
        )

        # Wait for workers to initialize
        console.print("[dim]Waiting for workers to initialize...[/dim]")
        wait_for_workers_ready(workers, result_queue, startup_timeout=60.0)
        console.print("[dim]Workers initialized successfully[/dim]")

        # Capture init times from workers (convert manager dict to regular dict)
        provider_init_seconds = dict(init_times)

        # Process results
        results = await self._process_results(
            trackers, result_queue, model_queues, workers, total_pairs, total_instances
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
        results_dict = self._aggregate_results(results, expanded_tasks, model_configs, "vllm")
        return self._finalize_and_save(
            results_dict,
            experiment_duration_seconds=experiment_duration_seconds,
            provider_init_seconds=provider_init_seconds,
        )

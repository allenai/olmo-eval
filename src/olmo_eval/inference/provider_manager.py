"""Manager for multiple provider instances across subprocesses."""

from __future__ import annotations

import multiprocessing as mp
from typing import TYPE_CHECKING, Any

from olmo_eval.common.logging import configure_worker_logging

if TYPE_CHECKING:
    from olmo_eval.harness.config import HarnessConfig


class ProviderManager:
    """Manages multiple provider instances across subprocesses.

    Mirrors SandboxManager pattern:
    - SandboxManager: N sandbox executors, round-robin dispatch
    - ProviderManager: N inference workers, shared queue (natural load balancing)

    Each worker:
    - Gets a subset of GPUs (gpu_ids divided evenly)
    - Pulls from shared item_queue (natural load balancing)
    - Writes to shared result_queue
    """

    def __init__(
        self,
        harness_config: HarnessConfig,
        num_inference_workers: int,
        gpu_ids: list[int],
        item_queue: mp.Queue,
        result_queue: mp.Queue,
        output_dir: str | None = None,
    ):
        """Initialize the provider manager.

        Args:
            harness_config: Configuration for harness/provider.
            num_inference_workers: Number of parallel inference workers to spawn.
            gpu_ids: List of GPU IDs to distribute across workers.
            item_queue: Shared queue for work items (workers pull from this).
            result_queue: Shared queue for results (workers write to this).
            output_dir: Output directory for logs and results.
        """
        self._harness_config = harness_config
        self._num_inference_workers = num_inference_workers
        self._gpu_ids = gpu_ids
        self._item_queue = item_queue
        self._result_queue = result_queue
        self._output_dir = output_dir
        self._processes: list[mp.process.BaseProcess] = []
        self._logger = configure_worker_logging("provider-mgr")

    def start(
        self,
        ctx: Any,
        total_instances: int,
        init_times: Any,
    ) -> list[mp.process.BaseProcess]:
        """Start all provider instances as subprocesses.

        Each instance:
        - Gets a subset of GPUs (gpu_ids divided evenly)
        - Pulls from shared item_queue (natural load balancing)
        - Writes to shared result_queue

        Args:
            ctx: Multiprocessing context (e.g., from mp.get_context("spawn")).
            total_instances: Total number of instances to process.
            init_times: Shared dict for tracking worker init times.

        Returns:
            List of started worker processes.
        """
        from olmo_eval.common.logging import get_worker_id
        from olmo_eval.runners.asynq.workers import inference_worker

        gpus_per_instance = len(self._gpu_ids) // self._num_inference_workers
        harness_config_dict = self._harness_config.to_dict()

        for i in range(self._num_inference_workers):
            start_gpu = i * gpus_per_instance
            instance_gpus = self._gpu_ids[start_gpu : start_gpu + gpus_per_instance]
            worker_id = get_worker_id(self._harness_config.provider.model, i)

            proc = ctx.Process(
                target=inference_worker,
                args=(
                    worker_id,
                    instance_gpus,
                    self._item_queue,
                    self._result_queue,
                    harness_config_dict,
                    total_instances,
                    init_times,
                    self._output_dir,
                    self._num_inference_workers,
                ),
            )
            proc.start()
            self._processes.append(proc)

        self._logger.info(
            f"Started {self._num_inference_workers} provider instance(s) "
            f"({gpus_per_instance} GPU(s) each)"
        )

        return self._processes

    def add_poison_pills(self) -> None:
        """Add termination signals (one per instance)."""
        for _ in range(self._num_inference_workers):
            self._item_queue.put(None)

    def join(self, timeout: float | None = 10.0) -> None:
        """Wait for all instances to complete.

        Args:
            timeout: Timeout in seconds for each worker join (None = no timeout).
        """
        for proc in self._processes:
            proc.join(timeout=timeout)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        self._logger.info("All provider instances stopped")

    @property
    def processes(self) -> list[mp.process.BaseProcess]:
        """Get the list of worker processes."""
        return self._processes


def validate_inference_workers(num_inference_workers: int, num_gpus: int) -> None:
    """Validate num_inference_workers configuration.

    Args:
        num_inference_workers: Number of parallel inference workers.
        num_gpus: Total number of available GPUs.

    Raises:
        ValueError: If configuration is invalid.
    """
    if num_inference_workers < 1:
        raise ValueError(f"num_inference_workers must be >= 1, got {num_inference_workers}")

    if num_gpus == 0:
        # CPU-only mode - only 1 worker makes sense
        if num_inference_workers > 1:
            raise ValueError(
                f"num_inference_workers={num_inference_workers} but no GPUs available. "
                "Data parallelism requires GPUs."
            )
        return

    if num_inference_workers > num_gpus:
        raise ValueError(
            f"num_inference_workers ({num_inference_workers}) > num_gpus ({num_gpus}). "
            "Cannot have more workers than GPUs."
        )

    if num_gpus % num_inference_workers != 0:
        raise ValueError(
            f"num_gpus ({num_gpus}) must be evenly divisible by "
            f"num_inference_workers ({num_inference_workers}). "
            f"Got remainder {num_gpus % num_inference_workers}."
        )


__all__ = ["ProviderManager", "validate_inference_workers"]

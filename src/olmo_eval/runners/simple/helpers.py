"""Helper functions for async evaluation runners."""

from __future__ import annotations

import asyncio
import dataclasses
import multiprocessing as mp
import queue
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.core.logging import get_logger
from olmo_eval.inference import InferenceProvider
from olmo_eval.runners.simple.queue import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.core.harness import Harness

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Worker health monitoring
# -----------------------------------------------------------------------------


def terminate_workers(
    workers: list[mp.process.BaseProcess],
    timeout: float = 5.0,
) -> None:
    """Terminate all worker processes and wait for them to exit.

    Args:
        workers: List of worker processes to terminate.
        timeout: Maximum time to wait for each worker to terminate.
    """
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
    for worker in workers:
        worker.join(timeout=timeout)
        if worker.is_alive():
            # Force kill if still alive
            worker.kill()
            worker.join(timeout=1)


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
                logger.error("FATAL: Worker crashed!")
                logger.error(result_item.error)
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
    except queue.Empty:
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
                logger.error("FATAL: Worker failed during startup!")
                logger.error(result_item.error)
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
        except queue.Empty:
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
# Batch processing
# -----------------------------------------------------------------------------


def process_batch(
    batch: list[QueueItem],
    provider: InferenceProvider,
    result_queue: mp.Queue,
    harness: Harness | None = None,
) -> None:
    """Process a batch of instances through the provider or harness.

    For harness-based requests (with or without tools), uses async harness.run()
    to stream results as they complete. For provider-only requests, groups by
    request type and uses batch generation.

    Args:
        batch: List of QueueItems to process
        provider: InferenceProvider instance
        result_queue: Queue to put results
        harness: Optional Harness instance for tool/prompt configuration
    """
    from tqdm import tqdm

    with tqdm(total=len(batch), desc="Processing instances", unit="inst") as pbar:
        # If harness is configured, use async harness.run() for all requests
        # This handles both tool and non-tool cases uniformly
        if harness is not None:
            _process_with_harness(batch, harness, result_queue, pbar)
            return

        # Provider-only path: group by request type for batch efficiency
        _process_with_provider(batch, provider, result_queue, pbar)


def _process_with_harness(
    batch: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    pbar: Any,
) -> None:
    """Process batch using harness.run(), streaming results as they complete."""
    from dataclasses import replace as dataclass_replace

    async def run_batch_async() -> None:
        semaphore = asyncio.Semaphore(harness.config.max_concurrency or 8)

        async def run_one(item: QueueItem) -> None:
            async with semaphore:
                try:
                    harness_result = await harness.run(item.request, item.sampling_params)

                    final_output = harness_result.final_output

                    # Add trajectory metadata if present (for tool-based execution)
                    if harness_result.trajectory.num_turns > 1 or harness.config.has_tools:
                        output_with_metadata = dataclass_replace(
                            final_output,
                            metadata={
                                **(final_output.metadata or {}),
                                "trajectory": harness_result.trajectory.to_dict(),
                                "max_turns_reached": harness_result.max_turns_reached,
                                "total_tool_calls": harness_result.total_tool_calls,
                                "num_turns": harness_result.num_turns,
                            },
                        )
                    else:
                        output_with_metadata = final_output

                    result_queue.put(
                        ResultItem(
                            model_name=item.model_name,
                            task_id=item.task_id,
                            instance_idx=item.instance_idx,
                            instance=item.instance,
                            request=item.request,
                            outputs=[output_with_metadata],
                            error=harness_result.error,
                            attempt=item.attempt,
                        )
                    )
                except Exception as e:
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
                pbar.update(1)

        await asyncio.gather(*[run_one(item) for item in batch])

    asyncio.run(run_batch_async())


def _process_with_provider(
    batch: list[QueueItem],
    provider: InferenceProvider,
    result_queue: mp.Queue,
    pbar: Any,
) -> None:
    """Process batch using provider directly, grouping by request type."""
    from olmo_eval.core.types import RequestType

    # Group items by (request_type, sampling_params) for batch efficiency
    groups: dict[tuple, list[QueueItem]] = {}
    for item in batch:
        sp_key = dataclasses.astuple(item.sampling_params) if item.sampling_params else None
        key = (item.request.request_type, sp_key)
        groups.setdefault(key, []).append(item)

    for group in groups.values():
        requests = [item.request for item in group]
        sampling_params = group[0].sampling_params
        request_type = group[0].request.request_type

        try:
            if request_type == RequestType.LOGLIKELIHOOD:
                outputs_list = provider.logprobs(requests)
            else:
                outputs_list = provider.generate(requests, sampling_params)

            for item, outputs in zip(group, outputs_list, strict=True):
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
                pbar.update(1)
        except Exception as e:
            for item in group:
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
                pbar.update(1)


__all__ = [
    "check_workers_alive",
    "wait_for_workers_ready",
    "process_batch",
]

"""Worker health monitoring for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.asynq.types import WORKER_FATAL_TASK_ID

logger = get_logger(__name__)


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
            if result_item.task_id == WORKER_FATAL_TASK_ID:
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
    """Wait for workers to start and check for early failures.

    Args:
        workers: List of worker processes
        result_queue: Queue to check for fatal error markers
        startup_timeout: How long to wait for workers to stabilize

    Raises:
        RuntimeError: If workers fail during startup
    """
    start_time = time.time()
    check_interval = 0.5

    def drain_fatal_errors() -> None:
        """Check queue for fatal errors and raise if found."""
        while True:
            try:
                result_item = result_queue.get_nowait()
                if result_item.task_id == WORKER_FATAL_TASK_ID:
                    # Terminate all workers
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                            worker.join(timeout=5)
                    result_queue.cancel_join_thread()
                    raise RuntimeError(f"Worker failed during startup: {result_item.error}")
                else:
                    # Put non-fatal item back
                    result_queue.put(result_item)
                    return
            except queue.Empty:
                return

    while time.time() - start_time < startup_timeout:
        time.sleep(check_interval)

        # Check for fatal errors in queue
        drain_fatal_errors()

        # Check if any worker died with non-zero exit code
        for worker in workers:
            if not worker.is_alive() and worker.exitcode is not None and worker.exitcode != 0:
                # Drain queue one more time to get the error message
                drain_fatal_errors()
                raise RuntimeError(f"Worker died during startup with exit code {worker.exitcode}")

        # If all workers are alive, do one final queue check before returning
        if all(w.is_alive() for w in workers):
            drain_fatal_errors()
            return

    # Final check
    check_workers_alive(workers, result_queue)


def wait_for_init_times(
    init_times: Any,
    num_workers: int,
    timeout: float = 300.0,
    check_interval: float = 1.0,
) -> dict[str, float]:
    """Wait for all workers to report their initialization times.

    Args:
        init_times: Shared manager dict that workers write their init times to.
        num_workers: Expected number of workers.
        timeout: Maximum time to wait for all init times.
        check_interval: How often to check for new entries.

    Returns:
        Dictionary mapping worker_id to init time in seconds.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        if len(init_times) >= num_workers:
            return dict(init_times)
        time.sleep(check_interval)

    # Return what we have even if incomplete
    logger.warning(f"Timed out waiting for init times: got {len(init_times)}/{num_workers} workers")
    return dict(init_times)


__all__ = [
    "terminate_workers",
    "check_workers_alive",
    "wait_for_workers_ready",
    "wait_for_init_times",
]

"""Helper functions for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import TYPE_CHECKING

from olmo_eval.core.logging import get_logger
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
    """Wait for workers to start and check for early failures.

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
# Request processing
# -----------------------------------------------------------------------------


def _format_error_detail(exc: Exception) -> str:
    """Format exception with HTTP details for debugging."""
    parts = [f"type: {type(exc).__qualname__}"]

    # HTTP status code
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status_code: {status}")

    # Request URL from response
    response = getattr(exc, "response", None)
    if response is not None:
        url = getattr(response, "url", None)
        if url is not None:
            parts.append(f"url: {url}")

    # Error message
    message = getattr(exc, "message", None) or str(exc)
    if len(message) > 500:
        message = message[:500] + "..."
    parts.append(f"message: {message}")

    # Root cause (e.g., httpx.ReadTimeout)
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"cause: {type(cause).__qualname__}: {cause}")

    return " | ".join(parts)


async def process_request(
    item: QueueItem,
    harness: Harness,
    result_queue: mp.Queue,
) -> None:
    """Process a single request via harness.

    Handles CHAT, COMPLETION, and LOGLIKELIHOOD request types with a unified
    interface. CHAT requests use harness.run(), others use generate/logprobs.

    Args:
        item: Queue item to process.
        harness: Harness instance for execution.
        result_queue: Queue to put results.
    """
    import asyncio
    from dataclasses import replace as dataclass_replace

    from olmo_eval.core.types import RequestType

    try:
        request_type = item.request.request_type

        if request_type == RequestType.CHAT:
            harness_result = await harness.run(item.request, item.sampling_params)
            final_output = harness_result.final_output

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

        else:
            # TODO(undfined): Come back to this as I don't like it being something different.
            # Why can't we just use the async methods?
            # COMPLETION or LOGLIKELIHOOD - run sync methods in executor
            loop = asyncio.get_event_loop()

            if request_type == RequestType.LOGLIKELIHOOD:
                outputs_list = await loop.run_in_executor(None, harness.logprobs, [item.request])
            else:
                outputs_list = await loop.run_in_executor(
                    None, harness.generate, [item.request], item.sampling_params
                )

            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=item.request,
                    outputs=outputs_list[0],
                    error=None,
                    attempt=item.attempt,
                )
            )

    except Exception as e:
        error_detail = _format_error_detail(e)
        logger.warning(f"Error on instance {item.instance_idx}: {error_detail}")

        result_queue.put(
            ResultItem(
                model_name=item.model_name,
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=item.request,
                outputs=[],
                error=error_detail,
                attempt=item.attempt,
            )
        )


__all__ = [
    "check_workers_alive",
    "wait_for_workers_ready",
    "process_request",
]

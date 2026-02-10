"""Helper functions for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import TYPE_CHECKING

from olmo_eval.core.logging import get_logger
from olmo_eval.runners.asynq.queue import WORKER_FATAL_TASK_ID, QueueItem, ResultItem

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


async def process_chat_request(
    item: QueueItem,
    harness: Harness,
    result_queue: mp.Queue,
) -> None:
    """Process a single CHAT request via harness.run().

    CHAT requests use the async harness.run() method which handles agentic
    loops with tool calls. These must be processed individually.

    Args:
        item: Queue item to process (must be CHAT type).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
    """
    from dataclasses import replace as dataclass_replace

    try:
        harness_result = await harness.run(item.request, item.sampling_params)
        final_output = harness_result.final_output

        if harness_result.trajectory is not None:
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
        error_detail = _format_error_detail(e)
        logger.warning(f"Error on CHAT instance {item.instance_idx}: {error_detail}")

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


async def process_batch(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
) -> None:
    """Process a batch of COMPLETION or LOGLIKELIHOOD requests.

    All items must have the same request_type and sampling_params.
    Calls harness.agenerate or harness.alogprobs once for the entire batch.

    Args:
        items: List of queue items to process (same type and sampling_params).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
    """
    from olmo_eval.core.types import RequestType

    if not items:
        return

    request_type = items[0].request.request_type
    sampling_params = items[0].sampling_params
    requests = [item.request for item in items]

    try:
        if request_type == RequestType.LOGLIKELIHOOD:
            all_outputs = await harness.alogprobs(requests)
        else:
            all_outputs = await harness.agenerate(requests, sampling_params)

        # Map outputs back to individual items
        for item, outputs in zip(items, all_outputs, strict=True):
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
        # Batch failed - report error for all items
        error_detail = _format_error_detail(e)
        logger.warning(f"Batch error ({len(items)} items): {error_detail}")

        for item in items:
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
    "process_chat_request",
    "process_batch",
]

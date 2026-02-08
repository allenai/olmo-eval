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
        if harness is not None:
            _process_with_harness(batch, harness, result_queue, pbar)
            return

        # Provider-only path: group by request type for batch efficiency
        _process_with_provider(batch, provider, result_queue, pbar)


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


def _is_retryable_error(error: Exception) -> bool:
    """Check if an error is transient and should be retried."""
    # Check HTTP status code first (most reliable)
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        status_code = int(status_code)
        # Non-retryable: 400, 401, 403, 404, 422
        if status_code in (400, 401, 403, 404, 422):
            return False
        # Retryable: 429, 5xx
        if status_code in (429, 500, 502, 503, 504):
            return True

    # Check exception type name
    error_type = type(error).__name__
    if error_type in ("BadRequestError", "UnprocessableEntityError", "NotFoundError"):
        return False
    if error_type in ("APITimeoutError", "APIConnectionError", "RateLimitError"):
        return True

    # Fall back to string matching for errors without status codes
    error_str = str(error).lower()

    # Timeout errors are retryable
    if "timeout" in error_str or "timed out" in error_str:
        return True

    # Connection errors are retryable
    if "connection" in error_str:
        return True

    # Rate limit errors are retryable
    if "rate" in error_str and "limit" in error_str:
        return True

    # Server errors (5xx) are retryable
    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return True

    return "internal" in error_str and "error" in error_str


def _process_with_harness(
    batch: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    pbar: Any,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> None:
    """Process batch using harness.run(), streaming results as they complete.

    Chunks the batch to avoid overwhelming the HTTP connection pool. Each chunk
    is processed with bounded concurrency before moving to the next.

    Args:
        batch: Queue items to process.
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        pbar: Progress bar to update.
        max_retries: Maximum retry attempts for transient errors.
        retry_delay: Base delay between retries (exponential backoff).
    """
    from dataclasses import replace as dataclass_replace

    # Chunk size based on concurrency - process N * concurrency items at a time
    # This provides backpressure without overwhelming connection pools
    max_concurrency = harness.config.max_concurrency or 8
    chunk_size = max_concurrency * 8  # 64 items per chunk at default concurrency

    async def run_one(item: QueueItem, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            last_error: Exception | None = None
            attempt = 0

            while attempt <= max_retries:
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
                            attempt=attempt,
                        )
                    )
                    pbar.update(1)
                    return  # Success, exit retry loop

                except Exception as e:
                    last_error = e
                    attempt += 1

                    # Log full error details for debugging
                    error_detail = _format_error_detail(e)
                    logger.warning(
                        f"Error on instance {item.instance_idx} "
                        f"(attempt {attempt}/{max_retries + 1}): {error_detail}"
                    )

                    if attempt <= max_retries and _is_retryable_error(e):
                        delay = retry_delay * (2 ** (attempt - 1))
                        logger.info(f"Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                    else:
                        break  # Non-retryable or max retries exceeded

            # All retries exhausted or non-retryable error
            error_detail = _format_error_detail(last_error)

            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=item.request,
                    outputs=[],
                    error=error_detail,
                    attempt=attempt,
                )
            )
            pbar.update(1)

    async def run_chunk(chunk: list[QueueItem], semaphore: asyncio.Semaphore) -> None:
        await asyncio.gather(*[run_one(item, semaphore) for item in chunk])

    async def run_all_chunks() -> None:
        # Create semaphore once for all chunks to maintain consistent backpressure
        semaphore = asyncio.Semaphore(max_concurrency)

        # Process batch in chunks to avoid overwhelming connection pools
        num_chunks = (len(batch) + chunk_size - 1) // chunk_size
        for chunk_idx, i in enumerate(range(0, len(batch), chunk_size)):
            chunk = batch[i : i + chunk_size]
            if num_chunks > 1:
                logger.info(
                    f"Processing chunk {chunk_idx + 1}/{num_chunks} "
                    f"({len(chunk)} instances, concurrency={max_concurrency})"
                )
            await run_chunk(chunk, semaphore)

    asyncio.run(run_all_chunks())


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

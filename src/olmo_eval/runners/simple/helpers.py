"""Helper functions for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

from rich.console import Console

from olmo_eval.inference import InferenceProvider
from olmo_eval.runners.simple.queue import QueueItem, ResultItem

console = Console()


# -----------------------------------------------------------------------------
# Worker health monitoring
# -----------------------------------------------------------------------------


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
                console.print("\n[bold red]FATAL: Worker crashed![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
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
                console.print("\n[bold red]FATAL: Worker failed during startup![/bold red]")
                console.print(f"[red]{result_item.error}[/red]")
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
) -> None:
    """Process a batch of instances through the provider.

    Args:
        batch: List of QueueItems to process
        provider: InferenceProvider instance
        result_queue: Queue to put results
    """
    from olmo_eval.core.types import RequestType

    requests = [item.request for item in batch]
    sampling_params = batch[0].sampling_params if batch else None

    try:
        # Use logprobs for LOGLIKELIHOOD requests (e.g., BPB tasks)
        if requests and requests[0].request_type == RequestType.LOGLIKELIHOOD:
            outputs_list = provider.logprobs(requests)
        else:
            outputs_list = provider.generate(requests, sampling_params)

        for item, outputs in zip(batch, outputs_list, strict=True):
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
        # On batch failure, report error for all items
        for item in batch:
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


__all__ = [
    "check_workers_alive",
    "wait_for_workers_ready",
    "process_batch",
]

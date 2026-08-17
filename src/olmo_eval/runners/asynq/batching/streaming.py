"""Streaming strategy - send items directly to provider."""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.harness import Harness
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem

from .base import BatchingStrategy


class StreamingStrategy(BatchingStrategy):
    """Stream items directly to the provider with no explicit batching.

    Items are sent to the provider as soon as they arrive. Used for
    providers like LLM() that handle their own internal batching.
    """

    async def run(
        self,
        item_queue: mp.Queue[QueueItem | None],
        harness: Harness,
        result_queue: mp.Queue[ResultItem],
        max_concurrency: int | None,
        worker_logger: logging.Logger,
        total_instances: int,
        num_workers: int = 1,
    ) -> None:
        """Execute streaming to provider."""
        import math

        from olmo_eval.common.beaker_status import BeakerStatusReporter
        from olmo_eval.common.progress import ProgressLogger
        from olmo_eval.runners.asynq.processing import process_items

        concurrency = max_concurrency or 64
        semaphore = asyncio.Semaphore(concurrency)
        in_flight: set[asyncio.Task[None]] = set()

        worker_instances = math.ceil(total_instances / num_workers)
        progress = ProgressLogger(
            total=worker_instances,
            desc="Processed",
            logger=worker_logger,
            color="green",
        )
        reporter = BeakerStatusReporter()
        report_progress = reporter.progress_callback("Processed")

        async def process_single(item: QueueItem) -> None:
            async with semaphore:
                await process_items(
                    [item], harness, result_queue, 1, worker_logger, show_progress=False
                )
                progress.update(1)
                report_progress(progress.count, progress.total)

        def raise_completed_task_errors() -> None:
            """Remove completed tasks and propagate the first failure.

            Retrieve every completed task's outcome before raising so concurrent
            failures cannot turn into unhandled task exceptions.
            """
            done = {task for task in in_flight if task.done()}
            in_flight.difference_update(done)

            first_error: BaseException | None = None
            for task in done:
                if task.cancelled():
                    continue
                try:
                    task.result()
                except BaseException as error:
                    if first_error is None:
                        first_error = error

            if first_error is not None:
                raise first_error

        async def get_item() -> QueueItem | None:
            """Get next item from queue asynchronously."""
            loop = asyncio.get_event_loop()
            while True:
                try:
                    return await loop.run_in_executor(None, lambda: item_queue.get(timeout=0.1))
                except queue.Empty:
                    raise_completed_task_errors()
                    if not in_flight:
                        try:
                            return await loop.run_in_executor(
                                None, lambda: item_queue.get(timeout=1.0)
                            )
                        except queue.Empty:
                            return None
                    await asyncio.sleep(0.01)

        try:
            while True:
                raise_completed_task_errors()
                item = await get_item()
                raise_completed_task_errors()

                if item is None:
                    break

                in_flight.add(asyncio.create_task(process_single(item)))

            if in_flight:
                await asyncio.gather(*in_flight)
        finally:
            # Stop and drain all sibling requests when any streamed request is
            # terminal. Keeping the tasks tracked until their outcomes are
            # retrieved prevents fast failures from disappearing.
            remaining = tuple(in_flight)
            for task in remaining:
                if not task.done():
                    task.cancel()
            if remaining:
                await asyncio.gather(*remaining, return_exceptions=True)

            progress.close()
            report_progress(progress.count, progress.total, force=True)

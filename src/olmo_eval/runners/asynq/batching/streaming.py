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
    ) -> None:
        """Execute streaming to provider."""
        from olmo_eval.runners.asynq.processing import process_items

        concurrency = max_concurrency or 64
        semaphore = asyncio.Semaphore(concurrency)
        in_flight: set[asyncio.Task] = set()
        processed = 0

        async def process_single(item: QueueItem) -> None:
            nonlocal processed
            async with semaphore:
                await process_items([item], harness, result_queue, 1, worker_logger)
                processed += 1
                if processed % 100 == 0:
                    worker_logger.info(f"Processed {processed}/{total_instances} items")

        async def get_item() -> QueueItem | None:
            """Get next item from queue asynchronously."""
            loop = asyncio.get_event_loop()
            while True:
                try:
                    return await loop.run_in_executor(None, lambda: item_queue.get(timeout=0.1))
                except queue.Empty:
                    if not in_flight:
                        try:
                            return await loop.run_in_executor(
                                None, lambda: item_queue.get(timeout=1.0)
                            )
                        except queue.Empty:
                            return None
                    await asyncio.sleep(0.01)

        worker_logger.info("Starting streaming processing")

        while True:
            item = await get_item()

            if item is None:
                break

            task = asyncio.create_task(process_single(item))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        if in_flight:
            worker_logger.info(f"Waiting for {len(in_flight)} in-flight items")
            await asyncio.gather(*in_flight)

        worker_logger.info(f"Streaming complete: {processed} items")

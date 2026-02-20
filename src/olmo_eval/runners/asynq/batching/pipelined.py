"""Pipelined batching strategy - multiple batches in flight concurrently."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.harness import Harness
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem

from .base import BatchingStrategy


class PipelinedStrategy(BatchingStrategy):
    """Keep multiple batches in flight for maximum GPU utilization.

    While batch N is processing, batch N+1 is already queued. This ensures
    the GPU/provider always has work available, minimizing idle time between
    batches.
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
        """Execute pipelined batching with overlapping batches."""
        from olmo_eval.runners.asynq.processing import process_items

        total_batches = math.ceil(total_instances / self.config.chunk_size)
        batch_num = 0
        in_flight: set[asyncio.Task] = set()
        saw_shutdown = False

        while True:
            # Collect next batch
            batch, saw_shutdown = await self.collect_batch(item_queue)

            if not batch and saw_shutdown:
                # Empty batch with shutdown - wait for in-flight and exit
                break

            if batch:
                batch_num += 1

                # Stagger batch starts to avoid synchronized completion times.
                # Before starting batch N+1, let batch N make some progress.
                if in_flight:
                    await asyncio.sleep(1.0)

                worker_logger.info(
                    f"Starting batch {batch_num}/{total_batches} ({len(batch)} items)"
                )

                # Start processing without waiting
                task = asyncio.create_task(
                    process_items(batch, harness, result_queue, max_concurrency, worker_logger)
                )
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)

                # If at capacity, wait for any one to complete
                if len(in_flight) >= self.config.max_in_flight:
                    _done, _pending = await asyncio.wait(
                        in_flight, return_when=asyncio.FIRST_COMPLETED
                    )

            if saw_shutdown:
                break

        # Drain remaining in-flight batches
        if in_flight:
            worker_logger.info(f"Waiting for {len(in_flight)} in-flight batches")
            await asyncio.gather(*in_flight)

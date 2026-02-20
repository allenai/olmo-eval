"""Sequential batching strategy - one batch at a time."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.harness import Harness
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem

from .base import BatchingStrategy


class SequentialStrategy(BatchingStrategy):
    """Process one batch at a time, wait for completion before starting next.

    This is the safest strategy with predictable memory usage and ordering.
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
        """Execute sequential batching."""
        from olmo_eval.runners.asynq.processing import process_items

        total_batches = math.ceil(total_instances / self.config.chunk_size)
        batch_num = 0

        while True:
            # Collect batch
            batch, saw_shutdown = await self.collect_batch(item_queue)

            if not batch and saw_shutdown:
                # Empty batch with shutdown signal - we're done
                return

            if batch:
                batch_num += 1
                worker_logger.info(
                    f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)"
                )
                await process_items(batch, harness, result_queue, max_concurrency, worker_logger)

            if saw_shutdown:
                # Processed final batch
                return

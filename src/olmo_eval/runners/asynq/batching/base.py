"""Abstract base class for batching strategies."""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from olmo_eval.runners.asynq.share_clock import ShareClock

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.harness import Harness
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem

from .config import BatchConfig


class BatchingStrategy(ABC):
    """Abstract base class for batching strategies.

    Each inference worker builds its own strategy, so ``clock`` is the worker's: every
    batch and rollout the worker runs shares it, and overlapping work splits the
    worker's time.

    ``per_request`` asks for each request's result as soon as it finishes, which a
    run that records progress needs so that a preemption loses little work. Only
    strategies that wait for whole batches act on it.
    """

    def __init__(
        self,
        config: BatchConfig,
        max_batch_sequences: int | None = None,
        per_request: bool = False,
    ):
        self.config = config
        self.clock = ShareClock()
        self.max_batch_sequences = max_batch_sequences
        self.per_request = per_request
        self._pending_item = None

    @abstractmethod
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
        """Execute the batching strategy.

        Args:
            item_queue: Queue of QueueItems (None signals shutdown).
            harness: Harness instance for execution.
            result_queue: Queue to put ResultItems.
            max_concurrency: Maximum concurrent requests.
            worker_logger: Logger with worker identification.
            total_instances: Total number of instances across all workers.
            num_workers: Number of parallel workers sharing the work.
        """
        ...

    async def collect_batch(
        self,
        item_queue: mp.Queue[QueueItem | None],
    ) -> tuple[list[QueueItem], bool]:
        """Collect items for a batch.

        Args:
            item_queue: Queue to collect from.

        Returns:
            Tuple of (items, saw_shutdown). If saw_shutdown is True,
            the returned items are the final batch before shutdown.
        """

        items: list[QueueItem] = []
        deadline = time.time() + self.config.chunk_timeout

        def collect_sync() -> tuple[list[QueueItem], bool]:
            sequences = 0
            while len(items) < self.config.resolved_chunk_size:
                remaining = max(0.01, deadline - time.time())
                try:
                    item = self._pending_item
                    if item is None:
                        item = item_queue.get(timeout=remaining)
                    self._pending_item = None
                    if item is None:
                        return items, True
                    count = item.sampling_params.num_samples if item.sampling_params else 1
                    if (
                        items
                        and self.max_batch_sequences
                        and sequences + count > self.max_batch_sequences
                    ):
                        self._pending_item = item
                        break
                    items.append(item)
                    sequences += count
                    if self.max_batch_sequences and sequences >= self.max_batch_sequences:
                        break
                except queue.Empty:
                    break
            return items, False

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, collect_sync)

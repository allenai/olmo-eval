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
        from olmo_eval.runners.asynq.types import QueueItem as QueueItemType

        total_batches = math.ceil(total_instances / self.config.chunk_size)
        batch_num = 0
        in_flight: set[asyncio.Task] = set()

        # Prefetch: always have the next batch ready
        pending_batch: list[QueueItemType] | None = None
        pending_shutdown = False
        collect_task: asyncio.Task | None = asyncio.create_task(self.collect_batch(item_queue))

        # For staggering: track when initial ramp is complete
        # During initial ramp, we stagger batch starts to avoid synchronized completions
        initial_ramp_complete = False

        while True:
            # Wait for either: batch collection completes OR an in-flight task completes
            wait_tasks: set[asyncio.Task] = set()
            if collect_task is not None:
                wait_tasks.add(collect_task)
            wait_tasks.update(in_flight)

            if not wait_tasks:
                break

            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

            # Check if batch collection completed
            if collect_task in done:
                batch, saw_shutdown = collect_task.result()
                collect_task = None

                if batch:
                    pending_batch = batch
                if saw_shutdown:
                    pending_shutdown = True

            # Check if any in-flight batch completed and remove them
            completed_tasks = done & in_flight
            completed_any = bool(completed_tasks)
            in_flight -= completed_tasks

            # After initial ramp, completions trigger new batch starts
            # This maintains the stagger established during ramp
            if initial_ramp_complete:
                can_start = (
                    pending_batch is not None
                    and len(in_flight) < self.config.max_in_flight
                    and completed_any
                )
            else:
                # During initial ramp: fill slots but with stagger delay
                can_start = pending_batch is not None and len(in_flight) < self.config.max_in_flight

            if can_start:
                batch_num += 1

                # During initial ramp, stagger batch starts to avoid synchronized completions
                # After the first batch, wait for earlier batches to make progress
                if not initial_ramp_complete and len(in_flight) > 0:
                    # Stagger by waiting - this spaces out the initial batches
                    # so they complete at different times
                    stagger_seconds = self.config.stagger_delay
                    worker_logger.info(f"Staggering batch start by {stagger_seconds}s")
                    await asyncio.sleep(stagger_seconds)

                worker_logger.info(
                    f"Starting batch {batch_num}/{total_batches} ({len(pending_batch)} items)"
                )

                task = asyncio.create_task(
                    process_items(
                        pending_batch, harness, result_queue, max_concurrency, worker_logger
                    )
                )
                in_flight.add(task)
                pending_batch = None

                # Check if initial ramp is complete (reached max capacity)
                if not initial_ramp_complete and len(in_flight) >= self.config.max_in_flight:
                    initial_ramp_complete = True

                # Start collecting next batch immediately (if not shutdown)
                if not pending_shutdown and collect_task is None:
                    collect_task = asyncio.create_task(self.collect_batch(item_queue))

            # Exit conditions
            if pending_shutdown and not in_flight and pending_batch is None:
                break

        # Drain remaining in-flight batches
        if in_flight:
            worker_logger.info(f"Waiting for {len(in_flight)} in-flight batches")
            await asyncio.gather(*in_flight)

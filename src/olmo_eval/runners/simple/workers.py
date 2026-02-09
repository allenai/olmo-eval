"""Worker processes for async evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.core.logging import get_logger
from olmo_eval.runners.simple.queue import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.core.harness import Harness

logger = get_logger(__name__)


async def run_worker_loop(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    max_concurrency: int,
) -> None:
    """Process items using async workers with bounded concurrency.

    Uses the asyncio.Queue + TaskGroup pattern: populate queue first,
    then run workers that consume from it.

    Args:
        items: List of queue items to process.
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        max_concurrency: Maximum number of concurrent workers.
    """
    from tqdm import tqdm

    from olmo_eval.runners.simple.helpers import process_request

    work_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
    pbar = tqdm(total=len(items), desc="Processing", unit="inst")

    # Populate queue with all items
    for item in items:
        await work_queue.put(item)

    async def worker() -> None:
        """Process items from the queue until empty."""
        while True:
            item = await work_queue.get()
            try:
                await process_request(item, harness, result_queue)
                pbar.update(1)
            finally:
                work_queue.task_done()

    try:
        async with asyncio.TaskGroup() as tg:
            workers = [tg.create_task(worker()) for _ in range(max_concurrency)]
            await work_queue.join()
            for w in workers:
                w.cancel()
    finally:
        pbar.close()


def instance_worker_process(
    worker_id: str,
    gpu_ids: list[int],
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    harness_config_dict: dict[str, Any],
    init_times: dict[str, float] | None = None,
) -> None:
    """Worker that processes items with bounded async concurrency.

    Collects items from the mp.Queue, then processes them using an asyncio
    worker pool with bounded concurrency.

    Args:
        worker_id: Unique worker identifier (e.g., "OLMo-2-7B-w0")
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        instance_queue: Queue of QueueItems (None = poison pill)
        result_queue: Queue to put ResultItems
        harness_config_dict: Serialized HarnessConfig dict with all provider configuration
        init_times: Shared dict for tracking worker initialization times
    """
    import sys

    from olmo_eval.core.logging import configure_logging, configure_worker_logging

    configure_logging()

    worker_logger = configure_worker_logging(worker_id)
    worker_logger.info(f"Starting on GPUs {gpu_ids}")

    from olmo_eval.core.harness import Harness, HarnessConfig

    harness_config = HarnessConfig.from_dict(harness_config_dict)
    provider_config = harness_config.provider
    model_name = provider_config.model

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        provider_kind = str(provider_config.kind)
        tokenizer = provider_config.tokenizer
        max_model_len = provider_config.max_model_len
        max_concurrency = provider_config.max_concurrency or harness_config.max_concurrency or 8
        provider_kwargs = dict(provider_config.kwargs) if provider_config.kwargs else {}

        attention_backend = provider_kwargs.get("attention_backend")
        load_format = provider_kwargs.get("load_format")
        extra_loader_config = provider_kwargs.get("model_loader_extra_config")

        worker_logger.info(f"Initializing provider: {provider_kind}")
        worker_logger.info(f"  Model: {model_name}")
        if tokenizer:
            worker_logger.info(f"  Tokenizer: {tokenizer}")
        if gpu_ids:
            worker_logger.info(f"  GPUs: {gpu_ids}")

        init_start = time.time()

        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

        has_tools = harness_config.has_tools
        enable_auto_tool_choice = has_tools and provider_kind == "vllm_server"

        worker_logger.info(f"  Has tools: {has_tools}")
        if has_tools:
            worker_logger.info(f"  Tools: {list(harness_config.tool_names)}")
        worker_logger.info(f"  Enable auto tool choice: {enable_auto_tool_choice}")

        harness_config = harness_config.with_provider_overrides(
            tensor_parallel_size=len(gpu_ids) if gpu_ids else None,
            max_model_len=max_model_len,
            max_concurrency=max_concurrency,
            tokenizer=tokenizer,
            load_format=load_format,
            model_loader_extra_config=extra_loader_config,
            enable_auto_tool_choice=enable_auto_tool_choice or None,
        )

        harness = Harness(harness_config)

        init_time = time.time() - init_start
        worker_logger.info(f"Provider ready ({init_time:.1f}s)")

        if init_times is not None:
            init_times[worker_id] = init_time

        try:
            # Collect all items from mp.Queue
            items: list[QueueItem] = []
            while True:
                item = instance_queue.get()
                if item is None:  # Poison pill
                    break
                items.append(item)

            worker_logger.info(f"Processing {len(items)} instances...")

            if items:
                asyncio.run(run_worker_loop(items, harness, result_queue, max_concurrency))

            worker_logger.info("Processing complete")
        finally:
            close_fn = getattr(harness.provider, "close", None)
            if callable(close_fn):
                close_fn()

    except Exception as e:
        worker_logger.error(f"Worker process failed: {e}")
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id="__WORKER_FATAL__",
                instance_idx=-1,
                instance=None,  # type: ignore[arg-type]
                request=None,  # type: ignore[arg-type]
                outputs=[],
                error=f"Worker process crashed: {e}",
            )
        )
        sys.exit(1)


__all__ = [
    "instance_worker_process",
]

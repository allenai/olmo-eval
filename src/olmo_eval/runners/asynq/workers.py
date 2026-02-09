"""Worker processes for evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.core.logging import get_logger
from olmo_eval.runners.asynq.queue import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.core.harness import Harness

logger = get_logger(__name__)


async def process_items(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    max_concurrency: int,
) -> None:
    """Process queue items, batching where possible.

    COMPLETION and LOGLIKELIHOOD requests are grouped by sampling_params and
    processed in batches. CHAT requests are processed individually with async
    concurrency.

    Args:
        items: Queue items to process.
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        max_concurrency: Maximum concurrent CHAT requests.
    """
    from tqdm import tqdm

    from olmo_eval.core.types import RequestType, SamplingParams
    from olmo_eval.runners.asynq.helpers import process_batch, process_chat_request

    chat_items: list[QueueItem] = []
    batchable_items: list[QueueItem] = []

    for item in items:
        if item.request.request_type == RequestType.CHAT:
            chat_items.append(item)
        else:
            batchable_items.append(item)

    pbar = tqdm(total=len(items), desc="Processing", unit="inst")

    if batchable_items:
        batches: dict[tuple[RequestType, SamplingParams | None], list[QueueItem]] = {}
        for item in batchable_items:
            key = (item.request.request_type, item.sampling_params)
            if key not in batches:
                batches[key] = []
            batches[key].append(item)

        loop = asyncio.get_event_loop()
        for batch in batches.values():
            await loop.run_in_executor(None, process_batch, batch, harness, result_queue)
            pbar.update(len(batch))

    if chat_items:
        work_queue: asyncio.Queue[QueueItem] = asyncio.Queue()
        for item in chat_items:
            await work_queue.put(item)

        async def worker() -> None:
            while True:
                item = await work_queue.get()
                try:
                    await process_chat_request(item, harness, result_queue)
                    pbar.update(1)
                finally:
                    work_queue.task_done()

        try:
            async with asyncio.TaskGroup() as tg:
                workers = [tg.create_task(worker()) for _ in range(max_concurrency)]
                await work_queue.join()
                for w in workers:
                    w.cancel()
        except* asyncio.CancelledError:
            pass

    pbar.close()


def worker_process(
    worker_id: str,
    gpu_ids: list[int],
    item_queue: mp.Queue,
    result_queue: mp.Queue,
    harness_config_dict: dict[str, Any],
    init_times: dict[str, float] | None = None,
) -> None:
    """Worker process that initializes a harness and processes items.

    Collects all items from the queue, then processes them with batching
    for COMPLETION/LOGLIKELIHOOD and async concurrency for CHAT requests.

    Args:
        worker_id: Unique worker identifier.
        gpu_ids: GPU IDs to use (sets CUDA_VISIBLE_DEVICES).
        item_queue: Queue of QueueItems (None signals shutdown).
        result_queue: Queue to put ResultItems.
        harness_config_dict: Serialized HarnessConfig.
        init_times: Optional shared dict for tracking initialization times.
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
            items: list[QueueItem] = []
            while True:
                item = item_queue.get()
                if item is None:
                    break
                items.append(item)

            worker_logger.info(f"Processing {len(items)} instances...")

            if items:
                asyncio.run(process_items(items, harness, result_queue, max_concurrency))

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
    "worker_process",
]

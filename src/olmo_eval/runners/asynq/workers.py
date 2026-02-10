"""Worker processes for evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from typing import TYPE_CHECKING, Any

from olmo_eval.core.logging import get_logger
from olmo_eval.runners.asynq.queue import WORKER_FATAL_TASK_ID, QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.core.harness import Harness

logger = get_logger(__name__)


async def process_items(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    max_concurrency: int | None = None,
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
    from olmo_eval.core.types import RequestType, SamplingParams
    from olmo_eval.runners.asynq.helpers import process_batch, process_chat_request

    chat_items: list[QueueItem] = []
    batchable_items: list[QueueItem] = []

    for item in items:
        if item.request.request_type == RequestType.CHAT:
            chat_items.append(item)
        else:
            batchable_items.append(item)

    if batchable_items:
        batches: dict[tuple[RequestType, SamplingParams | None], list[QueueItem]] = {}
        for item in batchable_items:
            key = (item.request.request_type, item.sampling_params)
            if key not in batches:
                batches[key] = []
            batches[key].append(item)

        for batch in batches.values():
            await process_batch(batch, harness, result_queue)

    if chat_items:
        from tqdm import tqdm
        from tqdm.contrib.logging import logging_redirect_tqdm

        semaphore = asyncio.Semaphore(max_concurrency or len(chat_items))

        async def process(item: QueueItem, pbar: tqdm) -> None:
            async with semaphore:
                await process_chat_request(item, harness, result_queue)
                pbar.update(1)

        with (
            logging_redirect_tqdm(),
            tqdm(total=len(chat_items), desc="Processing instances", unit="inst") as pbar,
        ):
            await asyncio.gather(*[process(item, pbar) for item in chat_items])


def inference_worker(
    worker_id: str,
    gpu_ids: list[int],
    item_queue: mp.Queue,
    result_queue: mp.Queue,
    harness_config_dict: dict[str, Any],
    init_times: dict[str, float] | None = None,
    output_dir: str | None = None,
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
        output_dir: Output directory for persisting logs (e.g., vLLM server logs).
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
        max_concurrency = provider_config.max_concurrency or harness_config.max_concurrency
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

        # Set log_dir for vllm_server provider to persist server logs
        log_dir = output_dir if provider_kind == "vllm_server" and output_dir else None

        harness_config = harness_config.with_provider_overrides(
            tensor_parallel_size=len(gpu_ids) if gpu_ids else None,
            max_model_len=max_model_len,
            max_concurrency=max_concurrency,
            tokenizer=tokenizer,
            load_format=load_format,
            model_loader_extra_config=extra_loader_config,
            enable_auto_tool_choice=enable_auto_tool_choice or None,
            log_dir=log_dir,
        )

        harness = Harness(harness_config)

        # Force provider creation to catch import errors early
        _ = harness.provider

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
                task_id=WORKER_FATAL_TASK_ID,
                instance_idx=-1,
                instance=None,  # type: ignore[arg-type]
                request=None,  # type: ignore[arg-type]
                outputs=[],
                error=f"Worker process crashed: {e}",
            )
        )
        sys.exit(1)


def scoring_worker(
    scoring_queue: mp.Queue,
    scored_queue: mp.Queue,
    total_instances: int,
) -> None:
    """Worker process that scores individual responses.

    Reads ScoringItems from scoring_queue, scores each response, and puts
    ScoredResponses on scored_queue.

    Args:
        scoring_queue: Queue of ScoringItems (None signals shutdown).
        scored_queue: Queue to put ScoredResponses.
        total_instances: Total number of instances to score (for progress bar).
    """
    from tqdm import tqdm

    from olmo_eval.core.logging import configure_logging

    configure_logging()

    from olmo_eval.runners.asynq.queue import ScoredResponse, ScoringItem

    try:
        with tqdm(total=total_instances, desc="Scoring instances", unit="inst") as pbar:
            while True:
                item: ScoringItem | None = scoring_queue.get()
                if item is None:
                    break

                try:
                    # Score single response
                    scored_list = item.task.score_responses([item.response])
                    scored = scored_list[0] if scored_list else item.response
                    scored_queue.put(
                        ScoredResponse(
                            spec=item.spec,
                            instance_idx=item.instance_idx,
                            scored=scored,
                        )
                    )
                except Exception as e:
                    # On error, return the original response (unscored)
                    logger.warning(f"Failed to score {item.spec}[{item.instance_idx}]: {e}")
                    scored_queue.put(
                        ScoredResponse(
                            spec=item.spec,
                            instance_idx=item.instance_idx,
                            scored=item.response,
                        )
                    )
                finally:
                    pbar.update(1)
    except Exception as e:
        logger.error(f"Scoring worker crashed: {e}")


__all__ = [
    "scoring_worker",
    "inference_worker",
]

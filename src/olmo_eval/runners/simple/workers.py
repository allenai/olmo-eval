"""Worker processes for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.inference import ProviderType
from olmo_eval.runners.simple.queue import QueueItem, ResultItem

logger = get_logger(__name__)


def instance_worker_process(
    worker_id: str,
    gpu_ids: list[int],
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    provider_type_str: str,
    attention_backend: str | None = None,
    tokenizer: str | None = None,
    max_model_len: int | None = None,
    load_format: str | None = None,
    extra_loader_config: dict[str, Any] | None = None,
    max_concurrency: int | None = None,
    init_times: dict[str, float] | None = None,
    harness_config_dict: dict[str, Any] | None = None,
) -> None:
    """Worker that collects all items and processes them at once.

    Collects all items from the queue, then processes them in a single
    provider call for maximum throughput. vLLM handles internal batching.

    Args:
        worker_id: Unique worker identifier (e.g., "OLMo-2-7B-w0")
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        instance_queue: Queue of QueueItems (None = poison pill)
        result_queue: Queue to put ResultItems
        model_name: Model name for provider
        provider_type_str: Provider type string
        attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN")
        tokenizer: Tokenizer path/identifier, defaults to model if None
        max_model_len: Maximum model context length (overrides model's default)
        load_format: vLLM model loading format (e.g., "runai_streamer")
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true})
        max_concurrency: Maximum concurrent API requests (for litellm and other API providers)
        init_times: Shared dict for tracking worker initialization times
        harness_config_dict: Serialized HarnessConfig dict for tool/prompt configuration
    """
    import sys

    from olmo_eval.core.logging import configure_logging, configure_worker_logging

    # Set up root logger so third-party loggers (e.g. litellm) have a
    # handler to propagate to.  Must happen before provider init.
    configure_logging()

    worker_logger = configure_worker_logging(worker_id)
    worker_logger.info(f"Starting on GPUs {gpu_ids}")

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        provider_type = ProviderType(provider_type_str)
        worker_logger.info(f"Initializing provider: {provider_type.value}")
        worker_logger.info(f"  Model: {model_name}")
        if tokenizer:
            worker_logger.info(f"  Tokenizer: {tokenizer}")
        if gpu_ids:
            worker_logger.info(f"  GPUs: {gpu_ids}")

        init_start = time.time()

        if not harness_config_dict:
            raise ValueError("harness_config_dict is required")

        # Set attention backend via env var (vLLM reads this)
        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

        # Create harness config with worker-specific provider overrides
        from olmo_eval.core.harness import Harness, HarnessConfig

        harness_config = HarnessConfig.from_dict(harness_config_dict)

        # Determine if auto tool choice should be enabled
        enable_auto_tool_choice = (
            harness_config.has_tools and provider_type == ProviderType.VLLM_SERVER
        )
        if enable_auto_tool_choice:
            worker_logger.info(f"  Tools configured: {list(harness_config.tool_names)}")

        # Apply worker-specific provider overrides
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
            # Collect all items from queue
            items: list[QueueItem] = []
            while True:
                item = instance_queue.get()
                if item is None:  # Poison pill
                    break
                items.append(item)

            worker_logger.info(f"Processing {len(items)} instances...")

            # Process items grouped by request type
            if items:
                from olmo_eval.core.types import RequestType
                from olmo_eval.runners.simple.helpers import (
                    process_chat_requests,
                    process_generate_requests,
                )

                # Separate by request type - worker handles this grouping
                chat_items = [i for i in items if i.request.request_type == RequestType.CHAT]
                batch_items = [i for i in items if i.request.request_type != RequestType.CHAT]

                # Process batch items (COMPLETION, LOGLIKELIHOOD) - sync
                if batch_items:
                    worker_logger.info(f"Processing {len(batch_items)} batch requests...")
                    process_generate_requests(batch_items, harness, result_queue)

                # Process chat items - async execution managed here
                if chat_items:
                    import asyncio

                    worker_logger.info(f"Processing {len(chat_items)} chat requests...")
                    asyncio.run(process_chat_requests(chat_items, harness, result_queue))

            worker_logger.info("Processing complete")
        finally:
            # Ensure provider cleanup (stops vLLM server if managed)
            close_fn = getattr(harness.provider, "close", None)
            if callable(close_fn):
                close_fn()

    except Exception as e:
        worker_logger.error(f"Worker process failed: {e}")
        # Put a fatal error marker in the result queue so main process knows we died
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

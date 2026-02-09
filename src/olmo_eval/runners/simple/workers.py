"""Worker processes for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.runners.simple.queue import QueueItem, ResultItem

logger = get_logger(__name__)


def instance_worker_process(
    worker_id: str,
    gpu_ids: list[int],
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    harness_config_dict: dict[str, Any],
    init_times: dict[str, float] | None = None,
) -> None:
    """Worker that collects all items and processes them at once.

    Collects all items from the queue, then processes them in a single
    provider call for maximum throughput. vLLM handles internal batching.

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

    # Set up root logger so third-party loggers (e.g. litellm) have a
    # handler to propagate to.  Must happen before provider init.
    configure_logging()

    worker_logger = configure_worker_logging(worker_id)
    worker_logger.info(f"Starting on GPUs {gpu_ids}")

    # Parse harness config early so we can use it for error reporting
    from olmo_eval.core.harness import Harness, HarnessConfig

    harness_config = HarnessConfig.from_dict(harness_config_dict)
    provider_config = harness_config.provider
    model_name = provider_config.model

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        # Extract provider configuration from harness_config
        provider_kind = str(provider_config.kind)
        tokenizer = provider_config.tokenizer
        max_model_len = provider_config.max_model_len
        max_concurrency = provider_config.max_concurrency or harness_config.max_concurrency
        provider_kwargs = dict(provider_config.kwargs) if provider_config.kwargs else {}

        # Extract vLLM-specific options from kwargs
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

        # Set attention backend via env var (vLLM reads this)
        if attention_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend

        # Determine if auto tool choice should be enabled
        has_tools = harness_config.has_tools
        enable_auto_tool_choice = has_tools and provider_kind == "vllm_server"

        worker_logger.info(f"  Has tools: {has_tools}")
        if has_tools:
            worker_logger.info(f"  Tools: {list(harness_config.tool_names)}")
        worker_logger.info(f"  Enable auto tool choice: {enable_auto_tool_choice}")

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
                generate_items = [i for i in items if i.request.request_type != RequestType.CHAT]

                # Process (COMPLETION, LOGLIKELIHOOD)
                if generate_items:
                    worker_logger.info(f"Processing {len(generate_items)} batch requests...")
                    process_generate_requests(generate_items, harness, result_queue)

                # Process chat items
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

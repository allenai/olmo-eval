"""Worker processes for async evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.inference import ProviderType, create_provider
from olmo_eval.runners.simple.helpers import process_batch
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
        model_name: Model name for provider
        provider_type_str: Provider type string
        attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN")
        tokenizer: Tokenizer path/identifier, defaults to model if None
        max_model_len: Maximum model context length (overrides model's default)
        load_format: vLLM model loading format (e.g., "runai_streamer")
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true})
    """
    import sys

    from olmo_eval.core.logging import configure_worker_logging

    worker_logger = configure_worker_logging(worker_id)
    worker_logger.info(f"Starting on GPUs {gpu_ids}")

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        worker_logger.info("Initializing vLLM engine...")
        init_start = time.time()

        provider_type = ProviderType(provider_type_str)
        # Pass tensor_parallel_size for vLLM to use all assigned GPUs
        engine_kwargs: dict[str, Any] = {"tensor_parallel_size": len(gpu_ids)} if gpu_ids else {}
        if attention_backend:
            engine_kwargs["attention_backend"] = attention_backend
        if max_model_len:
            engine_kwargs["max_model_len"] = max_model_len
        if load_format:
            engine_kwargs["load_format"] = load_format
        if extra_loader_config:
            engine_kwargs["model_loader_extra_config"] = extra_loader_config
        # Pass worker_id for scoped logging in vLLM
        provider = create_provider(
            provider_type, model_name, tokenizer=tokenizer, worker_id=worker_id, **engine_kwargs
        )

        init_time = time.time() - init_start
        worker_logger.info(f"Engine ready ({init_time:.1f}s)")
        if init_times is not None:
            init_times[worker_id] = init_time

        # Collect all items from queue
        items: list[QueueItem] = []
        while True:
            item = instance_queue.get()
            if item is None:  # Poison pill
                break
            items.append(item)

        worker_logger.info(f"Processing {len(items)} instances...")

        # Process all items at once - vLLM handles internal batching
        if items:
            process_batch(items, provider, result_queue)

        worker_logger.info("Processing complete")
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


def streaming_worker_process(
    worker_id: str,
    gpu_ids: list[int],
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    attention_backend: str | None = None,
    tokenizer: str | None = None,
    max_model_len: int | None = None,
    load_format: str | None = None,
    extra_loader_config: dict[str, Any] | None = None,
    init_times: dict[str, float] | None = None,
) -> None:
    """Worker using async streaming for true continuous batching.

    Unlike the batch worker, this worker uses vLLM's AsyncLLMEngine to:
    1. Add requests continuously as they arrive
    2. Stream results back as they complete
    3. Enable true continuous batching for optimal throughput

    Args:
        worker_id: Unique worker identifier (e.g., "OLMo-2-7B-w0")
        gpu_ids: List of GPU IDs to use (for CUDA_VISIBLE_DEVICES)
        instance_queue: Queue of QueueItems (None = poison pill)
        result_queue: Queue to put ResultItems
        model_name: Model name for provider
        attention_backend: Attention backend to use (e.g., "FLASHINFER", "FLASH_ATTN")
        tokenizer: Tokenizer path/identifier, defaults to model if None
        max_model_len: Maximum model context length (overrides model's default)
        load_format: vLLM model loading format (e.g., "runai_streamer")
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true})
    """
    import sys

    from olmo_eval.core.logging import configure_worker_logging

    worker_logger = configure_worker_logging(worker_id)
    worker_logger.info(f"Starting on GPUs {gpu_ids}")

    try:
        if gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

        # Run the async worker
        num_gpus = len(gpu_ids) if gpu_ids else 1
        asyncio.run(
            _streaming_worker_async(
                worker_id,
                instance_queue,
                result_queue,
                model_name,
                num_gpus,
                attention_backend,
                tokenizer,
                max_model_len,
                load_format,
                extra_loader_config,
                init_times,
            )
        )
        worker_logger.info("Processing complete")
    except Exception as e:
        worker_logger.error(f"Streaming worker process failed: {e}")
        # Put a fatal error marker in the result queue so main process knows we died
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id="__WORKER_FATAL__",
                instance_idx=-1,
                instance=None,  # type: ignore[arg-type]
                request=None,  # type: ignore[arg-type]
                outputs=[],
                error=f"Streaming worker process crashed: {e}",
            )
        )
        sys.exit(1)


async def _streaming_worker_async(
    worker_id: str,
    instance_queue: mp.Queue,
    result_queue: mp.Queue,
    model_name: str,
    num_gpus: int = 1,
    attention_backend: str | None = None,
    tokenizer: str | None = None,
    max_model_len: int | None = None,
    load_format: str | None = None,
    extra_loader_config: dict[str, Any] | None = None,
    init_times: dict[str, float] | None = None,
) -> None:
    """Async implementation of streaming worker.

    Uses AsyncVLLMProvider for true continuous batching with streaming results.
    """
    import logging

    from olmo_eval.inference.vllm import AsyncVLLMProvider

    # Get the worker logger (already configured by streaming_worker_process)
    worker_logger = logging.getLogger(f"olmo_eval.worker.{worker_id}")
    worker_logger.info("Initializing vLLM async engine...")
    init_start = time.time()

    engine_kwargs: dict[str, Any] = {"tensor_parallel_size": num_gpus}
    if max_model_len:
        engine_kwargs["max_model_len"] = max_model_len
    if load_format:
        engine_kwargs["load_format"] = load_format
    if extra_loader_config:
        engine_kwargs["model_loader_extra_config"] = extra_loader_config

    # Pass worker_id for scoped logging in vLLM
    provider = AsyncVLLMProvider(
        model_name,
        tokenizer=tokenizer,
        attention_backend=attention_backend,
        worker_id=worker_id,
        **engine_kwargs,
    )

    init_time = time.time() - init_start
    worker_logger.info(f"Async engine ready ({init_time:.1f}s)")
    if init_times is not None:
        init_times[worker_id] = init_time

    # Track request_id -> QueueItem mapping
    pending_requests: dict[str, QueueItem] = {}
    request_counter = 0
    all_items_received = False
    items_added = 0
    results_received = 0

    async def add_requests() -> None:
        """Coroutine to continuously add requests from the queue."""
        nonlocal all_items_received, request_counter, items_added

        loop = asyncio.get_event_loop()
        while True:
            # Non-blocking get from queue
            item = await loop.run_in_executor(None, instance_queue.get)

            if item is None:  # Poison pill
                all_items_received = True
                break

            # Generate request ID and track it
            request_counter += 1
            request_id = f"req-{request_counter}"
            pending_requests[request_id] = item
            items_added += 1

            # Add to engine (non-blocking)
            try:
                await provider.add_request(request_id, item.request)
            except Exception as e:
                # Report error immediately
                result_queue.put(
                    ResultItem(
                        model_name=item.model_name,
                        task_id=item.task_id,
                        instance_idx=item.instance_idx,
                        instance=item.instance,
                        request=item.request,
                        outputs=[],
                        error=str(e),
                        attempt=item.attempt,
                    )
                )
                del pending_requests[request_id]

    async def collect_results() -> None:
        """Coroutine to collect results as they stream back."""
        nonlocal results_received

        while pending_requests or not all_items_received:
            # Wait a bit if no pending requests yet
            if not pending_requests:
                await asyncio.sleep(0.01)
                continue

            try:
                async for request_id, outputs in provider.stream_results():
                    if request_id not in pending_requests:
                        continue

                    item = pending_requests.pop(request_id)
                    results_received += 1

                    result_queue.put(
                        ResultItem(
                            model_name=item.model_name,
                            task_id=item.task_id,
                            instance_idx=item.instance_idx,
                            instance=item.instance,
                            request=item.request,
                            outputs=outputs,
                            error=None,
                            attempt=item.attempt,
                        )
                    )

                    # Check if we're done
                    if all_items_received and not pending_requests:
                        return

            except Exception as e:
                # Handle engine errors - fail all pending requests
                logger.error(f"Streaming engine error: {e}")
                for _req_id, item in list(pending_requests.items()):
                    result_queue.put(
                        ResultItem(
                            model_name=item.model_name,
                            task_id=item.task_id,
                            instance_idx=item.instance_idx,
                            instance=item.instance,
                            request=item.request,
                            outputs=[],
                            error=str(e),
                            attempt=item.attempt,
                        )
                    )
                pending_requests.clear()
                return

    # Run both coroutines concurrently
    await asyncio.gather(add_requests(), collect_results())

    # Shutdown engine
    await provider.shutdown()


__all__ = [
    "instance_worker_process",
    "streaming_worker_process",
]

"""Worker processes for async evaluation runners."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from olmo_eval.core.logging import get_logger
from olmo_eval.inference import ProviderType, create_provider
from olmo_eval.runners.simple.helpers import process_batch
from olmo_eval.runners.simple.queue import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider

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

        # Build engine kwargs
        engine_kwargs: dict[str, Any] = {"tensor_parallel_size": len(gpu_ids)} if gpu_ids else {}
        if attention_backend:
            engine_kwargs["attention_backend"] = attention_backend
        if max_model_len:
            engine_kwargs["max_model_len"] = max_model_len
        if load_format:
            engine_kwargs["load_format"] = load_format
        if extra_loader_config:
            engine_kwargs["model_loader_extra_config"] = extra_loader_config
        if max_concurrency:
            engine_kwargs["max_concurrency"] = max_concurrency

        # For vllm_server, we need to start a vLLM server process
        if provider_type == ProviderType.VLLM_SERVER:
            server_context, provider = _create_vllm_server_provider(
                model_name=model_name,
                tokenizer=tokenizer,
                worker_logger=worker_logger,
                **engine_kwargs,
            )
        else:
            server_context = nullcontext()
            provider = create_provider(
                provider_type,
                model_name,
                tokenizer=tokenizer,
                worker_id=worker_id,
                **engine_kwargs,
            )

        # Use context manager to keep server alive for vllm_server
        with server_context:
            # Create Harness if config provided
            harness = None
            if harness_config_dict:
                from olmo_eval.core.harness import Harness, HarnessConfig

                harness_config = HarnessConfig.from_dict(harness_config_dict)
                harness = Harness(harness_config, provider=provider)
                worker_logger.info(f"Harness created with config: {harness_config.name}")

            init_time = time.time() - init_start
            worker_logger.info(f"Provider ready ({init_time:.1f}s)")
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
                process_batch(items, provider, result_queue, harness=harness)

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


def _create_vllm_server_provider(
    model_name: str,
    tokenizer: str | None = None,
    worker_logger: Any = None,
    **engine_kwargs: Any,
) -> tuple[Any, InferenceProvider]:
    """Create a vLLM server process and provider.

    Starts a vLLM server subprocess and returns a context manager that keeps
    it alive, along with the provider that connects to it.

    Args:
        model_name: Model name/path to serve.
        tokenizer: Optional tokenizer path.
        worker_logger: Logger for status messages.
        **engine_kwargs: Additional vLLM engine arguments.

    Returns:
        Tuple of (server_context, provider).
    """
    from olmo_eval.inference.vllm_server import VLLMServerProcess
    from olmo_eval.inference.vllm_server_provider import VLLMServerProvider

    # Build server kwargs from engine kwargs
    server_kwargs: dict[str, Any] = {}
    if engine_kwargs.get("tensor_parallel_size"):
        server_kwargs["tensor_parallel_size"] = engine_kwargs["tensor_parallel_size"]
    if engine_kwargs.get("max_model_len"):
        server_kwargs["max_model_len"] = engine_kwargs["max_model_len"]
    if engine_kwargs.get("attention_backend"):
        # vLLM server uses VLLM_ATTENTION_BACKEND env var
        os.environ["VLLM_ATTENTION_BACKEND"] = engine_kwargs["attention_backend"]

    if worker_logger:
        worker_logger.info("Starting vLLM server subprocess...")
        worker_logger.info(f"  Model: {model_name}")
        if tokenizer:
            worker_logger.info(f"  Tokenizer: {tokenizer}")
        if server_kwargs:
            worker_logger.info(f"  Server kwargs: {server_kwargs}")

    # Create server process
    server = VLLMServerProcess(
        model_name=model_name,
        tokenizer=tokenizer,
        **server_kwargs,
    )

    # Start the server (this blocks until ready)
    base_url = server.start()

    if worker_logger:
        worker_logger.info(f"vLLM server started at {base_url}")

    # Create provider that connects to the server
    provider = VLLMServerProvider(
        model_name=model_name,
        base_url=server.base_url,
    )

    return server, provider


__all__ = [
    "instance_worker_process",
]

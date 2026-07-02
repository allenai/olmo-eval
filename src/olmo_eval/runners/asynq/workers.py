"""Worker processes for evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import sys
import time
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.asynq.types import (
    WORKER_FATAL,
    ResultItem,
)

logger = get_logger(__name__)


def _configure_hf_modules_cache_for_worker(
    worker_id: str,
    output_dir: str | None,
    trust_remote_code: bool,
    worker_logger: Any,
) -> None:
    """Avoid concurrent writes to Transformers' dynamic remote-code cache.

    Local checkpoints with ``trust_remote_code=True`` are copied into
    Hugging Face's modules cache before import. When many eval workers cold-start
    the same checkpoint at once, they can observe a partially populated package.
    Give each worker its own tiny modules cache while keeping the large model
    and dataset caches shared.
    """
    if not trust_remote_code:
        return
    if os.environ.get("HF_MODULES_CACHE"):
        return

    base_dir = output_dir or os.path.join("/tmp", "olmo_eval")
    safe_worker_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in worker_id)
    modules_cache = os.path.join(base_dir, ".hf_modules_cache", safe_worker_id)
    os.makedirs(modules_cache, exist_ok=True)
    os.environ["HF_MODULES_CACHE"] = modules_cache

    # These constants are computed at import time. If Transformers was imported
    # before this worker initialized, update the already-imported modules too.
    for module_name in ("transformers.utils.hub", "transformers.dynamic_module_utils"):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "HF_MODULES_CACHE"):
            module.HF_MODULES_CACHE = modules_cache

    worker_logger.info(f"  HF modules cache: {modules_cache}")


def _configure_torch_threads_for_worker(worker_logger: Any) -> None:
    """Optionally cap PyTorch CPU thread pools inside inference workers."""
    torch_threads = os.environ.get("OLMO_EVAL_TORCH_NUM_THREADS")
    if not torch_threads:
        return

    try:
        intra_op_threads = int(torch_threads)
        inter_op_threads = int(os.environ.get("OLMO_EVAL_TORCH_INTEROP_THREADS", "1"))
    except ValueError:
        worker_logger.warning(
            "Ignoring invalid OLMO_EVAL_TORCH_NUM_THREADS/OLMO_EVAL_TORCH_INTEROP_THREADS"
        )
        return

    if intra_op_threads < 1 or inter_op_threads < 1:
        worker_logger.warning(
            "Ignoring non-positive OLMO_EVAL_TORCH_NUM_THREADS/OLMO_EVAL_TORCH_INTEROP_THREADS"
        )
        return

    try:
        import torch

        torch.set_num_threads(intra_op_threads)
        torch.set_num_interop_threads(inter_op_threads)
    except Exception as exc:
        worker_logger.warning(f"Failed to configure PyTorch CPU threads: {exc}")
        return

    worker_logger.info(
        f"  Torch CPU threads: intra_op={intra_op_threads}, inter_op={inter_op_threads}"
    )


def inference_worker(
    worker_id: str,
    gpu_ids: list[int],
    item_queue: mp.Queue,
    result_queue: mp.Queue,
    harness_config_dict: dict[str, Any],
    total_instances: int,
    init_queue: mp.Queue | None = None,
    output_dir: str | None = None,
    num_workers: int = 1,
    start_event: Any | None = None,
) -> None:
    """Worker process that initializes a harness and processes items.

    Processes items in streaming chunks for balanced latency and throughput.
    COMPLETION/LOGLIKELIHOOD requests are batched, CHAT requests use async
    concurrency.

    Args:
        worker_id: Unique worker identifier.
        gpu_ids: GPU IDs to use (sets CUDA_VISIBLE_DEVICES).
        item_queue: Queue of QueueItems (None signals shutdown).
        result_queue: Queue to put ResultItems.
        harness_config_dict: Serialized HarnessConfig.
        total_instances: Total number of instances across all workers.
        init_queue: Optional queue for reporting initialization times.
        output_dir: Output directory for persisting logs (e.g., vLLM server logs).
        num_workers: Number of parallel workers sharing the work.
        start_event: Optional event used to hold all ready workers at a start gate.
    """
    from olmo_eval.common.logging import configure_logging, configure_worker_logging

    configure_logging()

    worker_logger = configure_worker_logging(worker_id)

    from olmo_eval.harness import Harness, HarnessConfig

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

        load_format = provider_kwargs.get("load_format")
        extra_loader_config = provider_kwargs.get("model_loader_extra_config")

        from olmo_eval.launch.config import get_model_short_name

        short_name = get_model_short_name(model_name)
        worker_logger.info(f"Initializing provider: {provider_kind}")
        worker_logger.info(f"  Model: {short_name}")
        if tokenizer:
            worker_logger.info(f"  Tokenizer: {tokenizer}")
        if gpu_ids:
            worker_logger.info(f"  GPUs: {gpu_ids}")
        _configure_torch_threads_for_worker(worker_logger)
        _configure_hf_modules_cache_for_worker(
            worker_id,
            output_dir,
            provider_config.trust_remote_code,
            worker_logger,
        )

        init_start = time.time()

        has_tools = harness_config.has_tools
        enable_auto_tool_choice = has_tools and provider_kind == "vllm_server"

        # Set log_dir for vllm_server provider - matches metrics naming convention
        log_dir = None
        if provider_kind == "vllm_server" and output_dir:
            safe_model = model_name.replace("/", "_").replace("\\", "_")
            log_dir = os.path.join(output_dir, "logs", f"vllm_server_{safe_model}")

        # Only inject vllm-specific kwargs for vllm providers
        vllm_only_overrides: dict[str, Any] = {}
        if provider_kind in ("vllm", "vllm_server"):
            vllm_only_overrides = dict(
                tensor_parallel_size=len(gpu_ids) if gpu_ids else None,
                load_format=load_format,
                model_loader_extra_config=extra_loader_config,
                enable_auto_tool_choice=enable_auto_tool_choice or None,
                log_dir=log_dir,
            )

        harness_config = harness_config.with_provider_overrides(
            max_model_len=max_model_len,
            max_concurrency=max_concurrency,
            tokenizer=tokenizer,
            **vllm_only_overrides,
        )

        # Update metrics config with runtime values (output_dir, provider_kind, model_name)
        if harness_config.metrics is not None and harness_config.metrics.enabled:
            updated_metrics = harness_config.metrics.with_output_dir(
                output_dir or ""
            ).with_metadata(
                provider_kind=provider_kind,
                model_name=model_name,
            )
            harness_config = harness_config.with_metrics(updated_metrics)

        worker_logger.info("Building harness")
        harness = Harness(harness_config)
        worker_logger.info("Harness built")

        # Force provider creation to catch import errors early
        worker_logger.info(
            "Creating provider via harness.provider "
            "(vLLM may load weights, profile KV cache, and compile kernels)"
        )
        _ = harness.provider
        worker_logger.info("Provider object created")

        # Validate scaffold requirements early to fail fast
        if harness_config.scaffold:
            from olmo_eval.harness.scaffolds import validate_scaffold

            worker_logger.info("Validating scaffold")
            validate_scaffold(harness_config.scaffold)
            worker_logger.info("Scaffold validated")

        # Initialize metrics reporters early to establish database connections
        worker_logger.info("Initializing metrics reporters")
        harness.initialize_reporters()
        worker_logger.info("Metrics reporters initialized")

        init_time = time.time() - init_start
        worker_logger.info(f"Provider ready ({init_time:.1f}s)")

        try:
            # Configure agent trace output if using the openai_agents scaffold
            if harness_config.scaffold == "openai_agents" and output_dir:
                from olmo_eval.harness.scaffolds.tracing import configure_trace_output

                configure_trace_output(output_dir)

            # Initialize scaffold resources (e.g., sandbox manager) before processing
            if harness_config.scaffold:
                worker_logger.info("Initializing scaffold resources")
                asyncio.run(harness.scaffold.initialize(harness_config))
                worker_logger.info("Scaffold resources initialized")

            # Get batching strategy from config
            from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy

            batch_config = harness_config.batching or BatchConfig()
            batch_config.validate_for_provider(provider_kind)
            strategy = get_strategy(batch_config)

            concurrency_str = max_concurrency or "unlimited"
            if batch_config.strategy == "streaming":
                worker_logger.info(
                    f"Inference worker ready (strategy={batch_config.strategy}, "
                    f"max_in_flight={concurrency_str})"
                )
            else:
                worker_logger.info(
                    f"Inference worker ready (strategy={batch_config.strategy}, "
                    f"chunk_size={batch_config.chunk_size})"
                )

            if init_queue is not None:
                init_queue.put((worker_id, init_time))

            if start_event is not None:
                worker_logger.info("Waiting for worker start gate")
                start_event.wait()
                worker_logger.info("Worker start gate released")

            worker_logger.info("Starting inference processing loop")
            asyncio.run(
                strategy.run(
                    item_queue,
                    harness,
                    result_queue,
                    max_concurrency,
                    worker_logger,
                    total_instances,
                    num_workers,
                )
            )
            worker_logger.info("Inference processing loop returned")

            worker_logger.info("Processing complete")
        finally:
            # Clean up harness resources (including sandbox manager)
            worker_logger.info("Cleaning up harness resources")
            asyncio.run(harness.cleanup())
            # Clean up provider
            worker_logger.info("Closing provider if supported")
            close_fn = getattr(harness.provider, "close", None)
            if callable(close_fn):
                close_fn()
            worker_logger.info("Worker cleanup complete")

    except Exception as e:
        import traceback

        worker_logger.error(f"Worker process failed: {e}")
        worker_logger.error(traceback.format_exc())
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id=WORKER_FATAL,
                instance_idx=-1,
                instance=None,
                request=None,
                outputs=[],
                error=f"Worker process crashed: {e}",
            )
        )
        sys.exit(1)


__all__ = [
    "inference_worker",
]

"""Worker processes for evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import time
from multiprocessing.synchronize import Event as MPEvent
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.asynq.types import (
    DEFAULT_SCORING_CONCURRENCY,
    SCORER_FATAL,
    WORKER_FATAL,
    QueueItem,
    ResultItem,
    ScoredResponse,
)

logger = get_logger(__name__)


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

        # Set log_dir for vllm_server provider to persist server logs (in logs/ subdir)
        log_dir = None
        if provider_kind == "vllm_server" and output_dir:
            log_dir = os.path.join(output_dir, "logs")

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

        # Validate backend requirements early to fail fast
        if harness_config.backend:
            from olmo_eval.harness.backends import validate_backend

            validate_backend(harness_config.backend)

        init_time = time.time() - init_start
        worker_logger.info(f"Provider ready ({init_time:.1f}s)")

        if init_times is not None:
            init_times[worker_id] = init_time

        try:
            # Configure agent trace output if using openai_agents backend
            if harness_config.backend == "openai_agents" and output_dir:
                from olmo_eval.harness.backends.tracing import configure_trace_output

                configure_trace_output(output_dir)

            # Initialize backend resources (e.g., sandbox manager) before processing
            if harness_config.backend:
                asyncio.run(harness.backend.initialize(harness_config))

            items: list[QueueItem] = []
            while True:
                item = item_queue.get()
                if item is None:
                    break
                items.append(item)

            if items:
                from olmo_eval.runners.asynq.processing import process_items

                asyncio.run(process_items(items, harness, result_queue, max_concurrency))

            worker_logger.info("Processing complete")
        finally:
            # Clean up harness resources (including sandbox manager)
            asyncio.run(harness.cleanup())
            # Clean up provider
            close_fn = getattr(harness.provider, "close", None)
            if callable(close_fn):
                close_fn()

    except Exception as e:
        worker_logger.error(f"Worker process failed: {e}")
        result_queue.put(
            ResultItem(
                model_name=model_name,
                task_id=WORKER_FATAL,
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
    sandbox_configs_list: list[dict[str, Any]] | None = None,
    ready_event: MPEvent | None = None,
    max_concurrency: int = DEFAULT_SCORING_CONCURRENCY,
) -> None:
    """Worker process that scores responses with concurrent execution.

    Reads ScoringItems from scoring_queue, scores them concurrently, and puts
    ScoredResponses on scored_queue.

    This worker manages sandbox lifecycle when sandbox configs are provided,
    similar to how inference_worker manages provider lifecycle.

    Args:
        scoring_queue: Queue of ScoringItems (None signals shutdown).
        scored_queue: Queue to put ScoredResponses.
        total_instances: Total number of instances to score (for progress).
        sandbox_configs_list: Optional list of serialized SandboxConfigs for code execution.
        ready_event: Optional event to signal when worker is ready.
        max_concurrency: Maximum concurrent scoring operations.
    """
    import sys

    from olmo_eval.common.logging import configure_logging

    configure_logging()

    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.common.progress import ProgressLogger
    from olmo_eval.runners.asynq.types import ScoringItem

    sandbox_manager = None
    scoring_context: ScoringContext | None = None

    async def score_item_async(
        item: ScoringItem, context: ScoringContext | None, semaphore: asyncio.Semaphore
    ) -> ScoredResponse:
        """Score a single item with semaphore-controlled concurrency."""
        async with semaphore:
            try:
                scored_list = await item.task.score_responses([item.response], context=context)
                scored = scored_list[0] if scored_list else item.response
                return ScoredResponse(
                    spec=item.spec,
                    instance_idx=item.instance_idx,
                    scored=scored,
                )
            except Exception as e:
                logger.warning(f"Failed to score {item.spec}[{item.instance_idx}]: {e}")
                return ScoredResponse(
                    spec=item.spec,
                    instance_idx=item.instance_idx,
                    scored=item.response,
                )

    async def process_batch(
        items: list[ScoringItem],
        context: ScoringContext | None,
        progress: ProgressLogger,
    ) -> None:
        """Process a batch of items concurrently."""
        semaphore = asyncio.Semaphore(max_concurrency)
        tasks = [score_item_async(item, context, semaphore) for item in items]
        results = await asyncio.gather(*tasks)

        for result in results:
            scored_queue.put(result)
            progress.update(1)

    async def run_scoring_loop() -> None:
        """Main async scoring loop."""
        progress: ProgressLogger | None = None
        batch: list[ScoringItem] = []
        batch_size = max_concurrency * 2  # Buffer 2x concurrency for efficiency

        try:
            while True:
                # Non-blocking check with timeout to allow batching
                try:
                    item: ScoringItem | None = scoring_queue.get(timeout=0.1)
                except queue.Empty:
                    # Timeout - flush any pending batch
                    if batch:
                        if progress is None:
                            progress = ProgressLogger(
                                total=total_instances,
                                desc="Scored",
                                logger=logger,
                                color="blue",
                            )
                        await process_batch(batch, scoring_context, progress)
                        batch = []
                    continue

                if item is None:
                    # Shutdown signal - process remaining batch and exit
                    if batch:
                        if progress is None:
                            progress = ProgressLogger(
                                total=total_instances,
                                desc="Scored",
                                logger=logger,
                                color="blue",
                            )
                        await process_batch(batch, scoring_context, progress)
                    break

                # Create progress logger on first item
                if progress is None:
                    progress = ProgressLogger(
                        total=total_instances, desc="Scored", logger=logger, color="blue"
                    )

                batch.append(item)

                # Process batch when full
                if len(batch) >= batch_size:
                    await process_batch(batch, scoring_context, progress)
                    batch = []

        finally:
            if progress is not None:
                progress.close()

    try:
        if sandbox_configs_list is not None:
            from olmo_eval.harness.sandbox import SandboxConfig, SandboxManager

            sandbox_configs = [SandboxConfig.from_dict(d) for d in sandbox_configs_list]
            logger.info(f"Initializing sandbox manager with {len(sandbox_configs)} config(s)...")
            sandbox_manager = SandboxManager(sandbox_configs, owner="scorer")

            # Start sandbox synchronously using asyncio
            try:
                asyncio.run(sandbox_manager.start())
            except Exception as e:
                logger.error(f"Failed to start sandbox: {e}")
                scored_queue.put(
                    ScoredResponse(
                        spec=SCORER_FATAL,
                        instance_idx=-1,
                        scored=None,
                        error=f"Sandbox initialization failed: {e}",
                    )
                )
                sys.exit(1)

            scoring_context = ScoringContext(execution_env=sandbox_manager)

        # Signal that worker is ready
        if ready_event is not None:
            ready_event.set()

        # Run the async scoring loop
        asyncio.run(run_scoring_loop())

    except Exception as e:
        logger.error(f"Scoring worker failed: {e}")
        # Signal fatal error via the scored queue
        scored_queue.put(
            ScoredResponse(
                spec=SCORER_FATAL,
                instance_idx=-1,
                scored=None,
                error=f"Scoring worker crashed: {e}",
            )
        )
        sys.exit(1)

    finally:
        if sandbox_manager is not None:
            try:
                asyncio.run(sandbox_manager.stop())
            except Exception as e:
                logger.warning(f"Failed to stop sandbox: {e}")


__all__ = [
    "inference_worker",
    "scoring_worker",
]

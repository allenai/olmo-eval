"""Worker processes for evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time
from multiprocessing.synchronize import Event as MPEvent
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.asynq.types import (
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
                from olmo_eval.runners.asynq.processing import process_items

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
    sandbox_config_dict: dict[str, Any] | None = None,
    ready_event: MPEvent | None = None,
) -> None:
    """Worker process that scores individual responses.

    Reads ScoringItems from scoring_queue, scores each response, and puts
    ScoredResponses on scored_queue.

    This worker manages sandbox lifecycle when sandbox_config is provided,
    similar to how inference_worker manages provider lifecycle.

    Args:
        scoring_queue: Queue of ScoringItems (None signals shutdown).
        scored_queue: Queue to put ScoredResponses.
        total_instances: Total number of instances to score (for progress bar).
        sandbox_config_dict: Optional serialized SandboxConfig for code execution.
        ready_event: Optional event to signal when worker is ready.
    """
    import sys

    from tqdm import tqdm

    from olmo_eval.common.logging import configure_logging

    configure_logging()

    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.runners.asynq.types import ScoringItem

    pbar: tqdm | None = None
    sandbox_executor = None
    scoring_context: ScoringContext | None = None

    def score_item(item: ScoringItem, context: ScoringContext | None) -> None:
        try:
            # Score single response, passing context for sandboxed execution
            scored_list = item.task.score_responses([item.response], context=context)
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

    try:
        if sandbox_config_dict is not None:
            from olmo_eval.harness.sandbox import SandboxConfig, SandboxExecutor

            sandbox_config = SandboxConfig.from_dict(sandbox_config_dict)
            logger.info("Initializing sandbox for code execution...")
            sandbox_executor = SandboxExecutor(sandbox_config)

            # Start sandbox synchronously using asyncio
            asyncio.run(sandbox_executor.start())
            scoring_context = ScoringContext(execution_env=sandbox_executor)
            logger.info("Sandbox ready!")

        # Signal that worker is ready
        if ready_event is not None:
            ready_event.set()

        while True:
            item: ScoringItem | None = scoring_queue.get()
            if item is None:
                break

            # Create progress bar on first item (after worker startup logs)
            if pbar is None:
                pbar = tqdm(total=total_instances, desc="Scoring instances", unit="inst")

            score_item(item, scoring_context)
            pbar.update(1)

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
        if pbar is not None:
            pbar.close()

        # Clean up sandbox (similar to provider cleanup in inference_worker)
        if sandbox_executor is not None:
            try:
                asyncio.run(sandbox_executor.stop())
                logger.info("Sandbox stopped")
            except Exception as e:
                logger.warning(f"Failed to stop sandbox: {e}")


__all__ = [
    "inference_worker",
    "scoring_worker",
]

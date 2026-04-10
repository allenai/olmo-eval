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
    total_instances: int,
    init_times: dict[str, float] | None = None,
    output_dir: str | None = None,
    num_workers: int = 1,
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
        init_times: Optional shared dict for tracking initialization times.
        output_dir: Output directory for persisting logs (e.g., vLLM server logs).
        num_workers: Number of parallel workers sharing the work.
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

        from olmo_eval.launch.config import get_model_short_name

        short_name = get_model_short_name(model_name)
        worker_logger.info(f"Initializing provider: {provider_kind}")
        worker_logger.info(f"  Model: {short_name}")
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

        harness = Harness(harness_config)

        # Force provider creation to catch import errors early
        _ = harness.provider

        # Validate backend requirements early to fail fast
        if harness_config.backend:
            from olmo_eval.harness.backends import validate_backend

            validate_backend(harness_config.backend)

        # Initialize metrics reporters early to establish database connections
        harness.initialize_reporters()

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

            worker_logger.info("Processing complete")
        finally:
            # Clean up harness resources (including sandbox manager)
            asyncio.run(harness.cleanup())
            # Clean up provider
            close_fn = getattr(harness.provider, "close", None)
            if callable(close_fn):
                close_fn()

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


def scoring_worker(
    worker_id: str,
    scoring_queue: mp.Queue,
    scored_queue: mp.Queue,
    total_instances: int,
    sandbox_configs_list: list[dict[str, Any]] | None = None,
    ready_event: MPEvent | None = None,
    max_concurrency: int = DEFAULT_SCORING_CONCURRENCY,
    registry_config: dict[str, list[dict[str, Any]]] | None = None,
) -> None:
    """Worker process that scores responses with concurrent execution.

    Reads ScoringItems from scoring_queue, scores them concurrently, and puts
    ScoredResponses on scored_queue.

    This worker manages sandbox lifecycle when sandbox configs are provided,
    similar to how inference_worker manages provider lifecycle.

    Args:
        worker_id: Unique worker identifier (e.g., "scorer-0").
        scoring_queue: Queue of ScoringItems (None signals shutdown).
        scored_queue: Queue to put ScoredResponses.
        total_instances: Total number of instances to score (for progress).
        sandbox_configs_list: Optional list of serialized SandboxConfigs for code execution.
        ready_event: Optional event to signal when worker is ready.
        max_concurrency: Maximum concurrent scoring operations.
        registry_config: Optional config for ProviderRegistry (auxiliary providers).
            Format: {name: [config_dict, ...]} where each config_dict is a
            serialized ProviderConfig with base_url set.
    """
    import sys

    from olmo_eval.common.logging import configure_logging, configure_worker_logging

    configure_logging()
    worker_logger = configure_worker_logging(worker_id)

    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.common.progress import ProgressLogger
    from olmo_eval.runners.asynq.types import ScoringItem

    sandbox_manager = None
    scoring_context: ScoringContext | None = None

    # Cache Task objects by spec so only the first ScoringItem per spec
    # needs to carry (and unpickle) the full Task.
    from olmo_eval.evals.tasks.common import Task

    task_cache: dict[str, Task] = {}

    def get_task(item: ScoringItem) -> Task:
        if item.task is not None:
            task_cache[item.spec] = item.task
        task = task_cache.get(item.spec)
        if task is None:
            raise RuntimeError(
                f"No cached Task for spec {item.spec!r}. "
                "The first ScoringItem for each spec must include the Task."
            )
        return task

    async def run_scoring_loop() -> None:
        """Main async scoring loop using continuous dispatch pattern.

        Uses asyncio.wait(return_when=FIRST_COMPLETED) to maintain a pool of
        in-flight tasks without callbacks or sleeps. This approach is modeled
        after ContinuousBatchDispatcher in dispatch.py.
        """
        from olmo_eval.evals.tasks.common import Task

        progress: ProgressLogger | None = None
        in_flight: dict[asyncio.Task[ScoredResponse], ScoringItem] = {}
        shutdown = False

        # Cache Task objects by spec so only the first ScoringItem per spec
        # needs to carry (and unpickle) the full Task.
        task_cache: dict[str, Task] = {}

        def get_task(item: ScoringItem) -> Task:
            if item.task is not None:
                task_cache[item.spec] = item.task
            task = task_cache.get(item.spec)
            if task is None:
                raise RuntimeError(
                    f"No cached Task for spec {item.spec!r}. "
                    "The first ScoringItem for each spec must include the Task."
                )
            return task

        async def score_item(item: ScoringItem) -> ScoredResponse:
            """Score a single item."""
            try:
                task = get_task(item)
                scored_list = await task.score_responses([item.response], context=scoring_context)
                scored = scored_list[0] if scored_list else item.response
                return ScoredResponse(
                    spec=item.spec,
                    instance_idx=item.instance_idx,
                    scored=scored,
                )
            except Exception as e:
                worker_logger.warning(f"Failed to score {item.spec}[{item.instance_idx}]: {e}")
                return ScoredResponse(
                    spec=item.spec,
                    instance_idx=item.instance_idx,
                    scored=item.response,
                )

        try:
            while not shutdown or in_flight:
                # Top up in-flight tasks to max_concurrency
                while len(in_flight) < max_concurrency and not shutdown:
                    try:
                        item: ScoringItem | None = scoring_queue.get_nowait()
                    except queue.Empty:
                        break

                    if item is None:
                        shutdown = True
                        break

                    if progress is None:
                        worker_logger.info("Starting scoring")
                        progress = ProgressLogger(
                            total=total_instances,
                            desc="Scored",
                            logger=worker_logger,
                            color="blue",
                        )

                    task = asyncio.create_task(score_item(item))
                    in_flight[task] = item

                if not in_flight:
                    # Nothing in flight - blocking wait for item
                    if shutdown:
                        break
                    try:
                        item = scoring_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    if item is None:
                        shutdown = True
                        continue

                    if progress is None:
                        worker_logger.info("Starting scoring")
                        progress = ProgressLogger(
                            total=total_instances,
                            desc="Scored",
                            logger=worker_logger,
                            color="blue",
                        )

                    task = asyncio.create_task(score_item(item))
                    in_flight[task] = item
                    continue

                # Wait for any task to complete (with timeout to poll queue)
                done, _ = await asyncio.wait(
                    in_flight.keys(), return_when=asyncio.FIRST_COMPLETED, timeout=0.1
                )

                for task in done:
                    in_flight.pop(task)
                    result = task.result()
                    scored_queue.put(result)
                    if progress is not None:
                        progress.update(1)

        finally:
            # Wait for remaining tasks
            if in_flight:
                done, _ = await asyncio.wait(in_flight.keys())
                for task in done:
                    result = task.result()
                    scored_queue.put(result)
                    if progress is not None:
                        progress.update(1)
            if progress is not None:
                progress.close()

    try:
        if sandbox_configs_list is not None:
            from olmo_eval.harness.sandbox import SandboxConfig, SandboxManager

            sandbox_configs = [SandboxConfig.from_dict(d) for d in sandbox_configs_list]
            worker_logger.info(
                f"Initializing sandbox manager with {len(sandbox_configs)} config(s)..."
            )
            sandbox_manager = SandboxManager(sandbox_configs, owner="scorer")

            # Start sandbox synchronously using asyncio
            try:
                asyncio.run(sandbox_manager.start())
            except Exception as e:
                worker_logger.error(f"Failed to start sandbox: {e}")
                scored_queue.put(
                    ScoredResponse(
                        spec=SCORER_FATAL,
                        instance_idx=-1,
                        scored=None,
                        error=f"Sandbox initialization failed: {e}",
                    )
                )
                sys.exit(1)

        # Create provider registry from config (servers already running)
        provider_registry = None
        if registry_config:
            from olmo_eval.inference.registry import ProviderRegistry

            provider_registry = ProviderRegistry.from_serialized(registry_config)
            if provider_registry:
                worker_logger.info(
                    f"Provider registry ready with providers: {provider_registry.names}"
                )

        scoring_context = ScoringContext(
            execution_env=sandbox_manager,
            scoring_concurrency=max_concurrency,
            inference_pool=provider_registry,
        )

        # Signal that worker is ready
        worker_logger.info("Scorer ready")
        if ready_event is not None:
            ready_event.set()

        # Run the async scoring loop
        asyncio.run(run_scoring_loop())

    except Exception as e:
        worker_logger.error(f"Scoring worker failed: {e}")
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
                worker_logger.warning(f"Failed to stop sandbox: {e}")


__all__ = [
    "inference_worker",
    "scoring_worker",
]

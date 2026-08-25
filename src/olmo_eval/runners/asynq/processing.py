"""Request processing for async evaluation runners."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import replace
from typing import TYPE_CHECKING

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMRequest
from olmo_eval.inference.errors import classify_terminal_provider_error
from olmo_eval.inference.metrics.core.stats import compute_batch_hash
from olmo_eval.runners.asynq.types import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.harness import Harness

logger = get_logger(__name__)


def _result_request(request: LMRequest | None) -> LMRequest | None:
    """The request as echoed back to the parent: image payloads stripped.

    Scoring and storage read the instance and outputs, never the pixels, so the
    (possibly resolved) images tuple is dropped rather than pickled back through
    the result queue.
    """
    if request is None or request.images is None:
        return request
    return replace(request, images=None)


def _get_native_ids(items: list[QueueItem]) -> list[str]:
    """Extract native IDs from queue items for batch hashing."""
    return [f"{item.task_id}:{item.instance_idx}" for item in items]


def _flush_batch_metrics(harness: Harness, items: list[QueueItem], log: logging.Logger) -> None:
    """Flush batch metrics without letting telemetry failures interrupt inference."""
    try:
        harness.flush_metrics(compute_batch_hash(_get_native_ids(items)))
    except Exception as e:
        log.warning(f"Failed to flush batch metrics: {e}")


def _format_cause(cause: BaseException) -> str:
    """Format the cause of an exception with fallback for empty strings."""
    cause_type = type(cause).__qualname__
    cause_str = str(cause)

    # If str(cause) is non-empty, use it
    if cause_str:
        return f"{cause_type}: {cause_str}"

    # Try args (often contains the real message)
    if cause.args:
        args_str = ", ".join(str(a) for a in cause.args if a)
        if args_str:
            return f"{cause_type}: {args_str}"

    # Just return the type name
    return cause_type


def _format_error_detail(exc: Exception) -> str:
    """Format exception with HTTP details for debugging."""
    parts = [f"type: {type(exc).__qualname__}"]

    # HTTP status code
    status = getattr(exc, "status_code", None)
    if status is not None:
        parts.append(f"status_code: {status}")

    # Request URL from response
    response = getattr(exc, "response", None)
    if response is not None:
        url = getattr(response, "url", None)
        if url is not None:
            parts.append(f"url: {url}")

    # Error message
    message = getattr(exc, "message", None) or str(exc)
    if len(message) > 500:
        message = message[:500] + "..."
    parts.append(f"message: {message}")

    # Root cause (e.g., httpx.ReadTimeout)
    cause = exc.__cause__
    if cause is not None:
        parts.append(f"cause: {_format_cause(cause)}")

    return " | ".join(parts)


async def process_chat_request(
    item: QueueItem,
    harness: Harness,
    result_queue: mp.Queue,
    worker_logger: logging.Logger | None = None,
) -> None:
    """Process a single CHAT request via harness.run().

    CHAT requests use the async harness.run() method which handles agentic
    loops with tool calls. These must be processed individually.

    Args:
        item: Queue item to process (must be CHAT type).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        worker_logger: Logger with worker identification.
    """
    log = worker_logger or logger
    from dataclasses import replace as dataclass_replace

    # Build trace metadata for observability
    trace_metadata = {
        "task_id": item.task_id,
        "instance_idx": item.instance_idx,
        "instance_id": item.instance.metadata.get("id", str(item.instance_idx)),
        "date_cutoff": item.instance.metadata.get("retrieval_date_cutoff"),
    }
    prepared_request = harness._apply_config(item.request)
    request_trace = harness.provider.describe_request(prepared_request, item.sampling_params)

    try:
        harness_result = await harness.run(
            item.request, item.sampling_params, trace_metadata=trace_metadata
        )
        final_output = harness_result.final_output

        if harness_result.trajectory is not None:
            output_with_metadata = dataclass_replace(
                final_output,
                metadata={
                    **(final_output.metadata or {}),
                    "trajectory": harness_result.trajectory.to_dict(),
                    "max_turns_reached": harness_result.max_turns_reached,
                    "total_tool_calls": harness_result.total_tool_calls,
                    "num_turns": harness_result.num_turns,
                },
            )
        else:
            output_with_metadata = final_output

        result_queue.put(
            ResultItem(
                model_name=item.model_name,
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=_result_request(prepared_request),
                request_trace=request_trace,
                outputs=[output_with_metadata],
                error=harness_result.error,
                attempt=item.attempt,
            )
        )

    except Exception as e:
        import traceback

        terminal_error = classify_terminal_provider_error(e)
        if terminal_error is not None:
            raise terminal_error from e

        error_detail = _format_error_detail(e)
        full_tb = traceback.format_exc()
        log.error(f"Error on CHAT instance {item.instance_idx}: {error_detail}\n{full_tb}")

        result_queue.put(
            ResultItem(
                model_name=item.model_name,
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=_result_request(prepared_request),
                request_trace=request_trace,
                outputs=[],
                error=error_detail,
                attempt=item.attempt,
            )
        )


async def process_batch(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    worker_logger: logging.Logger | None = None,
) -> None:
    """Process a batch of COMPLETION or LOGLIKELIHOOD requests.

    All items must have the same request_type and sampling_params.
    Calls the underlying provider once for the entire batch using the harness-
    transformed requests.

    Args:
        items: List of queue items to process (same type and sampling_params).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
        worker_logger: Logger with worker identification.
    """
    from olmo_eval.common.types import RequestType

    log = worker_logger or logger

    if not items:
        return

    request_type = items[0].request.request_type
    sampling_params = items[0].sampling_params
    requests = [item.request for item in items]
    prepared_requests = [harness._apply_config(request) for request in requests]
    request_traces = [
        harness.provider.describe_request(request, sampling_params) for request in prepared_requests
    ]

    reported = 0
    try:
        if request_type == RequestType.LOGLIKELIHOOD:
            all_outputs = await harness.provider.alogprobs(prepared_requests, sampling_params)
        else:
            all_outputs = await harness.provider.agenerate(prepared_requests, sampling_params)

        # Map outputs back to individual items
        for item, prepared_request, request_trace, outputs in zip(
            items, prepared_requests, request_traces, all_outputs, strict=True
        ):
            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=_result_request(prepared_request),
                    request_trace=request_trace,
                    outputs=outputs,
                    error=None,
                    attempt=item.attempt,
                )
            )
            reported += 1

    except Exception as e:
        terminal_error = classify_terminal_provider_error(e)
        if terminal_error is not None:
            raise terminal_error from e

        # Batch failed - report errors only for items that never produced a result
        error_detail = _format_error_detail(e)
        log.error(
            f"Batch error ({len(items) - reported} of {len(items)} items affected): {error_detail}"
        )

        for item, prepared_request, request_trace in zip(
            items[reported:], prepared_requests[reported:], request_traces[reported:], strict=True
        ):
            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=_result_request(prepared_request),
                    request_trace=request_trace,
                    outputs=[],
                    error=error_detail,
                    attempt=item.attempt,
                )
            )
    else:
        _flush_batch_metrics(harness, items, log)


async def process_items(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
    max_concurrency: int | None = None,
    worker_logger: logging.Logger | None = None,
    show_progress: bool = True,
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
        worker_logger: Logger with worker identification.
        show_progress: Whether to show progress logging (disable for streaming).
    """
    from olmo_eval.common.types import RequestType, SamplingParams

    log = worker_logger or logger

    chat_items: list[QueueItem] = []
    batchable_items: list[QueueItem] = []

    has_scaffold = bool(harness.config.scaffold)

    for item in items:
        if item.request.request_type == RequestType.CHAT and has_scaffold:
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
            await process_batch(batch, harness, result_queue, log)

    if chat_items:
        from olmo_eval.inference.dispatch import dispatch_concurrent

        async def process(item: QueueItem) -> None:
            await process_chat_request(item, harness, result_queue, log)

        if show_progress:
            from olmo_eval.common.progress import ProgressLogger

            progress = ProgressLogger(
                total=len(chat_items), desc="Processed", logger=log, color="green"
            )

            def on_progress(done: int, total: int) -> None:
                progress.update(1)

            try:
                await dispatch_concurrent(
                    chat_items,
                    process,
                    max_in_flight=max_concurrency or len(chat_items),
                    on_progress=on_progress,
                )
            finally:
                progress.close()
        else:
            await dispatch_concurrent(
                chat_items,
                process,
                max_in_flight=max_concurrency or len(chat_items),
            )

        _flush_batch_metrics(harness, chat_items, log)


__all__ = [
    "process_chat_request",
    "process_batch",
    "process_items",
]

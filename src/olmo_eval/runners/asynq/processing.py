"""Request processing for async evaluation runners."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
from typing import TYPE_CHECKING

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.asynq.types import QueueItem, ResultItem

if TYPE_CHECKING:
    from olmo_eval.harness import Harness

logger = get_logger(__name__)


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
        parts.append(f"cause: {type(cause).__qualname__}: {cause}")

    return " | ".join(parts)


async def process_chat_request(
    item: QueueItem,
    harness: Harness,
    result_queue: mp.Queue,
) -> None:
    """Process a single CHAT request via harness.run().

    CHAT requests use the async harness.run() method which handles agentic
    loops with tool calls. These must be processed individually.

    Args:
        item: Queue item to process (must be CHAT type).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
    """
    from dataclasses import replace as dataclass_replace

    try:
        harness_result = await harness.run(item.request, item.sampling_params)
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
                request=item.request,
                outputs=[output_with_metadata],
                error=harness_result.error,
                attempt=item.attempt,
            )
        )

    except Exception as e:
        error_detail = _format_error_detail(e)
        logger.warning(f"Error on CHAT instance {item.instance_idx}: {error_detail}")

        result_queue.put(
            ResultItem(
                model_name=item.model_name,
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=item.request,
                outputs=[],
                error=error_detail,
                attempt=item.attempt,
            )
        )


async def process_batch(
    items: list[QueueItem],
    harness: Harness,
    result_queue: mp.Queue,
) -> None:
    """Process a batch of COMPLETION or LOGLIKELIHOOD requests.

    All items must have the same request_type and sampling_params.
    Calls harness.agenerate or harness.alogprobs once for the entire batch.

    Args:
        items: List of queue items to process (same type and sampling_params).
        harness: Harness instance for execution.
        result_queue: Queue to put results.
    """
    from olmo_eval.common.types import RequestType

    if not items:
        return

    request_type = items[0].request.request_type
    sampling_params = items[0].sampling_params
    requests = [item.request for item in items]

    try:
        if request_type == RequestType.LOGLIKELIHOOD:
            all_outputs = await harness.alogprobs(requests)
        else:
            all_outputs = await harness.agenerate(requests, sampling_params)

        # Map outputs back to individual items
        for item, outputs in zip(items, all_outputs, strict=True):
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

    except Exception as e:
        # Batch failed - report error for all items
        error_detail = _format_error_detail(e)
        logger.warning(f"Batch error ({len(items)} items): {error_detail}")

        for item in items:
            result_queue.put(
                ResultItem(
                    model_name=item.model_name,
                    task_id=item.task_id,
                    instance_idx=item.instance_idx,
                    instance=item.instance,
                    request=item.request,
                    outputs=[],
                    error=error_detail,
                    attempt=item.attempt,
                )
            )


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
    from olmo_eval.common.types import RequestType, SamplingParams

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


__all__ = [
    "process_chat_request",
    "process_batch",
    "process_items",
]

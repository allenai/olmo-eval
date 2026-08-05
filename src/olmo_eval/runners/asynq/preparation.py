"""Task preparation functions for async evaluation runners."""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import Response, SamplingParams
from olmo_eval.evals.tasks.common import Task, get_task
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker
from olmo_eval.runners.common.types import TaskResult
from olmo_eval.runners.io.builders import build_predictions, build_requests_from_responses
from olmo_eval.runners.processing.utils import get_metric_metadata

logger = get_logger(__name__)


def prepare_task_items(
    spec: str,
    model_name: str,
    overrides: dict[str, Any] | None,
    sampling_overrides: dict[str, Any] | None = None,
    harness_tool_names: Sequence[str] = (),
) -> tuple[Task, list[QueueItem]]:
    """Prepare a task and its queue items.

    Args:
        spec: Task specification string
        model_name: Model name this task is for
        overrides: Optional config overrides (num_fewshot, limit, fewshot_seed)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)
        harness_tool_names: Tool names the harness exposes, so a task that names
            tools in its prompt can describe the ones that really exist.

    Returns:
        Tuple of (Task instance for scoring, list of QueueItems)

    """
    task = get_task(spec)

    if overrides:
        task.config = replace(task.config, **overrides)

    if harness_tool_names:
        task.config = replace(task.config, harness_tool_names=tuple(harness_tool_names))

    # Build sampling params from overrides
    existing_params = task.config.sampling_params or SamplingParams()

    # Apply sampling_overrides
    if sampling_overrides:
        for key, value in sampling_overrides.items():
            if hasattr(existing_params, key):
                existing_params = replace(existing_params, **{key: value})

    # Always update task config with final sampling params (so finalize_task captures them)
    task.config = replace(task.config, sampling_params=existing_params)

    instances = select_instances(list(task.instances), task.config)

    items = [
        QueueItem(
            model_name=model_name,
            task_id=spec,
            instance_idx=idx,
            instance=inst,
            request=task.format_request(inst),
            sampling_params=task.get_sampling_params(inst) or existing_params,
        )
        for idx, inst in enumerate(instances)
    ]

    return task, items



def select_instances(population, config):
    """Which instances this run should do: the sample, minus any earlier run's share of it.

    `limit` samples reproducibly -- `random.Random(seed).sample(population, limit)` -- so a larger
    sample at the same seed contains a smaller one. It does not *begin* with it, though:
    `sample(100)[:40]` is not `sample(40)`. So growing a sample cheaply means excluding the earlier
    run by membership, which is what OLMO_EVAL_SKIP_SAMPLE_OF_SIZE does.

    Sampling and exclusion are decided together here because separating them hid a bug: the
    exclusion used to sit inside the sampling branch, and a task with exactly as many instances as
    the limit never enters that branch.
    """

    instances = list(population)
    if config.limit and len(instances) > config.limit:
        # Use random.sample for reproducible random sampling (matches oe-eval-internal behavior)
        instances = random.Random(config.seed).sample(instances, config.limit)
    return _drop_already_sampled(population, instances, config)


def _drop_already_sampled(population, sampled, config):
    """Remove the instances a smaller sample at the same seed would have selected.

    Growing a sample from 40 to 100 should cost 60 runs, not 100. The larger sample provably
    contains the smaller one at a fixed seed, but it does not *begin* with it -- `sample(100)[:40]`
    is not `sample(40)` -- so the earlier run's instances have to be excluded by membership rather
    than by position.

    Set OLMO_EVAL_SKIP_SAMPLE_OF_SIZE to the earlier run's limit. Off unless set, and ignored
    unless it is smaller than the current limit.
    """

    raw = os.environ.get("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE", "").strip()
    if not raw:
        return sampled
    try:
        earlier = int(raw)
    except ValueError:
        logger.warning("OLMO_EVAL_SKIP_SAMPLE_OF_SIZE=%r is not a number; ignoring", raw)
        return sampled
    limit = config.limit or len(population)
    if earlier <= 0 or earlier >= limit or earlier >= len(population):
        return sampled

    already = random.Random(config.seed).sample(population, earlier)
    seen = {id(instance) for instance in already}
    remaining = [instance for instance in sampled if id(instance) not in seen]
    logger.info(
        "OLMO_EVAL_SKIP_SAMPLE_OF_SIZE=%d: running %d of the %d sampled instances",
        earlier,
        len(remaining),
        len(sampled),
    )
    return remaining

def build_requests_from_items(items: list[QueueItem], task_name: str) -> list[dict]:
    """Build request objects from queue items for early writing.

    Args:
        items: List of QueueItems (with instance, request, sampling_params)
        task_name: Name of the task

    Returns:
        List of request dicts suitable for JSONL output
    """
    from olmo_eval.runners.io.builders import build_requests

    instances = [item.instance for item in items]
    requests = [item.request for item in items]
    sampling_params = items[0].sampling_params if items else None

    return build_requests(instances, requests, task_name, sampling_params)


async def finalize_task(tracker: TaskTracker) -> TaskResult:
    """Finalize a task tracker into a TaskResult.

    Args:
        tracker: Completed TaskTracker

    Returns:
        TaskResult with metrics and predictions
    """
    import time

    duration = time.time() - tracker.start_time

    # Task-level error (e.g., prep failed) - no results possible
    if tracker.error:
        return TaskResult(
            spec=tracker.spec,
            config={},
            num_instances=tracker.total_instances,
            metrics={},
            error=tracker.error,
            duration_seconds=duration,
        )

    if tracker.task is None:
        return TaskResult(
            spec=tracker.spec,
            config={},
            num_instances=tracker.total_instances,
            metrics={},
            error="Task preparation failed",
            duration_seconds=duration,
        )

    # Check if we have any successful responses
    if not tracker.responses:
        # All instances failed
        error_summary = tracker.get_error_summary() or "All instances failed"
        return TaskResult(
            spec=tracker.spec,
            config=tracker.task.config.to_dict(),
            num_instances=tracker.total_instances,
            metrics={},
            error=error_summary,
            duration_seconds=duration,
        )

    # Sort responses by index (only successful ones)
    responses = [tracker.responses[i] for i in sorted(tracker.responses.keys())]

    # Score and compute metrics
    scored = await tracker.task.score_responses(responses)
    metrics = tracker.task.compute_metrics(scored)

    # Build predictions
    predictions = build_predictions(scored, metrics=tracker.task.config.metrics)
    requests = build_requests_from_responses(scored, tracker.task.config.name)

    # Get task config for serialization
    task_config = tracker.task.config

    # Extract metric metadata (returns "metric:scorer" format)
    primary_metric = get_metric_metadata(tracker.task)

    # Add warning about failed instances if any
    error_summary = tracker.get_error_summary()
    if error_summary:
        # Log failed instances but still return partial results
        logger.warning(
            f"Task {tracker.spec} completed with failures: {error_summary}. "
            f"Computed metrics on {len(responses)}/{tracker.total_instances} instances."
        )

    return TaskResult(
        spec=tracker.spec,
        config=task_config.to_dict(),
        num_instances=len(responses),
        metrics=metrics,
        duration_seconds=duration,
        predictions=predictions,
        requests=requests,
        primary_metric=primary_metric,
        # Only set error if ALL instances failed (partial failures are logged as warnings)
    )


def compute_task_metrics(
    spec: str,
    task: Task,
    scored_responses: list[Response],
    failed_instances: dict[int, str],
    total_instances: int,
    duration_seconds: float,
) -> TaskResult:
    """Compute metrics from pre-scored responses.

    Args:
        spec: Task specification string.
        task: Task instance for metric computation.
        scored_responses: List of already-scored responses.
        failed_instances: Dict of instance_idx -> error message for failures.
        total_instances: Total number of instances in the task.
        duration_seconds: Duration of the task.

    Returns:
        TaskResult with metrics and predictions.
    """
    if not scored_responses:
        error_summary = (
            f"{len(failed_instances)} instances failed" if failed_instances else "No responses"
        )
        return TaskResult(
            spec=spec,
            config=task.config.to_dict(),
            num_instances=0,
            metrics={},
            error=error_summary,
            duration_seconds=duration_seconds,
        )

    # Compute metrics from pre-scored responses
    metrics = task.compute_metrics(scored_responses)

    # Build predictions
    predictions = build_predictions(scored_responses, metrics=task.config.metrics)
    requests = build_requests_from_responses(scored_responses, task.config.name)

    # Extract metric metadata
    primary_metric = get_metric_metadata(task)

    # Add warning about failed instances if any
    error_summary = None
    if failed_instances:
        if len(failed_instances) == 1:
            idx, err = next(iter(failed_instances.items()))
            error_summary = f"Instance {idx} failed: {err}"
        else:
            first_error = next(iter(failed_instances.values()))
            error_summary = f"{len(failed_instances)} instances failed (first: {first_error})"
        logger.warning(
            f"Task {spec} completed with failures: {error_summary}. "
            f"Computed metrics on {len(scored_responses)}/{total_instances} instances."
        )

    return TaskResult(
        spec=spec,
        config=task.config.to_dict(),
        num_instances=len(scored_responses),
        metrics=metrics,
        duration_seconds=duration_seconds,
        predictions=predictions,
        requests=requests,
        primary_metric=primary_metric,
        # Only set error if ALL instances failed (partial failures are logged as warnings)
    )


__all__ = [
    "prepare_task_items",
    "build_requests_from_items",
    "finalize_task",
    "compute_task_metrics",
]

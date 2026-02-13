"""Task preparation functions for async evaluation runners."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import Response, SamplingParams
from olmo_eval.evals.tasks.common import Task, get_task
from olmo_eval.runners.asynq.types import QueueItem, TaskTracker
from olmo_eval.runners.common.types import TaskResult
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.utils import get_metric_metadata

logger = get_logger(__name__)


def prepare_task_items(
    spec: str,
    model_name: str,
    overrides: dict[str, Any] | None,
    sampling_overrides: dict[str, Any] | None = None,
) -> tuple[Task, list[QueueItem]]:
    """Prepare a task and its queue items.

    Args:
        spec: Task specification string
        model_name: Model name this task is for
        overrides: Optional config overrides (num_fewshot, limit, fewshot_seed)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)

    Returns:
        Tuple of (Task instance for scoring, list of QueueItems)

    """
    task = get_task(spec)

    if overrides:
        task.config = replace(task.config, **overrides)

    # Build sampling params from overrides
    existing_params = task.config.sampling_params or SamplingParams()

    # Apply sampling_overrides
    if sampling_overrides:
        for key, value in sampling_overrides.items():
            if hasattr(existing_params, key):
                existing_params = replace(existing_params, **{key: value})

    # Always update task config with final sampling params (so finalize_task captures them)
    task.config = replace(task.config, sampling_params=existing_params)

    instances = list(task.instances)
    if task.config.limit:
        # Shuffle with seed for reproducible random sampling
        rng = random.Random(task.config.seed)
        instances = instances.copy()
        rng.shuffle(instances)
        instances = instances[: task.config.limit]

    items = [
        QueueItem(
            model_name=model_name,
            task_id=spec,
            instance_idx=idx,
            instance=inst,
            request=task.format_request(inst),
            sampling_params=existing_params,
        )
        for idx, inst in enumerate(instances)
    ]

    return task, items


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
    predictions = build_predictions(scored)

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
    predictions = build_predictions(scored_responses)

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
        primary_metric=primary_metric,
        # Only set error if ALL instances failed (partial failures are logged as warnings)
    )


__all__ = [
    "prepare_task_items",
    "build_requests_from_items",
    "finalize_task",
    "compute_task_metrics",
]

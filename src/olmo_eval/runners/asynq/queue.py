"""Data structures and task preparation for async evaluation runners."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.evals.tasks import Task, get_task
from olmo_eval.runners.utils import TaskResult, build_predictions, get_metric_metadata

logger = get_logger(__name__)

# Sentinel value for fatal worker errors
WORKER_FATAL_TASK_ID = "__WORKER_FATAL__"

# -----------------------------------------------------------------------------
# Data structures for instance-level queuing
# -----------------------------------------------------------------------------


@dataclass
class QueueItem:
    """Single instance ready for generation."""

    model_name: str  # Which model this is for
    task_id: str  # Task spec string
    instance_idx: int  # Index within task's instance list
    instance: Instance
    request: LMRequest  # Pre-formatted request
    sampling_params: SamplingParams | None = None
    attempt: int = 0  # Retry attempt number


@dataclass
class TaskTracker:
    """Tracks completion state for a single (model, task) pair."""

    model_name: str  # Which model this is for
    spec: str
    task: Task | None  # None if task prep failed
    total_instances: int
    completed_count: int = 0
    responses: dict[int, Response] = field(default_factory=dict)
    failed_instances: dict[int, str] = field(default_factory=dict)  # idx -> error message
    error: str | None = None  # Task-level error (e.g., prep failed)
    start_time: float = field(default_factory=time.time)

    def is_complete(self) -> bool:
        """Check if task is complete (all instances done, including failed ones)."""
        if self.error is not None:
            return True  # Task-level error stops everything
        processed = self.completed_count + len(self.failed_instances)
        return processed >= self.total_instances

    def add_response(self, idx: int, response: Response) -> bool:
        """Add a response. Returns True if task is now complete."""
        self.responses[idx] = response
        self.completed_count += 1
        return self.is_complete()

    def add_failure(self, idx: int, error: str) -> bool:
        """Record a failed instance. Returns True if task is now complete."""
        self.failed_instances[idx] = error
        return self.is_complete()

    def get_error_summary(self) -> str | None:
        """Get summary of failures, if any."""
        if self.error:
            return self.error
        if not self.failed_instances:
            return None
        if len(self.failed_instances) == 1:
            idx, err = next(iter(self.failed_instances.items()))
            return f"Instance {idx} failed: {err}"
        first_error = next(iter(self.failed_instances.values()))
        return f"{len(self.failed_instances)} instances failed (first: {first_error})"


@dataclass
class ResultItem:
    """Result for a single instance from the worker."""

    model_name: str  # Which model produced this result
    task_id: str
    instance_idx: int
    instance: Instance
    request: LMRequest
    outputs: list[LMOutput]
    error: str | None = None
    attempt: int = 0


@dataclass
class ScoringItem:
    """Item to be scored by the scoring worker."""

    spec: str
    tracker: TaskTracker


@dataclass
class ScoredResult:
    """Result from the scoring worker."""

    spec: str
    result: TaskResult


# -----------------------------------------------------------------------------
# Task preparation functions
# -----------------------------------------------------------------------------


def prepare_task_items(
    spec: str,
    model_name: str,
    overrides: dict[str, Any] | None,
    temperature: float | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> tuple[Task, list[QueueItem]]:
    """Prepare a task and its queue items.

    Args:
        spec: Task specification string
        model_name: Model name this task is for
        overrides: Optional config overrides (num_fewshot, limit, fewshot_seed)
        temperature: Optional temperature for sampling (deprecated, use sampling_overrides)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)

    Returns:
        Tuple of (Task instance for scoring, list of QueueItems)

    """
    task = get_task(spec)

    if overrides:
        task.config = replace(task.config, **overrides)

    # Build sampling params from overrides
    # Priority: sampling_overrides > temperature > task default
    existing_params = task.config.sampling_params or SamplingParams()

    # Apply legacy temperature parameter (deprecated)
    if temperature is not None:
        existing_params = replace(existing_params, temperature=temperature)

    # Apply sampling_overrides (highest priority)
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
    from olmo_eval.runners.utils import build_requests

    instances = [item.instance for item in items]
    requests = [item.request for item in items]
    sampling_params = items[0].sampling_params if items else None

    return build_requests(instances, requests, task_name, sampling_params)


def finalize_task(tracker: TaskTracker) -> TaskResult:
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
    scored = tracker.task.score_responses(responses)
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
        # Include error summary if there were partial failures
        error=error_summary if tracker.failed_instances else None,
    )


__all__ = [
    "WORKER_FATAL_TASK_ID",
    "QueueItem",
    "TaskTracker",
    "ResultItem",
    "ScoringItem",
    "ScoredResult",
    "prepare_task_items",
    "build_requests_from_items",
    "finalize_task",
]

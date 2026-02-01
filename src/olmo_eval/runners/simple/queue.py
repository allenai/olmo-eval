"""Data structures and task preparation for async evaluation runners."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import Instance, LMOutput, LMRequest, Response, SamplingParams
from olmo_eval.evals.tasks import Task, get_task
from olmo_eval.runners.utils import TaskResult, build_predictions, get_metric_metadata

logger = get_logger(__name__)


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
    error: str | None = None
    start_time: float = field(default_factory=time.time)

    def is_complete(self) -> bool:
        """Check if task is complete (all instances done or error occurred)."""
        return self.completed_count >= self.total_instances or self.error is not None

    def add_response(self, idx: int, response: Response) -> bool:
        """Add a response. Returns True if task is now complete."""
        self.responses[idx] = response
        self.completed_count += 1
        return self.is_complete()


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

    Raises:
        NotImplementedError: If the task is an AgentTask (use sync runner instead)
    """
    from olmo_eval.evals.tasks import AgentTask

    task = get_task(spec)

    # Agent tasks require special handling with vLLM server
    if isinstance(task, AgentTask):
        raise NotImplementedError(
            f"Agent task '{spec}' is not supported in async mode. "
            "Use the sync runner instead: olmo-eval run (without --async or --async-stream)"
        )

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

    # Sort responses by index
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

    return TaskResult(
        spec=tracker.spec,
        config=task_config.to_dict(),
        num_instances=len(responses),
        metrics=metrics,
        duration_seconds=duration,
        predictions=predictions,
        primary_metric=primary_metric,
    )


__all__ = [
    "QueueItem",
    "TaskTracker",
    "ResultItem",
    "prepare_task_items",
    "build_requests_from_items",
    "finalize_task",
]

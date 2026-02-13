"""Data structures for async evaluation runners."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from olmo_eval.common.types import Instance, LMOutput, LMRequest, Response, SamplingParams

if TYPE_CHECKING:
    from olmo_eval.evals.tasks.common import Task

# Sentinel values for fatal worker errors
WORKER_FATAL = "__WORKER_FATAL__"
SCORER_FATAL = "__SCORER_FATAL__"

# Default concurrency for scoring worker
DEFAULT_SCORING_CONCURRENCY = 8


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
    """Single response to be scored by the scoring worker."""

    spec: str
    instance_idx: int
    response: Response
    task: Task


@dataclass
class ScoredResponse:
    """Single scored response from the scoring worker."""

    spec: str
    instance_idx: int
    scored: Response | None  # Response with score populated (None if fatal error)
    error: str | None = None  # Error message if scoring failed


__all__ = [
    "WORKER_FATAL",
    "SCORER_FATAL",
    "QueueItem",
    "TaskTracker",
    "ResultItem",
    "ScoringItem",
    "ScoredResponse",
]

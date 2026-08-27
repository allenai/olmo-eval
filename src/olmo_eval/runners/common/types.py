"""Core types and constants for evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Filename suffixes for output files (consistent across all runners and storage backends)
PREDICTIONS_SUFFIX = "-predictions.jsonl"
REQUESTS_SUFFIX = "-requests.jsonl"

# Fraction of a task's instances allowed to hard-fail before the task is marked failed.
# A hard failure is an instance that came back with an error and no outputs at all, so it
# contributes nothing to the metrics. Soft failures (an error alongside a usable output,
# e.g. MaxTurnsExceeded with a fallback answer) are still scored and do not count here.
# Independently of this budget, a task that saves zero instances always fails: it
# produced no metrics, so there is nothing to publish at any rate.
DEFAULT_MAX_HARD_FAILURE_RATE = 0.05


@dataclass
class TaskResult:
    """Result from executing a single task.

    Metrics are stored in a nested structure: {metric_name: {scorer_name: score}}.
    The primary_metric uses "metric_name:scorer_name" format.
    """

    spec: str
    config: dict[str, Any]
    num_instances: int  # Instances that were scored and saved
    metrics: dict[str, dict[str, float]]
    error: str | None = None
    duration_seconds: float = 0.0
    predictions: list[dict] | None = None
    requests: list[dict] | None = None  # oe-eval compatible request objects
    primary_metric: str | None = None  # Format: "metric_name:scorer_name"
    # Failure accounting. `instances_processed` counts every instance the runner saw
    # (saved plus hard-failed), so `num_instances` vs `instances_processed` is the
    # saved-vs-processed gap that a partially failed run would otherwise hide.
    error_summary: str | None = None
    instances_processed: int = 0
    instances_failed: int = 0
    hard_failure_rate_exceeded: bool = False

    @property
    def hard_failure_rate(self) -> float:
        """Fraction of processed instances that hard-failed."""
        if self.instances_processed <= 0:
            return 0.0
        return self.instances_failed / self.instances_processed

    def to_dict(self, include_predictions: bool = False) -> dict[str, Any]:
        """Serialize to dictionary for JSON output.

        Args:
            include_predictions: Whether to include predictions in the output.
                Defaults to False since predictions are typically written separately.

        Returns:
            Dictionary with task result data.
        """
        result: dict[str, Any] = {
            "config": self.config,
            "num_instances": self.num_instances,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
            "instances_saved": self.num_instances,
            "instances_processed": self.instances_processed,
            "instances_failed": self.instances_failed,
        }
        if self.primary_metric:
            result["primary_metric"] = self.primary_metric
        if self.error:
            result["error"] = self.error
        if self.error_summary:
            result["error_summary"] = self.error_summary
        if include_predictions and self.predictions:
            result["predictions"] = self.predictions
        return result

"""Core types and constants for evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Filename suffixes for output files (consistent across all runners and storage backends)
PREDICTIONS_SUFFIX = "-predictions.jsonl"
REQUESTS_SUFFIX = "-requests.jsonl"


@dataclass
class TaskResult:
    """Result from executing a single task.

    Metrics are stored in a nested structure: {metric_name: {scorer_name: score}}.
    The primary_metric uses "metric_name:scorer_name" format.
    """

    spec: str
    config: dict[str, Any]
    num_instances: int
    metrics: dict[str, dict[str, float]]
    error: str | None = None
    duration_seconds: float = 0.0
    predictions: list[dict] | None = None
    requests: list[dict] | None = None  # oe-eval compatible request objects
    primary_metric: str | None = None  # Format: "metric_name:scorer_name"

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
        }
        if self.primary_metric:
            result["primary_metric"] = self.primary_metric
        if include_predictions and self.predictions:
            result["predictions"] = self.predictions
        return result

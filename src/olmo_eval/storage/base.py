"""Base classes and data models for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from olmo_eval.core.types import EvalResult, StoredTaskResult, compute_model_hash

__all__ = [
    "StorageBackend",
    "compute_model_hash",  # Re-exported from core for convenience
    "convert_runner_results",
]


class StorageBackend(ABC):
    """Abstract base class for result storage backends."""

    @abstractmethod
    def save(
        self,
        result: EvalResult,
        instances_by_task: dict[str, list[dict[str, Any]]] | None = None,
    ) -> str:
        """Save an evaluation result.

        Args:
            result: The evaluation result to save.
            instances_by_task: Optional dict mapping task_name -> list of instance dicts.
                Each instance dict should have native_id, doc_id, instance_metrics, etc.

        Returns:
            The experiment_id of the saved result.
        """
        ...

    @abstractmethod
    def get(self, experiment_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by experiment_id.

        Args:
            experiment_id: The unique identifier of the result.

        Returns:
            The evaluation result if found, None otherwise.
        """
        ...

    @abstractmethod
    def query(
        self,
        model_name: str | None = None,
        task_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[EvalResult]:
        """Query evaluation results by filters.

        Args:
            model_name: Filter by model name.
            task_name: Filter by task name (results containing this task).
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            limit: Maximum number of results to return.

        Returns:
            List of matching evaluation results.
        """
        ...

    @abstractmethod
    def delete(self, experiment_id: str) -> bool:
        """Delete an evaluation result.

        Args:
            experiment_id: The unique identifier of the result to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...


def convert_runner_results(
    results: dict[str, Any],
    experiment_id: str,
    s3_location: str | None = None,
    experiment_name: str | None = None,
    workspace: str | None = None,
    author: str | None = None,
    git_ref: str | None = None,
    model_hash: str | None = None,
    revision: str | None = None,
    tags: list[str] | None = None,
    model_path: str | None = None,
    experiment_group: str | None = None,
    experiment_duration_seconds: float | None = None,
    provider_init_seconds: dict[str, float] | None = None,
) -> EvalResult:
    """Convert EvalRunner results dict to EvalResult.

    Args:
        results: The results dict from EvalRunner.run()
        experiment_id: Unique identifier for this run.
        s3_location: Base S3 path where task results are stored.
        experiment_name: Descriptive name for the experiment.
        workspace: Beaker workspace name.
        author: Who ran the evaluation.
        git_ref: Git commit/ref for reproducibility.
        model_hash: Hash of model configuration.
        revision: Model revision/checkpoint.
        tags: List of tags for categorization.
        model_path: Original model path (when alias is used).
        experiment_group: Group for organizing related experiments.

    Returns:
        EvalResult instance.
    """
    tasks = []
    for task_idx, (spec, task_data) in enumerate(results.get("tasks", {}).items()):
        # Build S3 keys for this task if s3_location is provided
        s3_metrics_key = None
        s3_predictions_key = None
        s3_requests_key = None
        if s3_location:
            from olmo_eval.runners.utils import (
                PREDICTIONS_SUFFIX,
                REQUESTS_SUFFIX,
                sanitize_spec_for_filename,
            )

            base = s3_location.rstrip("/")
            sanitized_spec = sanitize_spec_for_filename(spec)
            s3_metrics_key = f"{base}/task-{task_idx:03d}-{sanitized_spec}-metrics.json"
            s3_predictions_key = f"{base}/task-{task_idx:03d}-{sanitized_spec}{PREDICTIONS_SUFFIX}"
            s3_requests_key = f"{base}/task-{task_idx:03d}-{sanitized_spec}{REQUESTS_SUFFIX}"

        # Extract primary metric info
        metrics = task_data.get("metrics", {})
        primary_metric = task_data.get("primary_metric")
        primary_score = metrics.get(primary_metric) if primary_metric else None

        # task_hash is required
        task_hash = task_data.get("task_hash")
        if not task_hash:
            raise ValueError(f"task_hash is required for task '{spec}'")

        # Get task config if available
        task_config = task_data.get("config")

        tasks.append(
            StoredTaskResult(
                task_name=spec,
                metrics=metrics,
                task_hash=task_hash,
                task_config=task_config,
                num_instances=task_data.get("num_instances"),
                primary_metric=primary_metric,
                primary_score=primary_score,
                s3_metrics_key=s3_metrics_key,
                s3_predictions_key=s3_predictions_key,
                s3_requests_key=s3_requests_key,
                duration_seconds=task_data.get("duration_seconds"),
            )
        )

    return EvalResult(
        experiment_id=experiment_id,
        model_name=results["model"],
        backend_name=results["provider"],
        timestamp=datetime.fromisoformat(results["timestamp"]),
        tasks=tasks,
        experiment_name=experiment_name,
        workspace=workspace,
        author=author,
        tags=tags,
        git_ref=git_ref,
        model_hash=model_hash,
        revision=revision,
        s3_location=s3_location,
        model_config=results.get("model_config"),
        metadata=results.get("metadata"),
        model_path=model_path,
        experiment_group=experiment_group,
        experiment_duration_seconds=experiment_duration_seconds,
        provider_init_seconds=provider_init_seconds,
    )

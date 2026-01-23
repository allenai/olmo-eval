"""Base classes and data models for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TaskResult:
    """Result for a single task within an evaluation.

    Stores task-level metrics and references to S3 locations where
    detailed predictions and metrics files are stored.
    """

    task_name: str
    metrics: dict[str, float]
    num_instances: int | None = None
    task_hash: str | None = None
    primary_metric: str | None = None
    primary_score: float | None = None
    # S3 references for detailed data
    s3_metrics_key: str | None = None
    s3_predictions_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "task_name": self.task_name,
            "metrics": self.metrics,
        }
        if self.num_instances is not None:
            result["num_instances"] = self.num_instances
        if self.task_hash is not None:
            result["task_hash"] = self.task_hash
        if self.primary_metric is not None:
            result["primary_metric"] = self.primary_metric
        if self.primary_score is not None:
            result["primary_score"] = self.primary_score
        if self.s3_metrics_key is not None:
            result["s3_metrics_key"] = self.s3_metrics_key
        if self.s3_predictions_key is not None:
            result["s3_predictions_key"] = self.s3_predictions_key
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        """Create from dictionary."""
        return cls(
            task_name=data["task_name"],
            metrics=data["metrics"],
            num_instances=data.get("num_instances"),
            task_hash=data.get("task_hash"),
            primary_metric=data.get("primary_metric"),
            primary_score=data.get("primary_score"),
            s3_metrics_key=data.get("s3_metrics_key"),
            s3_predictions_key=data.get("s3_predictions_key"),
        )


@dataclass
class EvalResult:
    """Complete result for an evaluation run.

    Stores run-level metadata and references to S3 locations where
    the full evaluation data (completions, metrics, predictions) is stored.

    Fields align with the evaluation tracking schema:
    - Core identifiers: run_id, model_name, backend_name
    - Experiment info: experiment_name, workspace, author, tags
    - Version tracking: git_ref, model_hash, revision
    - S3 reference: s3_location points to base path with all task results
    """

    run_id: str
    model_name: str
    backend_name: str
    timestamp: datetime
    tasks: list[TaskResult] = field(default_factory=list)
    # Experiment metadata
    experiment_name: str | None = None
    workspace: str | None = None
    author: str | None = None
    tags: list[str] | None = None
    # Version tracking
    git_ref: str | None = None
    model_hash: str | None = None
    revision: str | None = None
    # S3 reference - base path where all task results are stored
    s3_location: str | None = None
    # Flexible config and metadata
    config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "backend_name": self.backend_name,
            "timestamp": self.timestamp.isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
        }
        if self.experiment_name is not None:
            result["experiment_name"] = self.experiment_name
        if self.workspace is not None:
            result["workspace"] = self.workspace
        if self.author is not None:
            result["author"] = self.author
        if self.tags is not None:
            result["tags"] = self.tags
        if self.git_ref is not None:
            result["git_ref"] = self.git_ref
        if self.model_hash is not None:
            result["model_hash"] = self.model_hash
        if self.revision is not None:
            result["revision"] = self.revision
        if self.s3_location is not None:
            result["s3_location"] = self.s3_location
        if self.config is not None:
            result["config"] = self.config
        if self.metadata is not None:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        """Create from dictionary."""
        return cls(
            run_id=data["run_id"],
            model_name=data["model_name"],
            backend_name=data["backend_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            tasks=[TaskResult.from_dict(t) for t in data.get("tasks", [])],
            experiment_name=data.get("experiment_name"),
            workspace=data.get("workspace"),
            author=data.get("author"),
            tags=data.get("tags"),
            git_ref=data.get("git_ref"),
            model_hash=data.get("model_hash"),
            revision=data.get("revision"),
            s3_location=data.get("s3_location"),
            config=data.get("config"),
            metadata=data.get("metadata"),
        )


class StorageBackend(ABC):
    """Abstract base class for result storage backends."""

    @abstractmethod
    def save(self, result: EvalResult) -> str:
        """Save an evaluation result.

        Args:
            result: The evaluation result to save.

        Returns:
            The run_id of the saved result.
        """
        ...

    @abstractmethod
    def get(self, run_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by run_id.

        Args:
            run_id: The unique identifier of the result.

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
    def delete(self, run_id: str) -> bool:
        """Delete an evaluation result.

        Args:
            run_id: The unique identifier of the result to delete.

        Returns:
            True if deleted, False if not found.
        """
        ...


def convert_runner_results(
    results: dict[str, Any],
    run_id: str,
    s3_location: str | None = None,
    experiment_name: str | None = None,
    workspace: str | None = None,
    author: str | None = None,
    git_ref: str | None = None,
    model_hash: str | None = None,
    revision: str | None = None,
    tags: list[str] | None = None,
) -> EvalResult:
    """Convert EvalRunner results dict to EvalResult.

    Args:
        results: The results dict from EvalRunner.run()
        run_id: Unique identifier for this run.
        s3_location: Base S3 path where task results are stored.
        experiment_name: Descriptive name for the experiment.
        workspace: Beaker workspace name.
        author: Who ran the evaluation.
        git_ref: Git commit/ref for reproducibility.
        model_hash: Hash of model configuration.
        revision: Model revision/checkpoint.
        tags: List of tags for categorization.

    Returns:
        EvalResult instance.
    """
    tasks = []
    for task_idx, (spec, task_data) in enumerate(results.get("tasks", {}).items()):
        # Build S3 keys for this task if s3_location is provided
        s3_metrics_key = None
        s3_predictions_key = None
        if s3_location:
            from olmo_eval.runners.utils import PREDICTIONS_SUFFIX, sanitize_spec_for_filename

            base = s3_location.rstrip("/")
            sanitized_spec = sanitize_spec_for_filename(spec)
            s3_metrics_key = f"{base}/task-{task_idx:03d}-{sanitized_spec}-metrics.json"
            s3_predictions_key = f"{base}/task-{task_idx:03d}-{sanitized_spec}{PREDICTIONS_SUFFIX}"

        # Extract primary metric info
        metrics = task_data.get("metrics", {})
        primary_metric = task_data.get("primary_metric")
        primary_score = metrics.get(primary_metric) if primary_metric else None

        tasks.append(
            TaskResult(
                task_name=spec,
                metrics=metrics,
                num_instances=task_data.get("num_instances"),
                task_hash=task_data.get("task_hash"),
                primary_metric=primary_metric,
                primary_score=primary_score,
                s3_metrics_key=s3_metrics_key,
                s3_predictions_key=s3_predictions_key,
            )
        )

    return EvalResult(
        run_id=run_id,
        model_name=results["model"],
        backend_name=results["backend"],
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
        config=results.get("config"),
        metadata=results.get("metadata"),
    )

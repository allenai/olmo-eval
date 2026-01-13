"""Base classes and data models for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class TaskResult:
    """Result for a single task within an evaluation."""

    task_name: str
    metrics: dict[str, float]
    num_samples: int | None = None
    subset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "task_name": self.task_name,
            "metrics": self.metrics,
        }
        if self.num_samples is not None:
            result["num_samples"] = self.num_samples
        if self.subset is not None:
            result["subset"] = self.subset
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        """Create from dictionary."""
        return cls(
            task_name=data["task_name"],
            metrics=data["metrics"],
            num_samples=data.get("num_samples"),
            subset=data.get("subset"),
        )


@dataclass
class EvalResult:
    """Complete result for an evaluation run."""

    run_id: str
    model_name: str
    backend_name: str
    timestamp: datetime
    tasks: list[TaskResult] = field(default_factory=list)
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

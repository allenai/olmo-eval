"""Repository layer for database operations.

Encapsulates data access logic for Experiment, TaskResult, and InstancePrediction entities,
providing a clean separation between business logic and database operations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from olmo_eval.core.types import EvalResult, StoredTaskResult
from olmo_eval.storage.base import compute_model_hash
from olmo_eval.storage.db.models import Experiment, InstancePrediction, TaskResult


class ExperimentRepository:
    """Repository for Experiment database operations."""

    def __init__(self, session: Session):
        """Initialize repository with a database session.

        Args:
            session: Active SQLAlchemy session.
        """
        self.session = session

    def save(self, eval_result: EvalResult) -> str:
        """Save a new evaluation experiment.

        Args:
            eval_result: EvalResult dataclass containing experiment data.

        Returns:
            The experiment_id of the saved evaluation.
        """
        experiment_id = eval_result.experiment_id

        # Check if experiment already exists
        existing = self.session.get(Experiment, experiment_id)

        model_hash = compute_model_hash(eval_result.config)

        if existing:
            # Update existing experiment
            existing.model_name = eval_result.model_name
            existing.model_hash = model_hash
            existing.backend_name = eval_result.backend_name
            existing.timestamp = eval_result.timestamp
            existing.experiment_name = eval_result.experiment_name
            existing.workspace = eval_result.workspace
            existing.author = eval_result.author
            existing.tags = eval_result.tags
            existing.git_ref = eval_result.git_ref
            existing.revision = eval_result.revision
            existing.s3_location = eval_result.s3_location
            existing.config = eval_result.config
            existing.metadata_ = eval_result.metadata

            # Delete existing task results and instance predictions (will be replaced)
            self.session.execute(
                delete(TaskResult).where(TaskResult.experiment_id == experiment_id)
            )
            self.session.execute(
                delete(InstancePrediction).where(InstancePrediction.experiment_id == experiment_id)
            )
            experiment = existing
        else:
            # Create new experiment
            experiment = Experiment(
                experiment_id=experiment_id,
                model_name=eval_result.model_name,
                model_hash=model_hash,
                backend_name=eval_result.backend_name,
                timestamp=eval_result.timestamp,
                experiment_name=eval_result.experiment_name,
                workspace=eval_result.workspace,
                author=eval_result.author,
                tags=eval_result.tags,
                git_ref=eval_result.git_ref,
                revision=eval_result.revision,
                s3_location=eval_result.s3_location,
                config=eval_result.config,
                metadata_=eval_result.metadata,
            )
            self.session.add(experiment)

        # Add task results
        for task_data in eval_result.tasks:
            task_result = TaskResult(
                experiment_id=experiment_id,
                task_name=task_data.task_name,
                task_hash=task_data.task_hash,
                metrics=task_data.metrics,
                num_instances=task_data.num_instances,
                primary_metric=task_data.primary_metric,
                primary_score=task_data.primary_score,
                s3_metrics_key=task_data.s3_metrics_key,
                s3_predictions_key=task_data.s3_predictions_key,
            )
            self.session.add(task_result)

        self.session.flush()  # Ensure database constraints are checked
        return experiment_id

    def get(self, experiment_id: str) -> EvalResult | None:
        """Retrieve an evaluation experiment by ID.

        Args:
            experiment_id: Unique identifier of the evaluation experiment.

        Returns:
            EvalResult if found, None otherwise.
        """
        experiment = self.session.get(Experiment, experiment_id)
        if not experiment:
            return None

        return self._to_eval_result(experiment)

    def delete(self, experiment_id: str) -> bool:
        """Delete an evaluation experiment and its task results and instance predictions.

        Args:
            experiment_id: Unique identifier of the evaluation experiment.

        Returns:
            True if deleted, False if not found.
        """
        result = self.session.execute(
            delete(Experiment).where(Experiment.experiment_id == experiment_id)
        )
        return result.rowcount > 0  # type: ignore[union-attr]

    def query(
        self,
        model_name: str | None = None,
        model_hash: str | None = None,
        task_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EvalResult]:
        """Query evaluation experiments with filters.

        Args:
            model_name: Filter by model name (exact match).
            model_hash: Filter by model hash (hash of model config).
            task_name: Filter by task name (experiments containing this task).
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).

        Returns:
            List of matching EvalResult objects.
        """
        stmt = select(Experiment)

        # Apply filters
        if model_name:
            stmt = stmt.where(Experiment.model_name == model_name)

        if model_hash:
            stmt = stmt.where(Experiment.model_hash == model_hash)

        if start_time:
            stmt = stmt.where(Experiment.timestamp >= start_time)

        if end_time:
            stmt = stmt.where(Experiment.timestamp <= end_time)

        if task_name:
            # Subquery to find experiment_ids that have this task
            from sqlalchemy import exists

            stmt = stmt.where(
                exists()
                .where(TaskResult.experiment_id == Experiment.experiment_id)
                .where(TaskResult.task_name == task_name)
            )

        # Order by timestamp descending (most recent first)
        stmt = stmt.order_by(Experiment.timestamp.desc())

        # Apply pagination
        stmt = stmt.limit(limit).offset(offset)

        # Execute query
        experiments = self.session.execute(stmt).scalars().all()

        return [self._to_eval_result(exp) for exp in experiments]

    @staticmethod
    def _to_eval_result(experiment: Experiment) -> EvalResult:
        """Convert ORM model to EvalResult dataclass.

        Args:
            experiment: Experiment ORM instance.

        Returns:
            EvalResult dataclass.
        """
        tasks = [
            StoredTaskResult(
                task_name=task.task_name,
                metrics=task.metrics,
                num_instances=task.num_instances,
                task_hash=task.task_hash,
                primary_metric=task.primary_metric,
                primary_score=task.primary_score,
                s3_metrics_key=task.s3_metrics_key,
                s3_predictions_key=task.s3_predictions_key,
            )
            for task in experiment.task_results
        ]

        return EvalResult(
            experiment_id=experiment.experiment_id,
            model_name=experiment.model_name,
            backend_name=experiment.backend_name,
            timestamp=experiment.timestamp,
            tasks=tasks,
            experiment_name=experiment.experiment_name,
            workspace=experiment.workspace,
            author=experiment.author,
            tags=experiment.tags,
            git_ref=experiment.git_ref,
            model_hash=experiment.model_hash,
            revision=experiment.revision,
            s3_location=experiment.s3_location,
            config=experiment.config,
            metadata=experiment.metadata_,
        )


class InstancePredictionRepository:
    """Repository for InstancePrediction database operations."""

    def __init__(self, session: Session):
        """Initialize repository with a database session.

        Args:
            session: Active SQLAlchemy session.
        """
        self.session = session

    def save_instances(
        self,
        experiment_id: str,
        task_name: str,
        instances: list[dict[str, Any]],
        model_hash: str | None = None,
    ) -> None:
        """Save instance predictions for an experiment's task.

        Args:
            experiment_id: Experiment identifier.
            task_name: Task name.
            instances: List of instance dicts with keys:
                - native_id: Original dataset ID
                - doc_id: Sequential ID
                - instance_metrics: Dict of metric names to values
                - s3_prediction_key: Optional S3 key for full prediction
            model_hash: Optional model hash (denormalized).
        """
        for inst_data in instances:
            instance = InstancePrediction(
                experiment_id=experiment_id,
                model_hash=model_hash,
                task_name=task_name,
                native_id=inst_data["native_id"],
                doc_id=inst_data["doc_id"],
                instance_metrics=inst_data["instance_metrics"],
                s3_prediction_key=inst_data.get("s3_prediction_key"),
            )
            self.session.add(instance)

    def get_instances(
        self,
        model_hash: str | None = None,
        task_name: str | list[str] | None = None,
        experiment_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get instance predictions with filters.

        Args:
            model_hash: Filter by model hash.
            task_name: Filter by task name (single string) or task names (list).
            experiment_id: Filter by specific experiment.
            limit: Optional maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of instance dicts.
        """
        stmt = select(InstancePrediction)

        if model_hash:
            stmt = stmt.where(InstancePrediction.model_hash == model_hash)

        if task_name:
            if isinstance(task_name, list):
                stmt = stmt.where(InstancePrediction.task_name.in_(task_name))
            else:
                stmt = stmt.where(InstancePrediction.task_name == task_name)

        if experiment_id:
            stmt = stmt.where(InstancePrediction.experiment_id == experiment_id)

        stmt = stmt.order_by(InstancePrediction.id)

        if limit:
            stmt = stmt.limit(limit)

        if offset:
            stmt = stmt.offset(offset)

        instances = self.session.execute(stmt).scalars().all()

        return [self._to_instance_dict(inst) for inst in instances]

    @staticmethod
    def _to_instance_dict(instance: InstancePrediction) -> dict[str, Any]:
        """Convert ORM model to dict.

        Args:
            instance: InstancePrediction ORM instance.

        Returns:
            Instance dict.
        """
        return {
            "id": instance.id,
            "experiment_id": instance.experiment_id,
            "model_hash": instance.model_hash,
            "task_name": instance.task_name,
            "native_id": instance.native_id,
            "doc_id": instance.doc_id,
            "instance_metrics": instance.instance_metrics,
            "s3_prediction_key": instance.s3_prediction_key,
        }

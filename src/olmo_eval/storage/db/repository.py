"""Repository layer for database operations.

Encapsulates data access logic for Experiment, TaskResult, and InstancePrediction entities,
providing a clean separation between business logic and database operations.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from olmo_eval.core.types import EvalResult, StoredTaskResult
from olmo_eval.storage.db.models import Experiment, InstancePrediction, TaskResult


class ExperimentRepository:
    """Repository for Experiment database operations."""

    def __init__(self, session: Session):
        """Initialize repository with a database session.

        Args:
            session: Active SQLAlchemy session.
        """
        self.session = session

    def save(self, eval_result: EvalResult) -> int:
        """Save a new evaluation experiment.

        Args:
            eval_result: EvalResult dataclass containing experiment data.

        Returns:
            The auto-increment id (PK) of the saved experiment.
        """
        experiment = Experiment(
            experiment_id=eval_result.experiment_id,
            model_name=eval_result.model_name,
            model_hash=eval_result.model_hash,
            model_config=eval_result.model_config,
            backend_name=eval_result.backend_name,
            timestamp=eval_result.timestamp,
            experiment_name=eval_result.experiment_name,
            workspace=eval_result.workspace,
            author=eval_result.author,
            tags=eval_result.tags,
            git_ref=eval_result.git_ref,
            revision=eval_result.revision,
            s3_location=eval_result.s3_location,
            model_path=eval_result.model_path,
            metadata_=eval_result.metadata,
        )
        self.session.add(experiment)
        self.session.flush()  # Get the auto-generated id

        # Add task results using experiment.id as FK
        for task_data in eval_result.tasks:
            task_result = TaskResult(
                experiment_pk=experiment.id,
                model_hash=eval_result.model_hash,
                task_name=task_data.task_name,
                task_hash=task_data.task_hash,
                task_config=task_data.task_config,
                metrics=task_data.metrics,
                num_instances=task_data.num_instances,
                primary_metric=task_data.primary_metric,
                primary_score=task_data.primary_score,
                s3_metrics_key=task_data.s3_metrics_key,
                s3_predictions_key=task_data.s3_predictions_key,
                s3_requests_key=task_data.s3_requests_key,
            )
            self.session.add(task_result)

        self.session.flush()
        return experiment.id

    def get(self, experiment_pk: int) -> EvalResult | None:
        """Retrieve an evaluation experiment by its primary key (id).

        Args:
            experiment_pk: Auto-increment primary key of the experiment.

        Returns:
            EvalResult if found, None otherwise.
        """
        experiment = self.session.get(Experiment, experiment_pk)
        if not experiment:
            return None

        return self._to_eval_result(experiment)

    def get_by_experiment_id(self, experiment_id: str) -> list[EvalResult]:
        """Retrieve all experiments with a given experiment_id.

        Note: Multiple experiments can share the same experiment_id when
        running multiple models in a single launch.

        Args:
            experiment_id: Experiment ID.

        Returns:
            List of EvalResult objects (may be empty, one, or many).
        """
        stmt = select(Experiment).where(Experiment.experiment_id == experiment_id)
        experiments = self.session.execute(stmt).scalars().all()
        return [self._to_eval_result(exp) for exp in experiments]

    def delete(self, experiment_pk: int) -> bool:
        """Delete an evaluation experiment and its task results and instance predictions.

        Args:
            experiment_pk: Auto-increment primary key of the experiment.

        Returns:
            True if deleted, False if not found.
        """
        result = self.session.execute(delete(Experiment).where(Experiment.id == experiment_pk))
        return result.rowcount > 0  # type: ignore[union-attr]

    def delete_by_experiment_id(self, experiment_id: str) -> int:
        """Delete all experiments with a given experiment_id.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Number of experiments deleted.
        """
        result = self.session.execute(
            delete(Experiment).where(Experiment.experiment_id == experiment_id)
        )
        return result.rowcount  # type: ignore[union-attr]

    def query(
        self,
        model_name: str | None = None,
        model_hash: str | None = None,
        task_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        latest: bool = False,
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
            latest: If True, return only the most recent result (limit=1).
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
            # Subquery to find experiment ids that have this task
            from sqlalchemy import exists

            stmt = stmt.where(
                exists()
                .where(TaskResult.experiment_pk == Experiment.id)
                .where(TaskResult.task_name == task_name)
            )

        # Order by timestamp descending (most recent first)
        stmt = stmt.order_by(Experiment.timestamp.desc())

        # Apply pagination (latest overrides limit)
        effective_limit = 1 if latest else limit
        stmt = stmt.limit(effective_limit)
        if not latest:
            stmt = stmt.offset(offset)

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
                task_hash=task.task_hash,
                task_config=task.task_config,
                num_instances=task.num_instances,
                primary_metric=task.primary_metric,
                primary_score=task.primary_score,
                s3_metrics_key=task.s3_metrics_key,
                s3_predictions_key=task.s3_predictions_key,
                s3_requests_key=task.s3_requests_key,
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
            model_config=experiment.model_config,
            metadata=experiment.metadata_,
            model_path=experiment.model_path,
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
        experiment_pk: int,
        task_hash: str,
        instances: list[dict[str, Any]],
    ) -> None:
        """Save instance predictions for an experiment's task.

        Args:
            experiment_pk: Experiment primary key (id).
            task_hash: Task configuration hash.
            instances: List of instance dicts with keys:
                - native_id: Original dataset ID
                - instance_metrics: Dict of metric names to values
        """
        for inst_data in instances:
            instance = InstancePrediction(
                experiment_pk=experiment_pk,
                task_hash=task_hash,
                native_id=inst_data["native_id"],
                instance_metrics=inst_data["instance_metrics"],
            )
            self.session.add(instance)

    def get_instances(
        self,
        experiment_pk: int | None = None,
        task_hash: str | None = None,
        task_name: str | list[str] | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions with filters.

        Args:
            experiment_pk: Filter by experiment primary key.
            task_hash: Filter by task hash.
            task_name: Filter by task name (requires JOIN to task_results).
            limit: Optional maximum number of results.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with 'id' field for pagination.
        """
        stmt = select(InstancePrediction)

        # Keyset pagination - much faster than OFFSET for large datasets
        if after_id is not None:
            stmt = stmt.where(InstancePrediction.id > after_id)

        if experiment_pk:
            stmt = stmt.where(InstancePrediction.experiment_pk == experiment_pk)

        if task_hash:
            stmt = stmt.where(InstancePrediction.task_hash == task_hash)

        if task_name:
            # JOIN to task_results to filter by task_name
            stmt = stmt.join(
                TaskResult,
                (TaskResult.experiment_pk == InstancePrediction.experiment_pk)
                & (TaskResult.task_hash == InstancePrediction.task_hash),
            )
            if isinstance(task_name, list):
                stmt = stmt.where(TaskResult.task_name.in_(task_name))
            else:
                stmt = stmt.where(TaskResult.task_name == task_name)

        stmt = stmt.order_by(InstancePrediction.id)

        if limit:
            stmt = stmt.limit(limit)

        instances = self.session.execute(stmt).scalars().all()

        return [self._to_instance_dict(inst) for inst in instances]

    def get_instances_by_experiment_id(
        self,
        experiment_id: str,
        task_name: str | list[str] | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions by experiment_id (string).

        Convenience method for CLI that works with experiment_id strings.

        Args:
            experiment_id: Filter by experiment_id (string).
            task_name: Filter by task name.
            limit: Optional maximum number of results.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with enriched data (task_name included).
        """
        from olmo_eval.storage.db.models import Experiment

        # Build query with JOINs to get task_name
        stmt = (
            select(InstancePrediction, TaskResult.task_name)
            .join(Experiment, Experiment.id == InstancePrediction.experiment_pk)
            .join(
                TaskResult,
                (TaskResult.experiment_pk == InstancePrediction.experiment_pk)
                & (TaskResult.task_hash == InstancePrediction.task_hash),
            )
            .where(Experiment.experiment_id == experiment_id)
        )

        # Keyset pagination - much faster than OFFSET for large datasets
        if after_id is not None:
            stmt = stmt.where(InstancePrediction.id > after_id)

        if task_name:
            if isinstance(task_name, list):
                stmt = stmt.where(TaskResult.task_name.in_(task_name))
            else:
                stmt = stmt.where(TaskResult.task_name == task_name)

        stmt = stmt.order_by(InstancePrediction.id)

        if limit:
            stmt = stmt.limit(limit)

        results = self.session.execute(stmt).all()

        return [
            {
                "id": inst.id,
                "task_hash": inst.task_hash,
                "task_name": task_name_val,
                "native_id": inst.native_id,
                "instance_metrics": inst.instance_metrics,
            }
            for inst, task_name_val in results
        ]

    def get_instances_by_model(
        self,
        model_name: str | None = None,
        model_hash: str | None = None,
        task_name: str | list[str] | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions by model name or hash.

        Convenience method for CLI that works with model identifiers.

        Args:
            model_name: Filter by model name.
            model_hash: Filter by model hash.
            task_name: Filter by task name.
            limit: Optional maximum number of results.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with enriched data (task_name, model_hash included).
        """
        from olmo_eval.storage.db.models import Experiment

        if not model_name and not model_hash:
            raise ValueError("Either model_name or model_hash is required")

        # Build query with JOINs to get task_name and model_hash
        stmt = (
            select(InstancePrediction, TaskResult.task_name, Experiment.model_hash)
            .join(Experiment, Experiment.id == InstancePrediction.experiment_pk)
            .join(
                TaskResult,
                (TaskResult.experiment_pk == InstancePrediction.experiment_pk)
                & (TaskResult.task_hash == InstancePrediction.task_hash),
            )
        )

        # Keyset pagination - much faster than OFFSET for large datasets
        if after_id is not None:
            stmt = stmt.where(InstancePrediction.id > after_id)

        if model_name:
            stmt = stmt.where(Experiment.model_name == model_name)

        if model_hash:
            stmt = stmt.where(Experiment.model_hash == model_hash)

        if task_name:
            if isinstance(task_name, list):
                stmt = stmt.where(TaskResult.task_name.in_(task_name))
            else:
                stmt = stmt.where(TaskResult.task_name == task_name)

        stmt = stmt.order_by(InstancePrediction.id)

        if limit:
            stmt = stmt.limit(limit)

        results = self.session.execute(stmt).all()

        return [
            {
                "id": inst.id,
                "task_hash": inst.task_hash,
                "task_name": task_name_val,
                "model_hash": model_hash_val,
                "native_id": inst.native_id,
                "instance_metrics": inst.instance_metrics,
            }
            for inst, task_name_val, model_hash_val in results
        ]

    def get_instances_by_task(
        self,
        task_name: str | list[str] | None = None,
        task_hash: str | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions by task name or hash.

        Args:
            task_name: Filter by task name(s).
            task_hash: Filter by task hash (exact match).
            limit: Optional maximum number of results.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with task_name included.
        """
        if not task_name and not task_hash:
            raise ValueError("Either task_name or task_hash is required")

        # Build query with JOIN to get task_name
        stmt = select(InstancePrediction, TaskResult.task_name).join(
            TaskResult,
            (TaskResult.experiment_pk == InstancePrediction.experiment_pk)
            & (TaskResult.task_hash == InstancePrediction.task_hash),
        )

        # Keyset pagination - much faster than OFFSET for large datasets
        if after_id is not None:
            stmt = stmt.where(InstancePrediction.id > after_id)

        if task_hash:
            stmt = stmt.where(InstancePrediction.task_hash == task_hash)

        if task_name:
            if isinstance(task_name, list):
                stmt = stmt.where(TaskResult.task_name.in_(task_name))
            else:
                stmt = stmt.where(TaskResult.task_name == task_name)

        stmt = stmt.order_by(InstancePrediction.id)

        if limit:
            stmt = stmt.limit(limit)

        results = self.session.execute(stmt).all()

        return [
            {
                "id": inst.id,
                "task_hash": inst.task_hash,
                "task_name": task_name_val,
                "native_id": inst.native_id,
                "instance_metrics": inst.instance_metrics,
            }
            for inst, task_name_val in results
        ]

    @staticmethod
    def _to_instance_dict(instance: InstancePrediction) -> dict[str, Any]:
        """Convert ORM model to dict.

        Args:
            instance: InstancePrediction ORM instance.

        Returns:
            Instance dict with id for pagination.
        """
        return {
            "id": instance.id,
            "task_hash": instance.task_hash,
            "native_id": instance.native_id,
            "instance_metrics": instance.instance_metrics,
        }

    def stream_instances(
        self,
        experiment_pk: int,
        task_hash: str | None = None,
        batch_size: int = 1000,
    ) -> Iterator[dict[str, Any]]:
        """Stream instances in batches for memory efficiency.

        Uses SQLAlchemy's yield_per() to avoid loading all rows into memory.
        Useful for exporting large datasets.

        Args:
            experiment_pk: Experiment primary key (required for streaming).
            task_hash: Optional task hash filter.
            batch_size: Number of rows to fetch per batch.

        Yields:
            Instance dicts one at a time.
        """
        stmt = select(InstancePrediction).where(InstancePrediction.experiment_pk == experiment_pk)

        if task_hash:
            stmt = stmt.where(InstancePrediction.task_hash == task_hash)

        stmt = stmt.order_by(InstancePrediction.id)

        for instance in self.session.execute(stmt).scalars().yield_per(batch_size):
            yield self._to_instance_dict(instance)

"""Query helpers for common evaluation query patterns."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from olmo_eval.core.types import EvalResult
from olmo_eval.storage.backends.postgres.repository import (
    ExperimentRepository,
    InstancePredictionRepository,
)


class QueryHelper:
    """Helper class for common query patterns."""

    def __init__(self, session: Session):
        """Initialize with database session.

        Args:
            session: Active SQLAlchemy session.
        """
        self.session = session
        self.experiment_repo = ExperimentRepository(session)
        self.instance_repo = InstancePredictionRepository(session)

    def save(self, result: EvalResult) -> int:
        """Save an evaluation result.

        Args:
            result: EvalResult dataclass containing run data.

        Returns:
            The auto-increment id (PK) of the saved experiment.
        """
        return self.experiment_repo.save(result)

    def save_with_instances(
        self,
        result: EvalResult,
        instances_by_task: dict[str, list[dict[str, Any]]],
    ) -> int:
        """Save an evaluation result with instance-level predictions.

        Args:
            result: EvalResult dataclass containing run data.
            instances_by_task: Dict mapping task_name -> list of instance dicts.
                Each instance dict should have:
                - native_id: Original dataset ID
                - instance_metrics: Dict of metric names to values

        Returns:
            The auto-increment id (PK) of the saved experiment.
        """
        # Save experiment and task results, get the experiment PK
        experiment_pk = self.experiment_repo.save(result)

        # Get the experiment_group for denormalization - must never be empty
        experiment_group = result.experiment_group or result.experiment_name or result.experiment_id

        # Build task_hash lookup from result.tasks
        task_hash_lookup: dict[str, str] = {}
        for task in result.tasks:
            if task.task_hash:
                task_hash_lookup[task.task_name] = task.task_hash

        for task_name, instances in instances_by_task.items():
            task_hash = task_hash_lookup.get(task_name)
            if not task_hash:
                raise ValueError(
                    f"task_hash is required for task '{task_name}' instance predictions"
                )
            self.instance_repo.save_instances(
                experiment_pk=experiment_pk,
                task_hash=task_hash,
                instances=instances,
                experiment_group=experiment_group,
            )

        return experiment_pk

    def get(self, experiment_pk: int) -> EvalResult | None:
        """Retrieve an evaluation result by experiment primary key.

        Args:
            experiment_pk: The auto-increment primary key of the experiment.

        Returns:
            EvalResult if found, None otherwise.
        """
        return self.experiment_repo.get(experiment_pk)

    def get_by_experiment_id(self, experiment_id: str) -> list[EvalResult]:
        """Retrieve all experiments with a given experiment_id.

        Note: Multiple experiments can share the same experiment_id when
        running multiple models in a single launch.

        Args:
            experiment_id: Experiment ID.

        Returns:
            List of EvalResult objects (may be empty, one, or many).
        """
        return self.experiment_repo.get_by_experiment_id(experiment_id)

    def query(
        self,
        model_name: str | None = None,
        model_hash: str | None = None,
        task_name: str | None = None,
        experiment_group: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        latest: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[EvalResult]:
        """Query evaluation results by filters.

        Args:
            model_name: Filter by model name.
            model_hash: Filter by model hash.
            task_name: Filter by task name (results containing this task).
            experiment_group: Filter by experiment group.
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            latest: If True, return only the most recent result.
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).

        Returns:
            List of matching evaluation results.
        """
        return self.experiment_repo.query(
            model_name=model_name,
            model_hash=model_hash,
            task_name=task_name,
            experiment_group=experiment_group,
            start_time=start_time,
            end_time=end_time,
            latest=latest,
            limit=limit,
            offset=offset,
        )

    def delete(self, experiment_pk: int) -> bool:
        """Delete an evaluation result by primary key.

        Args:
            experiment_pk: The auto-increment primary key of the experiment.

        Returns:
            True if deleted, False if not found.
        """
        return self.experiment_repo.delete(experiment_pk)

    def delete_by_experiment_id(self, experiment_id: str) -> int:
        """Delete all experiments with a given experiment_id.

        Args:
            experiment_id: Experiment ID.

        Returns:
            Number of experiments deleted.
        """
        return self.experiment_repo.delete_by_experiment_id(experiment_id)

    def get_model_task_metrics(
        self,
        model_name: str | None = None,
        model_hash: str | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, float | None]:
        """Get task metrics for a model.

        Args:
            model_name: Model name filter.
            model_hash: Model hash filter.
            tasks: Optional list of tasks to include.

        Returns:
            Dict mapping task_name -> primary_score.
        """
        experiments = self.experiment_repo.query(
            model_name=model_name,
            model_hash=model_hash,
            latest=True,
        )

        if not experiments:
            return {}

        exp = experiments[0]
        results = {}

        for task in exp.tasks:
            if tasks and task.task_name not in tasks:
                continue
            results[task.task_name] = task.primary_score

        return results

    def get_model_task_instances(
        self,
        task_name: str | list[str],
        experiment_pk: int | None = None,
        task_hash: str | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions for a task.

        Args:
            task_name: Task name (single string) or task names (list) to query.
            experiment_pk: Specific experiment PK to filter by.
            task_hash: Task hash to filter by.
            limit: Optional maximum number of instances.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with metrics and metadata.
        """
        return self.instance_repo.get_instances(
            experiment_pk=experiment_pk,
            task_hash=task_hash,
            task_name=task_name,
            limit=limit,
            after_id=after_id,
        )

    def get_instances_by_experiment_id(
        self,
        experiment_id: str,
        task_name: str | list[str] | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions by experiment_id (string).

        Args:
            experiment_id: Experiment ID (string) to filter by.
            task_name: Optional task name filter.
            limit: Optional maximum number of instances.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with task_name included.
        """
        return self.instance_repo.get_instances_by_experiment_id(
            experiment_id=experiment_id,
            task_name=task_name,
            limit=limit,
            after_id=after_id,
        )

    def get_instances_by_model(
        self,
        task_name: str | list[str],
        model_name: str | None = None,
        model_hash: str | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions by model name or hash.

        Args:
            task_name: Task name (required).
            model_name: Model name to filter by.
            model_hash: Model hash to filter by.
            limit: Optional maximum number of instances.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with task_name and model_hash included.
        """
        return self.instance_repo.get_instances_by_model(
            model_name=model_name,
            model_hash=model_hash,
            task_name=task_name,
            limit=limit,
            after_id=after_id,
        )

    def get_instances_by_task(
        self,
        task_name: str | list[str] | None = None,
        task_hash: str | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Get instance predictions by task name or hash.

        Args:
            task_name: Task name(s) to filter by.
            task_hash: Task hash to filter by (exact match).
            limit: Optional maximum number of instances.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with task_name included.
        """
        return self.instance_repo.get_instances_by_task(
            task_name=task_name,
            task_hash=task_hash,
            limit=limit,
            after_id=after_id,
        )

    def query_instances(
        self,
        experiment_ids: list[str] | None = None,
        model_names: list[str] | None = None,
        model_hashes: list[str] | None = None,
        task_names: list[str] | None = None,
        task_hash: str | None = None,
        limit: int | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query instances with composable filters.

        All filters are optional and compose with AND logic. This is the
        preferred method for querying instances - filters can be combined
        freely.

        Args:
            experiment_ids: Filter by experiment ID strings.
            model_names: Filter by model names.
            model_hashes: Filter by model hashes.
            task_names: Filter by task names.
            task_hash: Filter by exact task hash.
            limit: Maximum number of results.
            after_id: Return instances with id > after_id (keyset pagination).

        Returns:
            List of instance dicts with task_name and model_hash included.
        """
        return self.instance_repo.query_instances(
            experiment_ids=experiment_ids,
            model_names=model_names,
            model_hashes=model_hashes,
            task_names=task_names,
            task_hash=task_hash,
            limit=limit,
            after_id=after_id,
        )

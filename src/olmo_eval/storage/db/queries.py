"""Query helpers for common evaluation query patterns."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from olmo_eval.core.types import EvalResult, compute_model_hash
from olmo_eval.storage.db.repository import ExperimentRepository, InstancePredictionRepository


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

    def save(self, result: EvalResult) -> str:
        """Save an evaluation result.

        Args:
            result: EvalResult dataclass containing run data.

        Returns:
            The experiment_id of the saved evaluation.
        """
        return self.experiment_repo.save(result)

    def save_with_instances(
        self,
        result: EvalResult,
        instances_by_task: dict[str, list[dict[str, Any]]],
    ) -> str:
        """Save an evaluation result with instance-level predictions.

        Args:
            result: EvalResult dataclass containing run data.
            instances_by_task: Dict mapping task_name -> list of instance dicts.
                Each instance dict should have:
                - native_id: Original dataset ID
                - doc_id: Sequential ID
                - instance_metrics: Dict of metric names to values
                - s3_prediction_key: Optional S3 key

        Returns:
            The experiment_id of the saved evaluation.
        """
        # Save experiment and task results
        experiment_id = self.experiment_repo.save(result)

        # Save instance predictions
        model_hash = compute_model_hash(result.config)

        for task_name, instances in instances_by_task.items():
            self.instance_repo.save_instances(
                experiment_id=experiment_id,
                task_name=task_name,
                instances=instances,
                model_hash=model_hash,
            )

        return experiment_id

    def get(self, experiment_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by experiment_id.

        Args:
            experiment_id: The unique identifier of the result.

        Returns:
            EvalResult if found, None otherwise.
        """
        return self.experiment_repo.get(experiment_id)

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
        """Query evaluation results by filters.

        Args:
            model_name: Filter by model name.
            model_hash: Filter by model hash.
            task_name: Filter by task name (results containing this task).
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
            start_time=start_time,
            end_time=end_time,
            latest=latest,
            limit=limit,
            offset=offset,
        )

    def delete(self, experiment_id: str) -> bool:
        """Delete an evaluation result.

        Args:
            experiment_id: The unique identifier of the result to delete.

        Returns:
            True if deleted, False if not found.
        """
        return self.experiment_repo.delete(experiment_id)

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
        model_name: str | None = None,
        model_hash: str | None = None,
        experiment_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get instance predictions for a model and task(s).

        Supports querying by human-readable model_name or computed model_hash.
        If model_name is provided, finds all model_hashes for experiments with that name.

        Args:
            task_name: Task name (single string) or task names (list) to query.
            model_name: Human-readable model name (e.g., "llama3.1-8b").
            model_hash: Computed model hash (alternative to model_name).
            experiment_id: Specific experiment ID to filter by.
            limit: Optional maximum number of instances.
            offset: Number of instances to skip.

        Returns:
            List of instance dicts with metrics and metadata.

        Raises:
            ValueError: If neither model_name nor model_hash nor experiment_id is provided.
        """
        if not model_name and not model_hash and not experiment_id:
            raise ValueError(
                "Must provide at least one of: model_name, model_hash, or experiment_id"
            )

        # If model_name is provided, look up the corresponding model_hashes
        if model_name and not model_hash:
            experiments = self.experiment_repo.query(model_name=model_name, limit=1000)
            if not experiments:
                return []

            # Get unique model_hashes from experiments with this model_name
            model_hashes = [exp.model_hash for exp in experiments if exp.model_hash is not None]
            model_hashes = list(set(model_hashes))
            if not model_hashes:
                return []

            # Query instances for all matching model_hashes
            all_instances = []
            for mh in model_hashes:
                instances = self.instance_repo.get_instances(
                    model_hash=mh,
                    task_name=task_name,
                    experiment_id=experiment_id,
                    limit=limit,
                    offset=offset,
                )
                all_instances.extend(instances)

            # Apply limit/offset to combined results if needed
            if limit:
                all_instances = all_instances[:limit]

            return all_instances

        # Direct query by model_hash or experiment_id
        return self.instance_repo.get_instances(
            model_hash=model_hash,
            task_name=task_name,
            experiment_id=experiment_id,
            limit=limit,
            offset=offset,
        )

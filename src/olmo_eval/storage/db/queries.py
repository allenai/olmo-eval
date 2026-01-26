"""Query helpers for common evaluation query patterns."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

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

    def get_model_task_metrics(
        self,
        model_name: str | None = None,
        model_id: str | None = None,
        tasks: list[str] | None = None,
    ) -> dict[str, float | None]:
        """Get task metrics for a model.

        Args:
            model_name: Model name filter.
            model_id: Model ID filter.
            tasks: Optional list of tasks to include.

        Returns:
            Dict mapping task_name -> primary_score.
        """
        experiments = self.experiment_repo.query(
            model_name=model_name,
            model_id=model_id,
            limit=1,
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
        model_id: str | None = None,
        experiment_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get instance predictions for a model and task(s).

        Supports querying by human-readable model_name or computed model_id.
        If model_name is provided, finds all model_ids for experiments with that name.

        Args:
            task_name: Task name (single string) or task names (list) to query.
            model_name: Human-readable model name (e.g., "llama3.1-8b").
            model_id: Computed model ID hash (alternative to model_name).
            experiment_id: Specific experiment ID to filter by.
            limit: Optional maximum number of instances.
            offset: Number of instances to skip.

        Returns:
            List of instance dicts with metrics and metadata.

        Raises:
            ValueError: If neither model_name nor model_id nor experiment_id is provided.
        """
        if not model_name and not model_id and not experiment_id:
            raise ValueError("Must provide at least one of: model_name, model_id, or experiment_id")

        # If model_name is provided, look up the corresponding model_ids
        if model_name and not model_id:
            from olmo_eval.storage.base import compute_model_id

            experiments = self.experiment_repo.query(model_name=model_name, limit=1000)
            if not experiments:
                return []

            # Get unique model_ids from experiments with this model_name
            model_ids = list({compute_model_id(exp.config) for exp in experiments})
            model_ids = [mid for mid in model_ids if mid is not None]
            if not model_ids:
                return []

            # Query instances for all matching model_ids
            all_instances = []
            for mid in model_ids:
                instances = self.instance_repo.get_instances(
                    model_id=mid,
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

        # Direct query by model_id or experiment_id
        return self.instance_repo.get_instances(
            model_id=model_id,
            task_name=task_name,
            experiment_id=experiment_id,
            limit=limit,
            offset=offset,
        )

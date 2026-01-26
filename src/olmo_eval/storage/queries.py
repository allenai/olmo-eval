"""Query helpers for common evaluation query patterns."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from olmo_eval.storage.repository import ExperimentRepository, InstancePredictionRepository


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
        model_id: str,
        task_name: str | list[str],
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get instance predictions for a model and task(s).

        Args:
            model_id: Model ID to query.
            task_name: Task name (single string) or task names (list) to query.
            limit: Optional maximum number of instances.
            offset: Number of instances to skip.

        Returns:
            List of instance dicts with metrics and metadata.
        """
        return self.instance_repo.get_instances(
            model_id=model_id,
            task_name=task_name,
            limit=limit,
            offset=offset,
        )

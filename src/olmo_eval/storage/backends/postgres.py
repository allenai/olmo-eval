"""PostgreSQL-based storage backend for evaluation results."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from olmo_eval.core.types import EvalResult
from olmo_eval.storage.base import StorageBackend
from olmo_eval.storage.db.queries import QueryHelper
from olmo_eval.storage.db.session import DatabaseSession

logger = logging.getLogger(__name__)


class PostgresBackend(StorageBackend):
    """PostgreSQL-based storage backend for evaluation results.

    This is a thin wrapper that provides session management and implements
    the StorageBackend interface. All actual query logic lives in QueryHelper.

    The database stores queryable metadata while S3 stores the full
    evaluation data (completions, predictions, detailed metrics).
    """

    # TODO(undfined): Get these from environment/config
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "olmo_eval",
        user: str = "postgres",
        password: str = "",
        password_env: str | None = None,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False,
    ):
        """Initialize the PostgreSQL backend.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            password_env: Environment variable containing password (takes precedence).
            pool_size: Connection pool size.
            max_overflow: Maximum overflow connections.
            echo: Whether to echo SQL statements (for debugging).
        """
        self.db = DatabaseSession(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            password_env=password_env,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo,
        )

    def initialize(self) -> None:
        """Initialize the database connection.

        Must be called before any database operations (save, get, query, delete).
        """
        self.db.initialize()

    def save(
        self,
        result: EvalResult,
        instances_by_task: dict[str, list[dict[str, Any]]] | None = None,
    ) -> str:
        """Save an evaluation result to PostgreSQL.

        Args:
            result: EvalResult dataclass containing run data.
            instances_by_task: Optional dict mapping task_name -> list of instance dicts.
                Each instance dict should have native_id, doc_id, instance_metrics, etc.

        Returns:
            The experiment_id of the saved evaluation.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            if instances_by_task:
                experiment_id = helper.save_with_instances(result, instances_by_task)
            else:
                experiment_id = helper.save(result)
            logger.debug(f"Saved experiment {experiment_id}")
            return experiment_id

    def get(self, experiment_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by experiment_id.

        Args:
            experiment_id: The unique identifier of the result.

        Returns:
            EvalResult if found, None otherwise.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            return helper.get(experiment_id)

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
        with self.db.session() as session:
            helper = QueryHelper(session)
            return helper.query(
                model_name=model_name,
                task_name=task_name,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

    def delete(self, experiment_id: str) -> bool:
        """Delete an evaluation result.

        Args:
            experiment_id: The unique identifier of the result to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            return helper.delete(experiment_id)

    def dispose(self) -> None:
        """Dispose of the database engine and close all connections."""
        self.db.dispose()

"""PostgreSQL-based storage backend for evaluation results."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from olmo_eval.core.types import EvalResult
from olmo_eval.storage.backends.postgres.queries import QueryHelper
from olmo_eval.storage.backends.postgres.session import DatabaseSession
from olmo_eval.storage.base import StorageBackend

logger = logging.getLogger(__name__)


class PostgresBackend(StorageBackend):
    """PostgreSQL-based storage backend for evaluation results.

    This is a thin wrapper that provides session management and implements
    the StorageBackend interface. All actual query logic lives in QueryHelper.

    The database stores queryable metadata while S3 stores the full
    evaluation data (completions, predictions, detailed metrics).

    Note: experiment_id can be shared across multiple models in a single
    experiment launch (e.g., when running multiple models in parallel).
    Use get_all() to retrieve all experiments with a given experiment_id.
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
        sslmode: str = "require",
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
            sslmode: SSL mode for connection (require, prefer, disable, etc.).
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
            sslmode=sslmode,
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
                Each instance dict should have native_id, instance_metrics, etc.

        Returns:
            The experiment_id (Beaker ID) of the saved evaluation.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            if instances_by_task:
                helper.save_with_instances(result, instances_by_task)
            else:
                helper.save(result)
            logger.debug(f"Saved experiment {result.experiment_id}")
            return result.experiment_id

    def get(self, experiment_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by experiment_id.

        Note: If multiple experiments share the same experiment_id (e.g., from
        a single launch with multiple models), this returns the first one.
        Use get_all() to retrieve all experiments with a given experiment_id.

        Args:
            experiment_id: The experiment ID.

        Returns:
            EvalResult if found, None otherwise.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_by_experiment_id(experiment_id)
            return results[0] if results else None

    def get_all(self, experiment_id: str) -> list[EvalResult]:
        """Retrieve all experiments with a given experiment_id.

        Note: Multiple experiments can share the same experiment_id when
        running multiple models in a single launch.

        Args:
            experiment_id: The experiment ID.

        Returns:
            List of EvalResult objects (may be empty, one, or many).
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            return helper.get_by_experiment_id(experiment_id)

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
            model_name: Filter by model name prefix.
            task_name: Filter by task name prefix.
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            limit: Maximum number of results to return.

        Returns:
            List of matching evaluation results.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            return helper.query(
                model_names=[model_name] if model_name else None,
                task_names=[task_name] if task_name else None,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
            )

    def delete(self, experiment_id: str) -> bool:
        """Delete all experiments with a given experiment_id.

        Note: This deletes ALL experiments with the given experiment_id,
        including all their task results and instance predictions.

        Args:
            experiment_id: The experiment ID.

        Returns:
            True if any experiments were deleted, False if none found.
        """
        with self.db.session() as session:
            helper = QueryHelper(session)
            deleted_count = helper.delete_by_experiment_id(experiment_id)
            return deleted_count > 0

    def dispose(self) -> None:
        """Dispose of the database engine and close all connections."""
        self.db.dispose()

"""Database reporter for metrics storage."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ..core.schema import BatchMetrics, RequestMetrics

logger = logging.getLogger(__name__)


class DbReporter:
    """Store metrics in PostgreSQL database.

    Writes batch metrics to inference_samples table.

    Connection parameters are read from standard PostgreSQL environment
    variables (same as harness):
    - PGHOST: Database host (required)
    - PGPORT: Database port (default: 5432)
    - PGUSER: Database user (default: postgres)
    - PGPASSWORD: Database password

    The database name defaults to 'olmo_eval_metrics' (separate from results DB).
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str = "olmo_eval_metrics",
        user: str | None = None,
        password: str | None = None,
        password_env: str | None = None,
        sslmode: str = "require",
    ) -> None:
        # Read from standard PostgreSQL environment variables (same as harness)
        self._host = host or os.environ.get("PGHOST")
        self._port = port or int(os.environ.get("PGPORT", "5432"))
        self._database = database
        self._user = user or os.environ.get("PGUSER", "postgres")
        self._password = password or os.environ.get("PGPASSWORD", "")
        self._password_env = password_env
        self._sslmode = sslmode
        self._session: Session | None = None
        self._db_session: Any = None  # DatabaseSession instance

    @property
    def reporter_name(self) -> str:
        return "db"

    def configure(
        self,
        host: str | None = None,
        port: int | None = None,
        database: str | None = None,
        user: str | None = None,
        password: str | None = None,
        password_env: str | None = None,
        sslmode: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure the reporter.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            password_env: Environment variable containing password.
            sslmode: SSL mode for connection.
        """
        if host is not None:
            self._host = host
        if port is not None:
            self._port = port
        if database is not None:
            self._database = database
        if user is not None:
            self._user = user
        if password is not None:
            self._password = password
        if password_env is not None:
            self._password_env = password_env
        if sslmode is not None:
            self._sslmode = sslmode

    def initialize(self) -> None:
        """Initialize the database connection eagerly.

        Call this at job start to establish the connection early rather than
        waiting until the first batch is processed.
        """
        self._ensure_connection()

    def _ensure_connection(self) -> Session:
        """Ensure database connection is established."""
        if self._session is None:
            if not self._host:
                raise ValueError(
                    "Database host not configured. Set PGHOST environment variable "
                    "or pass host parameter to DbReporter."
                )

            from olmo_eval.storage.backends.postgres import DatabaseSession

            self._db_session = DatabaseSession(
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                password=self._password,
                password_env=self._password_env,
                sslmode=self._sslmode,
            )
            self._db_session.initialize()
            self._session = self._db_session.session_factory()
        return self._session

    def report_request(self, metrics: RequestMetrics) -> None:
        """Store a single request metric.

        Note: Individual request metrics are typically stored as part of
        report_batch. Use this for streaming individual metrics if needed.
        """
        # Individual request metrics are stored as part of batch
        # This method is provided for protocol compliance
        pass

    def report_batch(self, metrics: BatchMetrics) -> None:
        """Store batch metrics in the database."""
        from olmo_eval.storage.backends.postgres.metrics_models import InferenceSample

        session = self._ensure_connection()

        # Convert tags dict to list format for PostgreSQL ARRAY
        tags_list = [f"{k}:{v}" for k, v in metrics.tags.items()] if metrics.tags else None

        # Build metadata dict for flexible storage
        metadata: dict[str, Any] = {}
        if metrics.gpu_snapshots:
            metadata["gpu_summary"] = metrics._compute_gpu_summary()
            metadata["gpu_devices"] = metrics._group_gpu_snapshots()

        # Create InferenceSample record
        sample = InferenceSample(
            experiment_id=metrics.experiment_id,
            experiment_name=metrics.experiment_name,
            experiment_group=metrics.experiment_group,
            model_name=metrics.model_name,
            model_hash=metrics.model_hash,
            task_name=metrics.task_name,
            task_hash=metrics.task_hash,
            workspace=metrics.workspace,
            author=metrics.author,
            provider_kind=metrics.provider_kind,
            timestamp=metrics.timestamp or datetime.now(UTC),
            total_requests=metrics.total_requests,
            successful_requests=metrics.successful_requests,
            failed_requests=metrics.failed_requests,
            total_prompt_tokens=metrics.total_prompt_tokens,
            total_completion_tokens=metrics.total_completion_tokens,
            wall_clock_time_s=metrics.wall_clock_time_s,
            output_tokens_per_second=metrics.output_tokens_per_second,
            mean_latency_s=metrics.mean_latency_s,
            tags=tags_list,
            metadata_=metadata if metadata else None,
        )

        session.add(sample)

    def flush(self) -> None:
        """Commit pending transactions."""
        if self._session is not None:
            try:
                self._session.commit()
            except Exception as e:
                logger.error(f"Failed to commit metrics: {e}")
                self._session.rollback()
                raise

    def shutdown(self) -> None:
        """Close database connection."""
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._db_session is not None:
            self._db_session.dispose()
            self._db_session = None

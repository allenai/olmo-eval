"""PostgreSQL-based storage backend for evaluation results."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row

from olmo_eval.storage.base import EvalResult, StorageBackend, TaskResult


class PostgresBackend(StorageBackend):
    """PostgreSQL-based storage backend for evaluation results.

    Uses two tables:
        - eval_runs: Main evaluation run metadata with S3 reference
        - task_results: Individual task results with metrics and S3 keys

    The database stores queryable metadata while S3 stores the full
    evaluation data (completions, predictions, detailed metrics).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS eval_runs (
        run_id VARCHAR(64) PRIMARY KEY,
        model_name VARCHAR(255) NOT NULL,
        backend_name VARCHAR(50) NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        -- Experiment metadata
        experiment_name VARCHAR(255),
        workspace VARCHAR(100),
        author VARCHAR(100),
        tags TEXT[],
        -- Version tracking
        git_ref VARCHAR(100),
        model_hash VARCHAR(64),
        revision VARCHAR(255),
        -- S3 reference for full evaluation data
        s3_location VARCHAR(512),
        -- Flexible storage
        config JSONB,
        metadata JSONB,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS task_results (
        id SERIAL PRIMARY KEY,
        run_id VARCHAR(64) REFERENCES eval_runs(run_id) ON DELETE CASCADE,
        task_name VARCHAR(255) NOT NULL,
        task_hash VARCHAR(64),
        metrics JSONB NOT NULL,
        num_instances INTEGER,
        primary_metric VARCHAR(100),
        primary_score DOUBLE PRECISION,
        -- S3 keys for detailed task data
        s3_metrics_key VARCHAR(512),
        s3_predictions_key VARCHAR(512)
    );

    -- Indexes for common query patterns
    CREATE INDEX IF NOT EXISTS idx_eval_runs_model ON eval_runs(model_name);
    CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs(timestamp);
    CREATE INDEX IF NOT EXISTS idx_eval_runs_workspace ON eval_runs(workspace);
    CREATE INDEX IF NOT EXISTS idx_eval_runs_author ON eval_runs(author);
    CREATE INDEX IF NOT EXISTS idx_task_results_task ON task_results(task_name);
    CREATE INDEX IF NOT EXISTS idx_task_results_run ON task_results(run_id);
    CREATE INDEX IF NOT EXISTS idx_task_results_primary_score ON task_results(primary_score);
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "olmo_eval",
        user: str = "postgres",
        password: str = "",
        password_env: str | None = None,
    ):
        """Initialize the PostgreSQL backend.

        Args:
            host: Database host.
            port: Database port.
            database: Database name.
            user: Database user.
            password: Database password.
            password_env: Environment variable containing password (takes precedence).
        """
        import os

        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = os.environ.get(password_env, password) if password_env else password

        self._conninfo = (
            f"host={host} port={port} dbname={database} user={user} password={self.password}"
        )

    def _get_connection(self) -> psycopg.Connection:
        """Create a new database connection."""
        return psycopg.connect(self._conninfo, row_factory=dict_row)  # type: ignore[arg-type]

    def initialize(self) -> None:
        """Create the database schema."""
        with self._get_connection() as conn:
            conn.execute(self.SCHEMA)
            conn.commit()

    def cleanup(self) -> None:
        """Drop all tables (for testing)."""
        with self._get_connection() as conn:
            conn.execute("DROP TABLE IF EXISTS task_results CASCADE")
            conn.execute("DROP TABLE IF EXISTS eval_runs CASCADE")
            conn.commit()

    def save(self, result: EvalResult) -> str:
        """Save an evaluation result to PostgreSQL."""
        with self._get_connection() as conn:
            # Insert eval_run
            conn.execute(
                """
                INSERT INTO eval_runs
                    (run_id, model_name, backend_name, timestamp,
                     experiment_name, workspace, author, tags,
                     git_ref, model_hash, revision, s3_location,
                     config, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    model_name = EXCLUDED.model_name,
                    backend_name = EXCLUDED.backend_name,
                    timestamp = EXCLUDED.timestamp,
                    experiment_name = EXCLUDED.experiment_name,
                    workspace = EXCLUDED.workspace,
                    author = EXCLUDED.author,
                    tags = EXCLUDED.tags,
                    git_ref = EXCLUDED.git_ref,
                    model_hash = EXCLUDED.model_hash,
                    revision = EXCLUDED.revision,
                    s3_location = EXCLUDED.s3_location,
                    config = EXCLUDED.config,
                    metadata = EXCLUDED.metadata
                """,
                (
                    result.run_id,
                    result.model_name,
                    result.backend_name,
                    result.timestamp,
                    result.experiment_name,
                    result.workspace,
                    result.author,
                    result.tags,
                    result.git_ref,
                    result.model_hash,
                    result.revision,
                    result.s3_location,
                    json.dumps(result.config) if result.config else None,
                    json.dumps(result.metadata) if result.metadata else None,
                ),
            )

            # Delete existing task results for this run (for upsert behavior)
            conn.execute("DELETE FROM task_results WHERE run_id = %s", (result.run_id,))

            # Insert task results
            for task in result.tasks:
                conn.execute(
                    """
                    INSERT INTO task_results
                        (run_id, task_name, task_hash, metrics, num_instances,
                         primary_metric, primary_score, s3_metrics_key, s3_predictions_key)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result.run_id,
                        task.task_name,
                        task.task_hash,
                        json.dumps(task.metrics),
                        task.num_instances,
                        task.primary_metric,
                        task.primary_score,
                        task.s3_metrics_key,
                        task.s3_predictions_key,
                    ),
                )

            conn.commit()

        return result.run_id

    def get(self, run_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by run_id."""
        with self._get_connection() as conn:
            # Get eval_run
            row = conn.execute("SELECT * FROM eval_runs WHERE run_id = %s", (run_id,)).fetchone()

            if not row:
                return None

            # Get task results
            task_rows = conn.execute(
                "SELECT * FROM task_results WHERE run_id = %s", (run_id,)
            ).fetchall()

            tasks = [
                TaskResult(
                    task_name=tr["task_name"],  # type: ignore[index]
                    metrics=tr["metrics"],  # type: ignore[index]
                    num_instances=tr["num_instances"],  # type: ignore[index]
                    task_hash=tr["task_hash"],  # type: ignore[index]
                    primary_metric=tr["primary_metric"],  # type: ignore[index]
                    primary_score=tr["primary_score"],  # type: ignore[index]
                    s3_metrics_key=tr["s3_metrics_key"],  # type: ignore[index]
                    s3_predictions_key=tr["s3_predictions_key"],  # type: ignore[index]
                )
                for tr in task_rows
            ]

            return EvalResult(
                run_id=str(row["run_id"]),
                model_name=row["model_name"],
                backend_name=row["backend_name"],
                timestamp=row["timestamp"],
                tasks=tasks,
                experiment_name=row["experiment_name"],
                workspace=row["workspace"],
                author=row["author"],
                tags=row["tags"],
                git_ref=row["git_ref"],
                model_hash=row["model_hash"],
                revision=row["revision"],
                s3_location=row["s3_location"],
                config=row["config"],
                metadata=row["metadata"],
            )

    def query(
        self,
        model_name: str | None = None,
        task_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[EvalResult]:
        """Query evaluation results by filters."""
        conditions = []
        params: list[Any] = []

        if model_name:
            conditions.append("e.model_name = %s")
            params.append(model_name)

        if task_name:
            conditions.append(
                "EXISTS (SELECT 1 FROM task_results t "
                "WHERE t.run_id = e.run_id AND t.task_name = %s)"
            )
            params.append(task_name)

        if start_time:
            conditions.append("e.timestamp >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("e.timestamp <= %s")
            params.append(end_time)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)

        query = f"""
            SELECT e.run_id
            FROM eval_runs e
            {where_clause}
            ORDER BY e.timestamp DESC
            LIMIT %s
        """

        results = []
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()  # type: ignore[arg-type]

            for row in rows:
                result = self.get(str(row["run_id"]))
                if result:
                    results.append(result)

        return results

    def delete(self, run_id: str) -> bool:
        """Delete an evaluation result."""
        with self._get_connection() as conn:
            # CASCADE will delete task_results
            result = conn.execute("DELETE FROM eval_runs WHERE run_id = %s", (run_id,))
            conn.commit()
            return result.rowcount > 0

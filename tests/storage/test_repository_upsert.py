"""Tests that repository saves are keyed upserts rather than plain inserts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy.dialects import postgresql

from olmo_eval.common.types import EvalResult, StoredTaskResult
from olmo_eval.storage.backends.postgres.models import Experiment, TaskResult
from olmo_eval.storage.backends.postgres.repository import (
    ExperimentRepository,
    InstancePredictionRepository,
)


class _ScalarResult:
    def scalar_one(self) -> int:
        return 7


class RecordingSession:
    def __init__(self) -> None:
        self.statements: list[Any] = []

    def execute(self, stmt: Any, params: Any = None) -> _ScalarResult:
        self.statements.append(stmt)
        return _ScalarResult()


def _sql(stmt: Any) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def _set_clause(sql: str) -> str:
    return sql.split("DO UPDATE SET", 1)[1]


def _result(tasks: list[StoredTaskResult]) -> EvalResult:
    return EvalResult(
        experiment_id="exp-1",
        model_name="model-a",
        backend_name="vllm",
        timestamp=datetime(2026, 10, 1, tzinfo=UTC),
        tasks=tasks,
        experiment_name="exp-1",
        workspace="ws",
        author="tester",
        git_ref="abc",
        model_hash="mh",
        revision="main",
        metadata={"k": "v"},
    )


def _task(name: str) -> StoredTaskResult:
    return StoredTaskResult(
        task_name=name,
        metrics={"accuracy": {"exact_match": 0.5}},
        task_hash=f"{name}-hash",
        num_instances=2,
    )


def test_identity_constraints_are_declared_on_the_models() -> None:
    def unique_columns(model: Any, name: str) -> list[str]:
        constraint = next(c for c in model.__table__.constraints if c.name == name)
        return [column.name for column in constraint.columns]

    assert unique_columns(Experiment, "uq_experiments_identity") == [
        "experiment_id",
        "model_name",
        "model_hash",
    ]
    assert unique_columns(TaskResult, "uq_task_results_experiment_task") == [
        "experiment_pk",
        "task_name",
    ]


def test_experiment_save_upserts_on_identity_key() -> None:
    session = RecordingSession()

    experiment_pk = ExperimentRepository(session).save(_result([_task("gsm8k")]))  # type: ignore[arg-type]

    assert experiment_pk == 7
    sql = _sql(session.statements[0])
    assert "ON CONFLICT ON CONSTRAINT uq_experiments_identity DO UPDATE" in sql
    assert "RETURNING experiments.id" in sql
    updated = _set_clause(sql)
    assert "metadata = excluded.metadata" in updated
    assert "author = excluded.author" in updated
    for key in ("experiment_id", "model_name", "model_hash", "created_at"):
        assert f" {key} = " not in updated


def test_experiment_save_keeps_stored_s3_location_when_new_one_is_null() -> None:
    session = RecordingSession()

    ExperimentRepository(session).save(_result([]))  # type: ignore[arg-type]

    updated = _set_clause(_sql(session.statements[0]))
    assert "s3_location = coalesce(excluded.s3_location, experiments.s3_location)" in updated
    assert "model_path = excluded.model_path" in updated


def test_task_results_upsert_on_experiment_and_task_name() -> None:
    session = RecordingSession()

    ExperimentRepository(session).save(_result([_task("gsm8k"), _task("mmlu")]))  # type: ignore[arg-type]

    assert len(session.statements) == 2
    sql = _sql(session.statements[1])
    assert "ON CONFLICT ON CONSTRAINT uq_task_results_experiment_task DO UPDATE" in sql
    updated = _set_clause(sql)
    assert "metrics = excluded.metrics" in updated
    assert "task_hash = excluded.task_hash" in updated
    assert " experiment_pk = " not in updated
    assert " task_name = " not in updated


def test_experiment_save_without_tasks_skips_task_upsert() -> None:
    session = RecordingSession()

    ExperimentRepository(session).save(_result([]))  # type: ignore[arg-type]

    assert len(session.statements) == 1


def test_save_instances_replaces_existing_rows_for_the_task() -> None:
    session = RecordingSession()

    InstancePredictionRepository(session).save_instances(  # type: ignore[arg-type]
        experiment_pk=7,
        task_hash="gsm8k-hash",
        instances=[{"native_id": "a", "instance_metrics": {"acc": {"acc": 1.0}}}],
    )

    delete_sql = _sql(session.statements[0])
    assert delete_sql.startswith("DELETE FROM instance_predictions")
    assert "instance_predictions.experiment_pk = %(experiment_pk_1)s" in delete_sql
    assert "instance_predictions.task_hash = %(task_hash_1)s" in delete_sql
    assert _sql(session.statements[1]).startswith("INSERT INTO instance_predictions")


def test_save_instances_with_no_instances_still_clears_the_task() -> None:
    session = RecordingSession()

    InstancePredictionRepository(session).save_instances(  # type: ignore[arg-type]
        experiment_pk=7, task_hash="gsm8k-hash", instances=[]
    )

    assert len(session.statements) == 1
    assert _sql(session.statements[0]).startswith("DELETE FROM instance_predictions")

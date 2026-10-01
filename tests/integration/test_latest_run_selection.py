"""Integration tests for latest-run selection against a real Postgres."""

from datetime import datetime

import pytest

MODEL_HASH = "model-hash-latest"
TASK_HASH = "task-hash-latest"
COMPLETE_METRICS = {"accuracy": {"exact_match": 0.5}}


def _save_run(
    backend,
    *,
    experiment_id: str,
    hour: int,
    metrics: dict,
    instances_failed: int | None,
    native_ids: list[str],
) -> int:
    """Save one run of the shared task and return its experiment primary key."""
    from olmo_eval.common.types import EvalResult, StoredTaskResult
    from olmo_eval.storage.backends.postgres.repository import (
        ExperimentRepository,
        InstancePredictionRepository,
    )

    result = EvalResult(
        experiment_id=experiment_id,
        experiment_name=experiment_id,
        workspace="ai2/olmo-test",
        author="test-runner",
        git_ref="abc123",
        revision="main",
        model_name="model-a",
        backend_name="vllm",
        timestamp=datetime(2026, 9, 1, hour, 0, 0),
        model_hash=MODEL_HASH,
        tasks=[
            StoredTaskResult(
                task_name="task-a",
                metrics=metrics,
                task_hash=TASK_HASH,
                num_instances=len(native_ids),
                instances_processed=len(native_ids) + (instances_failed or 0),
                instances_failed=instances_failed,
                primary_metric="accuracy:exact_match",
            )
        ],
    )
    with backend.db.session() as session:
        experiment_pk = ExperimentRepository(session).save(result)
    if native_ids:
        with backend.db.session() as session:
            InstancePredictionRepository(session).save_instances(
                experiment_pk=experiment_pk,
                task_hash=TASK_HASH,
                instances=[
                    {"native_id": native_id, "instance_metrics": {"acc": 1.0}}
                    for native_id in native_ids
                ],
            )
    return experiment_pk


def _selected_pks(backend, data_table, source_pks: list[int]) -> list[int]:
    from sqlalchemy import select

    from olmo_eval.analysis.pairwise import _latest_source_experiment_pks_subquery

    subquery = _latest_source_experiment_pks_subquery(
        data_table=data_table,
        source_pks=source_pks,
        extra_filters=[],
    )
    with backend.db.session() as session:
        return [int(row.experiment_pk) for row in session.execute(select(subquery)).all()]


class TestLatestRunSelection:
    """A newer run replaces an older one only when it is at least as complete."""

    @pytest.mark.integration
    def test_newer_failed_run_does_not_replace_older_complete_run(self, postgres_backend):
        from olmo_eval.storage.backends.postgres.models import TaskResult

        complete = _save_run(
            postgres_backend,
            experiment_id="complete",
            hour=8,
            metrics=COMPLETE_METRICS,
            instances_failed=0,
            native_ids=["q1", "q2"],
        )
        failed = _save_run(
            postgres_backend,
            experiment_id="failed",
            hour=12,
            metrics={},
            instances_failed=2,
            native_ids=[],
        )

        assert _selected_pks(postgres_backend, TaskResult, [complete, failed]) == [complete]

    @pytest.mark.integration
    def test_newer_partial_run_does_not_replace_older_complete_run(self, postgres_backend):
        from olmo_eval.storage.backends.postgres.models import InstancePrediction, TaskResult

        complete = _save_run(
            postgres_backend,
            experiment_id="complete",
            hour=8,
            metrics=COMPLETE_METRICS,
            instances_failed=0,
            native_ids=["q1", "q2", "q3"],
        )
        partial = _save_run(
            postgres_backend,
            experiment_id="partial",
            hour=12,
            metrics={"accuracy": {"exact_match": 0.9}},
            instances_failed=1,
            native_ids=["q1", "q2"],
        )

        source_pks = [complete, partial]
        assert _selected_pks(postgres_backend, TaskResult, source_pks) == [complete]
        assert _selected_pks(postgres_backend, InstancePrediction, source_pks) == [complete]

    @pytest.mark.integration
    def test_partial_run_is_preferred_over_failed_run(self, postgres_backend):
        from olmo_eval.storage.backends.postgres.models import TaskResult

        partial = _save_run(
            postgres_backend,
            experiment_id="partial",
            hour=8,
            metrics=COMPLETE_METRICS,
            instances_failed=1,
            native_ids=["q1"],
        )
        failed = _save_run(
            postgres_backend,
            experiment_id="failed",
            hour=12,
            metrics={},
            instances_failed=2,
            native_ids=[],
        )

        assert _selected_pks(postgres_backend, TaskResult, [partial, failed]) == [partial]

    @pytest.mark.integration
    def test_newest_run_wins_among_equally_complete_runs(self, postgres_backend):
        from olmo_eval.storage.backends.postgres.models import InstancePrediction, TaskResult

        older = _save_run(
            postgres_backend,
            experiment_id="older",
            hour=8,
            metrics=COMPLETE_METRICS,
            instances_failed=0,
            native_ids=["q1"],
        )
        legacy = _save_run(
            postgres_backend,
            experiment_id="legacy",
            hour=10,
            metrics=COMPLETE_METRICS,
            instances_failed=None,
            native_ids=["q1"],
        )

        source_pks = [older, legacy]
        assert _selected_pks(postgres_backend, TaskResult, source_pks) == [legacy]
        assert _selected_pks(postgres_backend, InstancePrediction, source_pks) == [legacy]

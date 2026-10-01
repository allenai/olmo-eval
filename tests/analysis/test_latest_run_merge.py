"""Tests for the shared latest-run task-row merge."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from olmo_eval.analysis.latest_run_merge import (
    COMPLETE_RUN_RANK,
    FAILED_RUN_RANK,
    PARTIAL_RUN_RANK,
    LatestTaskRowInput,
    completeness_rank,
    merge_latest_task_rows,
)

METRICS = {"accuracy": {"exact_match": 0.5}}


@pytest.mark.parametrize(
    ("metrics", "instances_failed", "expected"),
    [
        (METRICS, 0, COMPLETE_RUN_RANK),
        (METRICS, None, COMPLETE_RUN_RANK),
        (METRICS, 2, PARTIAL_RUN_RANK),
        ({}, 0, FAILED_RUN_RANK),
        ({}, 2, FAILED_RUN_RANK),
        (None, None, FAILED_RUN_RANK),
    ],
)
def test_completeness_rank(metrics, instances_failed: int | None, expected: int) -> None:
    assert completeness_rank(metrics, instances_failed) == expected


def _experiment(experiment_id: int, *, hour: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=experiment_id,
        model_hash="model-hash",
        timestamp=datetime(2026, 9, 1, hour, 0, tzinfo=UTC),
    )


def _row(experiment_pk: int, score: float, instances_failed: int | None) -> LatestTaskRowInput:
    return LatestTaskRowInput(
        experiment_pk=experiment_pk,
        task_name="task-a",
        task_hash="task-hash-a",
        metrics={"accuracy": {"exact_match": score}},
        primary_metric="accuracy:exact_match",
        instances_failed=instances_failed,
    )


def _merged_score(rows: list[LatestTaskRowInput]) -> float:
    older = _experiment(10, hour=8)
    newer = _experiment(11, hour=12)
    (merged,) = merge_latest_task_rows(
        task_rows=rows, source_experiments=[older, newer], display_experiments=[newer]
    )
    return merged.metrics["accuracy"]["exact_match"]


def test_complete_run_values_beat_newer_partial_run() -> None:
    assert _merged_score([_row(10, 0.5, 0), _row(11, 0.9, 1)]) == pytest.approx(0.5)


def test_newer_run_wins_among_equally_complete_runs() -> None:
    assert _merged_score([_row(10, 0.5, 0), _row(11, 0.9, None)]) == pytest.approx(0.9)

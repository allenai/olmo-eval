"""Tests for olmo_eval.analysis.scope_scores."""

from collections.abc import Iterator

import pytest

from olmo_eval.analysis.scope_scores import compute_scope_score, weights_are_complete
from olmo_eval.evals.suites.registry import _REGISTRY, AggregationStrategy, Suite


@pytest.fixture
def weighted_suite() -> Iterator[Suite]:
    suite = Suite(
        name="_test_weighted_scope",
        tasks=("task_small", "task_large"),
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    )
    _REGISTRY[suite.name] = suite
    yield suite
    del _REGISTRY[suite.name]


def test_weighted_average_weights_tasks_by_instance_count(weighted_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_small": [0.9], "task_large": [0.5]},
        task_instance_counts_by_name={"task_small": [100], "task_large": [300]},
        suite_name=weighted_suite.name,
    )

    # (0.9 * 100 + 0.5 * 300) / 400 = 0.6, versus 0.7 unweighted.
    assert score == pytest.approx(0.6)


def test_weighted_average_collapses_task_hash_variants(weighted_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_small": [0.8, 1.0], "task_large": [0.5]},
        task_instance_counts_by_name={"task_small": [100, 100], "task_large": [300]},
        suite_name=weighted_suite.name,
    )

    assert score == pytest.approx(0.6)


def test_weighted_average_ignores_unscored_variants(weighted_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_small": [0.9, None], "task_large": [0.5]},
        task_instance_counts_by_name={"task_small": [100, None], "task_large": [300]},
        suite_name=weighted_suite.name,
    )

    assert score == pytest.approx(0.6)


def test_weighted_average_skips_tasks_without_scores(weighted_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_small": [0.9], "task_large": [None]},
        task_instance_counts_by_name={"task_small": [100], "task_large": [300]},
        suite_name=weighted_suite.name,
    )

    assert score == pytest.approx(0.9)


@pytest.mark.parametrize(
    "counts",
    [
        None,
        {},
        {"task_small": [100]},
        {"task_small": [100], "task_large": [None]},
        {"task_small": [100], "task_large": [0]},
    ],
)
def test_weighted_average_falls_back_to_unweighted_mean(
    weighted_suite: Suite,
    counts: dict[str, list[int | None]] | None,
) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_small": [0.9], "task_large": [0.5]},
        task_instance_counts_by_name=counts,
        suite_name=weighted_suite.name,
    )

    assert score == pytest.approx(0.7)


def test_weighted_average_without_scores_is_none(weighted_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_small": [None], "task_large": [None]},
        task_instance_counts_by_name={"task_small": [100], "task_large": [300]},
        suite_name=weighted_suite.name,
    )

    assert score is None


def test_instance_counts_are_ignored_by_unweighted_strategies() -> None:
    suite = Suite(
        name="_test_unweighted_scope",
        tasks=("task_small", "task_large"),
        aggregation=AggregationStrategy.AVERAGE,
    )
    _REGISTRY[suite.name] = suite
    try:
        score = compute_scope_score(
            task_scores_by_name={"task_small": [0.9], "task_large": [0.5]},
            task_instance_counts_by_name={"task_small": [100], "task_large": [300]},
            suite_name=suite.name,
        )
    finally:
        del _REGISTRY[suite.name]

    assert score == pytest.approx(0.7)


def test_weighted_child_of_average_of_averages_is_weighted() -> None:
    child = Suite(
        name="_test_weighted_child",
        tasks=("task_small", "task_large"),
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    )
    parent = Suite(
        name="_test_weighted_child_parent",
        tasks=("task_standalone", child),
        aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
    )
    _REGISTRY[parent.name] = parent
    try:
        score = compute_scope_score(
            task_scores_by_name={
                "task_standalone": [1.0],
                "task_small": [0.9],
                "task_large": [0.5],
            },
            task_instance_counts_by_name={
                "task_standalone": [50],
                "task_small": [100],
                "task_large": [300],
            },
            suite_name=parent.name,
        )
    finally:
        del _REGISTRY[parent.name]

    # The child collapses to its weighted mean of 0.6, then the parent weights
    # its two children equally: (1.0 + 0.6) / 2.
    assert score == pytest.approx(0.8)


def test_weights_are_complete_reports_missing_counts(weighted_suite: Suite) -> None:
    scores = {"task_small": [0.9], "task_large": [0.5]}

    assert weights_are_complete(
        task_scores_by_name=scores,
        task_instance_counts_by_name={"task_small": [100], "task_large": [300]},
        suite_name=weighted_suite.name,
    )
    assert not weights_are_complete(
        task_scores_by_name=scores,
        task_instance_counts_by_name={"task_small": [100], "task_large": [None]},
        suite_name=weighted_suite.name,
    )
    # An unscored task needs no weight.
    assert weights_are_complete(
        task_scores_by_name={"task_small": [0.9], "task_large": [None]},
        task_instance_counts_by_name={"task_small": [100]},
        suite_name=weighted_suite.name,
    )


def test_weights_are_complete_ignores_unweighted_scopes() -> None:
    suite = Suite(
        name="_test_unweighted_complete",
        tasks=("task_small", "task_large"),
        aggregation=AggregationStrategy.AVERAGE,
    )
    _REGISTRY[suite.name] = suite
    try:
        assert weights_are_complete(
            task_scores_by_name={"task_small": [0.9], "task_large": [0.5]},
            task_instance_counts_by_name=None,
            suite_name=suite.name,
        )
    finally:
        del _REGISTRY[suite.name]

    assert weights_are_complete(task_scores_by_name={"task_small": [0.9]})
    assert weights_are_complete(
        task_scores_by_name={"task_small": [0.9]}, suite_name="_not_a_registered_suite"
    )

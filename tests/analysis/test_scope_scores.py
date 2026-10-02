"""Tests for olmo_eval.analysis.scope_scores."""

from collections.abc import Iterator

import pytest

from olmo_eval.analysis.scope_scores import compute_scope_score
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
def test_weighted_average_without_a_count_has_no_score(
    weighted_suite: Suite,
    counts: dict[str, list[int | None]] | None,
) -> None:
    """A weighted suite reports a weighted mean or nothing, never a macro mean."""
    score = compute_scope_score(
        task_scores_by_name={"task_small": [0.9], "task_large": [0.5]},
        task_instance_counts_by_name=counts,
        suite_name=weighted_suite.name,
    )

    assert score is None


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


def test_weighted_child_without_a_count_drops_out_of_its_parent() -> None:
    child = Suite(
        name="_test_weighted_child_missing",
        tasks=("task_small", "task_large"),
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
    )
    parent = Suite(
        name="_test_weighted_child_missing_parent",
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
            task_instance_counts_by_name={"task_standalone": [50], "task_small": [100]},
            suite_name=parent.name,
        )
    finally:
        del _REGISTRY[parent.name]

    # The child contributes nothing, leaving the standalone task alone.
    assert score == pytest.approx(1.0)


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


@pytest.fixture
def gap_suite() -> Iterator[Suite]:
    suite = Suite(
        name="_test_gap_scope",
        tasks=("task_in", "task_out"),
        aggregation=AggregationStrategy.GAP,
    )
    _REGISTRY[suite.name] = suite
    yield suite
    del _REGISTRY[suite.name]


_SAME_METRIC = {"task_in": ["exact_match"], "task_out": ["exact_match"]}


def test_gap_is_companion_minus_reference(gap_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_in": [0.6], "task_out": [0.45]},
        task_metrics_by_name=_SAME_METRIC,
        suite_name=gap_suite.name,
    )

    assert score == pytest.approx(-0.15)


def test_gap_without_both_tasks_is_none(gap_suite: Suite) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_in": [0.6], "task_out": [None]},
        task_metrics_by_name=_SAME_METRIC,
        suite_name=gap_suite.name,
    )

    assert score is None


@pytest.mark.parametrize(
    "task_metrics_by_name",
    [
        None,
        {"task_in": ["exact_match"], "task_out": ["exact_match_flex"]},
        {"task_in": ["exact_match"], "task_out": [None]},
        {"task_in": ["exact_match", "exact_match_flex"], "task_out": ["exact_match"]},
    ],
    ids=["unknown", "different", "missing", "mixed-variants"],
)
def test_gap_needs_one_shared_metric(
    gap_suite: Suite, task_metrics_by_name: dict[str, list[str | None]] | None
) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_in": [0.6], "task_out": [0.45]},
        task_metrics_by_name=task_metrics_by_name,
        suite_name=gap_suite.name,
    )

    assert score is None


@pytest.mark.parametrize(
    ("task_in_scores", "expected"),
    [([0.6, None], -0.15), ([0.6, 0.5], None)],
    ids=["unscored-variant-ignored", "scored-variants-differ"],
)
def test_gap_reads_metrics_of_scored_variants_only(
    gap_suite: Suite, task_in_scores: list[float | None], expected: float | None
) -> None:
    score = compute_scope_score(
        task_scores_by_name={"task_in": task_in_scores, "task_out": [0.45]},
        task_metrics_by_name={
            "task_in": ["exact_match", "exact_match_flex"],
            "task_out": ["exact_match"],
        },
        suite_name=gap_suite.name,
    )

    assert score == (pytest.approx(expected) if expected is not None else None)

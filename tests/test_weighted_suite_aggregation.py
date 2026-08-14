"""Regression tests for weighted suite aggregation."""

import pytest

from olmo_eval.evals.suites.registry import _REGISTRY, AggregationStrategy, Suite, register
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations


@pytest.fixture
def isolated_suite_registry():
    original = _REGISTRY.copy()
    _REGISTRY.clear()
    try:
        yield
    finally:
        _REGISTRY.clear()
        _REGISTRY.update(original)


def _result(score: float) -> dict:
    return {
        "metrics": {"accuracy": {"exact": score}},
        "primary_metric": "accuracy:exact",
    }


def test_weighted_average_uses_direct_child_weights(isolated_suite_registry):
    register(
        Suite(
            name="weighted",
            tasks=("a", "b"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
            weights=(0.75, 0.25),
        )
    )

    results = compute_suite_aggregations(
        ["weighted"],
        {"a": _result(0.8), "b": _result(0.2)},
    )

    weighted = results["weighted"]
    assert weighted["metrics"]["accuracy"]["exact"] == pytest.approx(0.65)
    assert weighted["metrics"]["primary_score"]["average"] == pytest.approx(0.65)
    assert weighted["weights"] == [0.75, 0.25]


def test_weighted_average_renormalizes_when_child_is_missing(isolated_suite_registry):
    register(
        Suite(
            name="weighted",
            tasks=("a", "b"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
            weights=(0.75, 0.25),
        )
    )

    results = compute_suite_aggregations(["weighted"], {"a": _result(0.8)})

    assert results["weighted"]["metrics"]["accuracy"]["exact"] == pytest.approx(0.8)


def test_nested_weighted_suites_preserve_internal_weights(isolated_suite_registry):
    inner = Suite(
        name="inner",
        tasks=("a", "b"),
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        weights=(0.25, 0.75),
    )
    outer = Suite(
        name="outer",
        tasks=(inner, "c"),
        aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        weights=(0.4, 0.6),
    )
    register(outer)

    results = compute_suite_aggregations(
        ["outer"],
        {"a": _result(1.0), "b": _result(0.0), "c": _result(0.5)},
    )

    assert results["inner"]["metrics"]["accuracy"]["exact"] == pytest.approx(0.25)
    assert results["outer"]["metrics"]["accuracy"]["exact"] == pytest.approx(0.4)
    assert results["outer"]["nested_suites"] == ["inner"]


def test_weighted_suite_validates_weights():
    with pytest.raises(ValueError, match="weights are required"):
        Suite(
            name="missing",
            tasks=("a",),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
        )

    with pytest.raises(ValueError, match="weights must match tasks"):
        Suite(
            name="mismatch",
            tasks=("a", "b"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
            weights=(1.0,),
        )

    with pytest.raises(ValueError, match="finite and non-negative"):
        Suite(
            name="negative",
            tasks=("a",),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
            weights=(-1.0,),
        )

    with pytest.raises(ValueError, match="at least one weight"):
        Suite(
            name="zero",
            tasks=("a", "b"),
            aggregation=AggregationStrategy.WEIGHTED_AVERAGE,
            weights=(0.0, 0.0),
        )

    with pytest.raises(ValueError, match="weights can only be set"):
        Suite(name="wrong-strategy", tasks=("a",), weights=(1.0,))

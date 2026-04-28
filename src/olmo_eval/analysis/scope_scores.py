"""Shared scope-score helpers for viewer summaries and exports."""

from __future__ import annotations

from typing import Any


def _mean_numeric(values: list[float | None]) -> float | None:
    scored_values = [float(value) for value in values if value is not None]
    if not scored_values:
        return None
    return sum(scored_values) / len(scored_values)


def _task_score(
    task_name: str,
    task_scores_by_name: dict[str, list[float | None]],
) -> float | None:
    return _mean_numeric(task_scores_by_name.get(task_name, []))


def _child_scope_score(
    child: str | Any,
    task_scores_by_name: dict[str, list[float | None]],
) -> float | None:
    from olmo_eval.evals.suites.registry import Suite

    if isinstance(child, Suite):
        # Mirror the current runner behavior for nested children in an
        # average-of-averages parent: collapse the child to the mean of its
        # expanded task leaves before the parent averages across children.
        return _mean_numeric(
            [_task_score(task_name, task_scores_by_name) for task_name in child.expand()]
        )

    return _task_score(str(child), task_scores_by_name)


def compute_scope_score(
    *,
    task_scores_by_name: dict[str, list[float | None]],
    suite_name: str | None = None,
    task_name: str | None = None,
) -> float | None:
    """Compute a scalar scope score using the suite registry's aggregation rules.

    ``task_scores_by_name`` is keyed by canonical task name so multiple task-hash
    variants of the same task collapse to a single leaf score before suite
    aggregation is applied.
    """
    from olmo_eval.evals.suites.registry import AggregationStrategy, get_suite, suite_exists

    if suite_name:
        if not suite_exists(suite_name):
            return None

        suite = get_suite(suite_name)
        if suite.aggregation == AggregationStrategy.NONE:
            return None
        if suite.aggregation == AggregationStrategy.AVERAGE_OF_AVERAGES:
            return _mean_numeric(
                [_child_scope_score(child, task_scores_by_name) for child in suite.tasks]
            )

        return _mean_numeric(
            [_task_score(expanded_task, task_scores_by_name) for expanded_task in suite.expand()]
        )

    if task_name:
        return _task_score(task_name, task_scores_by_name)

    return _mean_numeric(
        [score for task_scores in task_scores_by_name.values() for score in task_scores]
    )

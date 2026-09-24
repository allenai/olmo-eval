"""Shared scope-score helpers for viewer summaries and exports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import zip_longest
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


def _task_weight(
    task_name: str,
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[float | None]] | None,
) -> float | None:
    """Instance count to weight one leaf task by, or None when it is unusable.

    Task-hash variants of the same task collapse to a single leaf score, so the
    variants that contributed that score also collapse to a single weight. That
    collapse is an unweighted mean, matching how the variants' scores collapse:
    the variants are reruns of one task rather than distinct populations of
    instances, so a rerun on more instances does not make the task itself count
    for more in the suite. Every contributing variant must carry a positive
    count, otherwise the leaf has no weight at all.
    """
    if not task_instance_counts_by_name:
        return None

    scores = task_scores_by_name.get(task_name, [])
    counts = task_instance_counts_by_name.get(task_name, [])
    weights = [
        float(count)
        for score, count in zip_longest(scores, counts)
        if score is not None and count is not None and count > 0
    ]
    scored_count = sum(1 for score in scores if score is not None)
    if not weights or len(weights) != scored_count:
        return None
    return sum(weights) / len(weights)


def _weighted_mean_over_tasks(
    task_names: Iterable[str],
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[float | None]] | None,
) -> float | None:
    """Instance-weighted mean over leaf tasks, or None when a weight is missing.

    A weighted suite reports its instance-weighted mean or no score at all. An
    unweighted mean published under the same suite name would be a different
    statistic wearing the same label, and nothing downstream could tell the two
    apart.
    """
    weighted_scores: list[tuple[float, float]] = []
    for task_name in task_names:
        score = _task_score(task_name, task_scores_by_name)
        if score is None:
            continue
        weight = _task_weight(task_name, task_scores_by_name, task_instance_counts_by_name)
        if weight is None:
            return None
        weighted_scores.append((score, weight))

    total_weight = sum(weight for _, weight in weighted_scores)
    if not weighted_scores or total_weight <= 0:
        return None
    return sum(score * weight for score, weight in weighted_scores) / total_weight


def _child_scope_score(
    child: str | Any,
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[float | None]] | None,
) -> float | None:
    from olmo_eval.evals.suites.registry import AggregationStrategy, Suite

    if isinstance(child, Suite):
        # Mirror the current runner behavior for nested children in an
        # average-of-averages parent: collapse the child to the mean of its
        # expanded task leaves before the parent averages across children.
        if child.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            return _weighted_mean_over_tasks(
                child.expand(),
                task_scores_by_name,
                task_instance_counts_by_name,
            )
        return _mean_numeric(
            [_task_score(task_name, task_scores_by_name) for task_name in child.expand()]
        )

    return _task_score(str(child), task_scores_by_name)


def compute_scope_score(
    *,
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[float | None]] | None = None,
    suite_name: str | None = None,
    task_name: str | None = None,
) -> float | None:
    """Compute a scalar scope score using the suite registry's aggregation rules.

    ``task_scores_by_name`` is keyed by canonical task name so multiple task-hash
    variants of the same task collapse to a single leaf score before suite
    aggregation is applied. ``task_instance_counts_by_name`` carries the instance
    count of each of those variants in the same order, and is only read by
    instance-weighted aggregation strategies. Those strategies return None when
    a contributing task has no instance count.
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
                [
                    _child_scope_score(child, task_scores_by_name, task_instance_counts_by_name)
                    for child in suite.tasks
                ]
            )
        if suite.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            return _weighted_mean_over_tasks(
                suite.expand(),
                task_scores_by_name,
                task_instance_counts_by_name,
            )

        return _mean_numeric(
            [_task_score(expanded_task, task_scores_by_name) for expanded_task in suite.expand()]
        )

    if task_name:
        return _task_score(task_name, task_scores_by_name)

    return _mean_numeric(
        [score for task_scores in task_scores_by_name.values() for score in task_scores]
    )

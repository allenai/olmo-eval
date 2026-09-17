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
    task_instance_counts_by_name: Mapping[str, Sequence[int | None]] | None,
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
    task_instance_counts_by_name: Mapping[str, Sequence[int | None]] | None,
) -> float | None:
    """Instance-weighted mean over leaf tasks, unweighted when any weight is missing.

    Mixing weighted and unweighted terms would produce a number that belongs to
    neither convention, so a single missing instance count drops the whole suite
    back to the unweighted mean.
    """
    scores: list[float] = []
    weights: list[float] = []
    for task_name in task_names:
        score = _task_score(task_name, task_scores_by_name)
        if score is None:
            continue
        scores.append(score)
        weight = _task_weight(task_name, task_scores_by_name, task_instance_counts_by_name)
        if weight is not None:
            weights.append(weight)

    if not scores:
        return None
    if len(weights) != len(scores) or sum(weights) <= 0:
        return sum(scores) / len(scores)
    return sum(score * weight for score, weight in zip(scores, weights, strict=True)) / sum(weights)


def _child_scope_score(
    child: str | Any,
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[int | None]] | None,
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


def weights_are_complete(
    *,
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[int | None]] | None = None,
    suite_name: str | None = None,
) -> bool:
    """Whether every scored leaf of a weighted suite carries an instance count.

    A scope score falls back to the unweighted mean when this is False. Callers
    that render several models side by side ask this of every model first and
    weight none of them unless all of them qualify, so one comparison column
    never mixes instance-weighted and unweighted aggregates.

    Always True for suites that do not weight by instance count, and for scopes
    that are not suites, since those ignore instance counts entirely.
    """
    from olmo_eval.evals.suites.registry import (
        AggregationStrategy,
        Suite,
        get_suite,
        suite_exists,
    )

    if not suite_name or not suite_exists(suite_name):
        return True

    suite = get_suite(suite_name)
    weighted_leaves: tuple[str, ...]
    if suite.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
        weighted_leaves = suite.expand()
    elif suite.aggregation == AggregationStrategy.AVERAGE_OF_AVERAGES:
        weighted_leaves = tuple(
            task_name
            for child in suite.tasks
            if isinstance(child, Suite)
            and child.aggregation == AggregationStrategy.WEIGHTED_AVERAGE
            for task_name in child.expand()
        )
    else:
        return True

    return all(
        _task_weight(task_name, task_scores_by_name, task_instance_counts_by_name) is not None
        for task_name in weighted_leaves
        if _task_score(task_name, task_scores_by_name) is not None
    )


def compute_scope_score(
    *,
    task_scores_by_name: dict[str, list[float | None]],
    task_instance_counts_by_name: Mapping[str, Sequence[int | None]] | None = None,
    suite_name: str | None = None,
    task_name: str | None = None,
) -> float | None:
    """Compute a scalar scope score using the suite registry's aggregation rules.

    ``task_scores_by_name`` is keyed by canonical task name so multiple task-hash
    variants of the same task collapse to a single leaf score before suite
    aggregation is applied. ``task_instance_counts_by_name`` carries the instance
    count of each of those variants in the same order, and is only read by
    instance-weighted aggregation strategies.
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

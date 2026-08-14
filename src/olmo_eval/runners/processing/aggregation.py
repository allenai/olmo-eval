"""Suite aggregation utilities for computing aggregate metrics across tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ChildAverageResult:
    """Result from computing a child average."""

    metrics: dict[str, dict[str, float]]  # Nested: {metric: {scorer: value}}
    tasks: list[str]
    primary_score: float | None = None  # Average of primary metric values
    # If child was a Suite, include its info for separate reporting
    nested_suite: Any | None = None  # Suite or None
    nested_suite_key: str | None = None  # Key to use in results (with suffixes)


def _flatten_nested_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    """Flatten nested metrics to simple dict for aggregation.

    Converts {metric: {scorer: value}} to {"metric:scorer": value}.
    """
    result: dict[str, float] = {}
    for metric_name, scorers in metrics.items():
        for scorer_name, value in scorers.items():
            result[f"{metric_name}:{scorer_name}"] = value
    return result


def _unflatten_metrics(flat_metrics: dict[str, float]) -> dict[str, dict[str, float]]:
    """Convert flat metrics back to nested structure.

    Converts {"metric:scorer": value} to {metric: {scorer: value}}.
    """
    from olmo_eval.runners.processing.utils import parse_metric_key

    result: dict[str, dict[str, float]] = {}
    for key, value in flat_metrics.items():
        parsed = parse_metric_key(key)
        if parsed:
            metric_name, scorer_name = parsed
        else:
            metric_name, scorer_name = key, "default"
        if metric_name not in result:
            result[metric_name] = {}
        result[metric_name][scorer_name] = value
    return result


def _extract_primary_score(
    task_data: dict[str, Any],
) -> float | None:
    """Extract the primary metric value from a task result."""
    from olmo_eval.runners.processing.utils import extract_score_from_metrics

    primary_metric_key = task_data.get("primary_metric")
    if not primary_metric_key:
        return None
    return extract_score_from_metrics(task_data.get("metrics", {}), primary_metric_key)


def _compute_child_average(
    child: str | Any,  # str or Suite
    priority_suffix: str,
    task_results: dict[str, dict[str, Any]],
) -> ChildAverageResult | None:
    """Compute average metrics for a single child (task string or nested Suite).

    Returns:
        ChildAverageResult with metrics and task info, or None if no results found.
    """
    from olmo_eval.evals.suites.registry import Suite

    if isinstance(child, Suite):
        # Child is a nested Suite - average all its expanded tasks
        child_metrics: dict[str, list[float]] = {}
        primary_scores: list[float] = []
        tasks_included = []

        for task_spec in child.expand():
            full_task_spec = f"{task_spec}{priority_suffix}"
            if full_task_spec not in task_results:
                continue

            task_data = task_results[full_task_spec]
            nested_metrics = task_data.get("metrics", {})
            if not nested_metrics:
                continue

            # Flatten nested metrics for averaging
            flat_metrics = _flatten_nested_metrics(nested_metrics)
            if not flat_metrics:
                continue

            tasks_included.append(full_task_spec)
            for metric_key, value in flat_metrics.items():
                if metric_key not in child_metrics:
                    child_metrics[metric_key] = []
                child_metrics[metric_key].append(value)

            primary_value = _extract_primary_score(task_data)
            if primary_value is not None:
                primary_scores.append(primary_value)

        if not child_metrics:
            return None

        averaged_flat = {name: sum(vals) / len(vals) for name, vals in child_metrics.items()}
        averaged = _unflatten_metrics(averaged_flat)
        avg_primary = sum(primary_scores) / len(primary_scores) if primary_scores else None
        # Build the key for this nested suite (with suffix)
        nested_key = f"{child.name}{priority_suffix}"
        return ChildAverageResult(
            metrics=averaged,
            tasks=tasks_included,
            primary_score=avg_primary,
            nested_suite=child,
            nested_suite_key=nested_key,
        )
    else:
        # Child is a task string - get its metrics directly
        full_task_spec = f"{child}{priority_suffix}"
        if full_task_spec not in task_results:
            return None

        task_data = task_results[full_task_spec]
        metrics = task_data.get("metrics", {})
        if not metrics:
            return None

        return ChildAverageResult(
            metrics=dict(metrics),
            tasks=[full_task_spec],
            primary_score=_extract_primary_score(task_data),
            nested_suite=None,
            nested_suite_key=None,
        )


def _compute_weighted_child_average(
    child: str | Any,
    priority_suffix: str,
    task_results: dict[str, dict[str, Any]],
) -> ChildAverageResult | None:
    """Compute a direct child's score, preserving nested weighted suites."""
    from olmo_eval.evals.suites.registry import AggregationStrategy, Suite

    if isinstance(child, Suite) and child.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
        return _compute_weighted_suite_average(child, priority_suffix, task_results)
    return _compute_child_average(child, priority_suffix, task_results)


def _compute_weighted_suite_average(
    suite: Any,
    priority_suffix: str,
    task_results: dict[str, dict[str, Any]],
) -> ChildAverageResult | None:
    """Compute a weighted average over a suite's direct children.

    Each metric is normalized over the weights of children that actually produced
    that metric. This keeps a missing task from implicitly contributing a zero.
    Nested weighted suites are recursively reduced before the parent weight is
    applied.
    """
    assert suite.weights is not None

    metric_totals: dict[str, float] = {}
    metric_weights: dict[str, float] = {}
    primary_total = 0.0
    primary_weight = 0.0
    tasks_included: list[str] = []

    for child, weight in zip(suite.tasks, suite.weights, strict=True):
        if weight == 0:
            continue
        result = _compute_weighted_child_average(child, priority_suffix, task_results)
        if result is None:
            continue

        tasks_included.extend(result.tasks)
        for metric_key, value in _flatten_nested_metrics(result.metrics).items():
            metric_totals[metric_key] = metric_totals.get(metric_key, 0.0) + value * weight
            metric_weights[metric_key] = metric_weights.get(metric_key, 0.0) + weight

        if result.primary_score is not None:
            primary_total += result.primary_score * weight
            primary_weight += weight

    averaged_flat = {
        metric_key: total / metric_weights[metric_key]
        for metric_key, total in metric_totals.items()
        if metric_weights[metric_key] > 0
    }
    if not averaged_flat:
        return None

    return ChildAverageResult(
        metrics=_unflatten_metrics(averaged_flat),
        tasks=tasks_included,
        primary_score=primary_total / primary_weight if primary_weight > 0 else None,
        nested_suite=suite,
        nested_suite_key=f"{suite.name}{priority_suffix}",
    )


def _add_nested_suite_result(
    suite_aggregations: dict[str, dict[str, Any]],
    result: ChildAverageResult,
    parent_suite: str,
) -> None:
    """Record a nested suite result using the same shape as top-level aggregates."""
    if result.nested_suite is None or not result.nested_suite_key:
        return

    nested_metrics = dict(result.metrics)
    nested_result: dict[str, Any] = {
        "metrics": nested_metrics,
        "tasks": result.tasks,
        "num_tasks": len(result.tasks),
        "aggregation": result.nested_suite.aggregation.value,
        "parent_suite": parent_suite,
    }
    if result.nested_suite.weights is not None:
        nested_result["weights"] = list(result.nested_suite.weights)
    if result.primary_score is not None:
        nested_metrics["primary_score"] = {"average": result.primary_score}
        nested_result["primary_metric"] = "primary_score:average"
    suite_aggregations[result.nested_suite_key] = nested_result


def compute_suite_aggregations(
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute aggregated metrics for suites in the task specs.

    For each suite in task_specs, computes aggregated metrics based on the
    suite's aggregation strategy:
    - AVERAGE: Simple average of all expanded task scores
    - AVERAGE_OF_AVERAGES: Average over children, where nested suites are
      averaged first (each child gets equal weight)
    - WEIGHTED_AVERAGE: Weighted average over direct children, recursively
      preserving weights for nested weighted suites

    Handles specs with priority suffixes (@priority).
    When a suite has these suffixes, they are propagated to expanded task lookups.

    Args:
        task_specs: Original task specs (may include suite names with priority)
        task_results: Dict mapping task spec -> {"metrics": {...}, ...}

    Returns:
        Dict mapping suite name -> {"metrics": {...}, "tasks": [...], "aggregation": ...}
    """
    from olmo_eval.evals.suites import get_suite, suite_exists
    from olmo_eval.evals.suites.registry import AggregationStrategy, Suite

    suite_aggregations: dict[str, dict[str, Any]] = {}

    for spec in task_specs:
        # Parse out priority suffix (e.g., "suite@high" -> "suite", "@high")
        priority_suffix = ""
        base_spec = spec
        if "@" in spec:
            base_spec, priority = spec.rsplit("@", 1)
            priority_suffix = f"@{priority}"

        # Check if the base spec (without priority) is a suite
        if not suite_exists(base_spec):
            continue

        suite = get_suite(base_spec)
        if suite.aggregation == AggregationStrategy.NONE:
            continue

        if suite.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            weighted_result = _compute_weighted_suite_average(suite, priority_suffix, task_results)
            if weighted_result is None:
                continue

            aggregated_metrics = dict(weighted_result.metrics)
            nested_suites_included: list[str] = []
            assert suite.weights is not None
            for child, weight in zip(suite.tasks, suite.weights, strict=True):
                if weight == 0 or not isinstance(child, Suite):
                    continue
                child_result = _compute_weighted_child_average(child, priority_suffix, task_results)
                if child_result is None or not child_result.nested_suite_key:
                    continue
                nested_suites_included.append(child_result.nested_suite_key)
                _add_nested_suite_result(suite_aggregations, child_result, spec)

            suite_result: dict[str, Any] = {
                "metrics": aggregated_metrics,
                "tasks": weighted_result.tasks,
                "num_tasks": len(weighted_result.tasks),
                "nested_suites": nested_suites_included,
                "aggregation": suite.aggregation.value,
                "weights": list(suite.weights),
            }
            if weighted_result.primary_score is not None:
                aggregated_metrics["primary_score"] = {"average": weighted_result.primary_score}
                suite_result["primary_metric"] = "primary_score:average"
            suite_aggregations[spec] = suite_result
        elif suite.aggregation == AggregationStrategy.AVERAGE_OF_AVERAGES:
            # Average of averages: each child (task or nested suite) gets equal weight
            # Process each child separately, then average the child averages
            child_averages: dict[str, list[float]] = {}  # Flat "metric:scorer" -> values
            child_primary_scores: list[float] = []
            all_tasks_included: list[str] = []
            children_included = 0
            nested_suites_included: list[str] = []

            for child in suite.tasks:
                result = _compute_child_average(child, priority_suffix, task_results)
                if result is None:
                    continue

                all_tasks_included.extend(result.tasks)
                children_included += 1

                if result.primary_score is not None:
                    child_primary_scores.append(result.primary_score)

                # Flatten nested metrics for aggregation
                flat_metrics = _flatten_nested_metrics(result.metrics)
                for metric_key, value in flat_metrics.items():
                    if metric_key not in child_averages:
                        child_averages[metric_key] = []
                    child_averages[metric_key].append(value)

                # If this child is a nested Suite, also report its aggregation separately
                if result.nested_suite is not None and result.nested_suite_key:
                    nested_suites_included.append(result.nested_suite_key)
                    _add_nested_suite_result(suite_aggregations, result, spec)

            if not child_averages:
                continue

            # Average the child averages (each child weighted equally)
            averaged_flat = {
                name: sum(values) / len(values) for name, values in child_averages.items()
            }
            aggregated_metrics = _unflatten_metrics(averaged_flat)

            suite_result = {
                "metrics": aggregated_metrics,
                "tasks": all_tasks_included,
                "num_tasks": len(all_tasks_included),
                "num_children": children_included,
                "nested_suites": nested_suites_included,
                "aggregation": suite.aggregation.value,
            }

            if child_primary_scores:
                avg_primary = sum(child_primary_scores) / len(child_primary_scores)
                aggregated_metrics["primary_score"] = {"average": avg_primary}
                suite_result["primary_metric"] = "primary_score:average"

            suite_aggregations[spec] = suite_result
        else:
            # AVERAGE or DISPLAY_ONLY: simple average of all expanded tasks
            suite_tasks = suite.expand()
            suite_metrics: dict[str, list[float]] = {}  # Flat "metric:scorer" -> values
            task_primary_scores: list[float] = []
            tasks_included: list[str] = []

            for task_spec in suite_tasks:
                # Build the full task spec with the same suffix as the suite
                full_task_spec = f"{task_spec}{priority_suffix}"

                if full_task_spec not in task_results:
                    continue

                task_data = task_results[full_task_spec]
                nested_metrics = task_data.get("metrics", {})

                if not nested_metrics:
                    continue

                tasks_included.append(full_task_spec)

                # Flatten nested metrics for averaging
                flat_metrics = _flatten_nested_metrics(nested_metrics)
                for metric_key, value in flat_metrics.items():
                    if metric_key not in suite_metrics:
                        suite_metrics[metric_key] = []
                    suite_metrics[metric_key].append(value)

                primary_value = _extract_primary_score(task_data)
                if primary_value is not None:
                    task_primary_scores.append(primary_value)

            if not suite_metrics:
                continue

            # Compute averages and unflatten back to nested structure
            averaged_flat = {
                name: sum(values) / len(values) for name, values in suite_metrics.items()
            }
            aggregated_metrics = _unflatten_metrics(averaged_flat)

            avg_suite_result: dict[str, Any] = {
                "metrics": aggregated_metrics,
                "tasks": tasks_included,
                "num_tasks": len(tasks_included),
                "aggregation": suite.aggregation.value,
            }

            if task_primary_scores:
                avg_primary = sum(task_primary_scores) / len(task_primary_scores)
                aggregated_metrics["primary_score"] = {"average": avg_primary}
                avg_suite_result["primary_metric"] = "primary_score:average"

            suite_aggregations[spec] = avg_suite_result

    return suite_aggregations
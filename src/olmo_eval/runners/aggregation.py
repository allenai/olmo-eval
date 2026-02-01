"""Suite aggregation utilities for computing aggregate metrics across tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _ChildAverageResult:
    """Result from computing a child average."""

    metrics: dict[str, dict[str, float]]  # Nested: {metric: {scorer: value}}
    tasks: list[str]
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
    from olmo_eval.runners.common import parse_metric_key

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


def _compute_child_average(
    child: str | Any,  # str or Suite
    override_suffix: str,
    priority_suffix: str,
    task_results: dict[str, dict[str, Any]],
) -> _ChildAverageResult | None:
    """Compute average metrics for a single child (task string or nested Suite).

    Returns:
        _ChildAverageResult with metrics and task info, or None if no results found.
    """
    from olmo_eval.evals.suites.registry import Suite

    if isinstance(child, Suite):
        # Child is a nested Suite - average all its expanded tasks
        child_metrics: dict[str, list[float]] = {}
        tasks_included = []

        for task_spec in child.expand():
            full_task_spec = f"{task_spec}{override_suffix}{priority_suffix}"
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

        if not child_metrics:
            return None

        averaged_flat = {name: sum(vals) / len(vals) for name, vals in child_metrics.items()}
        averaged = _unflatten_metrics(averaged_flat)
        # Build the key for this nested suite (with suffixes)
        nested_key = f"{child.name}{override_suffix}{priority_suffix}"
        return _ChildAverageResult(
            metrics=averaged,
            tasks=tasks_included,
            nested_suite=child,
            nested_suite_key=nested_key,
        )
    else:
        # Child is a task string - get its metrics directly
        full_task_spec = f"{child}{override_suffix}{priority_suffix}"
        if full_task_spec not in task_results:
            return None

        task_data = task_results[full_task_spec]
        metrics = task_data.get("metrics", {})
        if not metrics:
            return None

        return _ChildAverageResult(
            metrics=dict(metrics),
            tasks=[full_task_spec],
            nested_suite=None,
            nested_suite_key=None,
        )


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

    Handles specs with inline overrides (::key=value) and priority suffixes (@priority).
    When a suite has these suffixes, they are propagated to expanded task lookups.

    Args:
        task_specs: Original task specs (may include suite names with overrides/priority)
        task_results: Dict mapping task spec -> {"metrics": {...}, ...}

    Returns:
        Dict mapping suite name -> {"metrics": {...}, "tasks": [...], "aggregation": ...}
    """
    from olmo_eval.evals.suites import get_suite, suite_exists
    from olmo_eval.evals.suites.registry import AggregationStrategy

    suite_aggregations: dict[str, dict[str, Any]] = {}

    for spec in task_specs:
        # Parse out priority suffix first (e.g., "suite::temp=0@high" -> "suite::temp=0", "@high")
        priority_suffix = ""
        spec_without_priority = spec
        if "@" in spec:
            spec_without_priority, priority = spec.rsplit("@", 1)
            priority_suffix = f"@{priority}"

        # Parse out overrides (e.g., "suite:variant::temp=0" -> "suite:variant", "::temp=0")
        override_suffix = ""
        base_spec = spec_without_priority
        if "::" in spec_without_priority:
            base_spec, overrides = spec_without_priority.split("::", 1)
            override_suffix = f"::{overrides}"

        # Check if the base spec (without suffixes) is a suite
        if not suite_exists(base_spec):
            continue

        suite = get_suite(base_spec)
        if suite.aggregation == AggregationStrategy.NONE:
            continue

        if suite.aggregation == AggregationStrategy.AVERAGE_OF_AVERAGES:
            # Average of averages: each child (task or nested suite) gets equal weight
            # Process each child separately, then average the child averages
            child_averages: dict[str, list[float]] = {}  # Flat "metric:scorer" -> values
            all_tasks_included: list[str] = []
            children_included = 0
            nested_suites_included: list[str] = []

            for child in suite.tasks:
                result = _compute_child_average(
                    child, override_suffix, priority_suffix, task_results
                )
                if result is None:
                    continue

                all_tasks_included.extend(result.tasks)
                children_included += 1

                # Flatten nested metrics for aggregation
                flat_metrics = _flatten_nested_metrics(result.metrics)
                for metric_key, value in flat_metrics.items():
                    if metric_key not in child_averages:
                        child_averages[metric_key] = []
                    child_averages[metric_key].append(value)

                # If this child is a nested Suite, also report its aggregation separately
                if result.nested_suite is not None and result.nested_suite_key:
                    nested_suites_included.append(result.nested_suite_key)
                    suite_aggregations[result.nested_suite_key] = {
                        "metrics": result.metrics,
                        "tasks": result.tasks,
                        "num_tasks": len(result.tasks),
                        "aggregation": result.nested_suite.aggregation.value,
                        "parent_suite": spec,  # Track which parent suite this belongs to
                    }

            if not child_averages:
                continue

            # Average the child averages (each child weighted equally)
            averaged_flat = {
                name: sum(values) / len(values) for name, values in child_averages.items()
            }
            aggregated_metrics = _unflatten_metrics(averaged_flat)

            suite_aggregations[spec] = {
                "metrics": aggregated_metrics,
                "tasks": all_tasks_included,
                "num_tasks": len(all_tasks_included),
                "num_children": children_included,
                "nested_suites": nested_suites_included,
                "aggregation": suite.aggregation.value,
            }
        else:
            # AVERAGE or DISPLAY_ONLY: simple average of all expanded tasks
            suite_tasks = suite.expand()
            suite_metrics: dict[str, list[float]] = {}  # Flat "metric:scorer" -> values
            tasks_included: list[str] = []

            for task_spec in suite_tasks:
                # Build the full task spec with the same suffixes as the suite
                full_task_spec = f"{task_spec}{override_suffix}{priority_suffix}"

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

            if not suite_metrics:
                continue

            # Compute averages and unflatten back to nested structure
            averaged_flat = {
                name: sum(values) / len(values) for name, values in suite_metrics.items()
            }
            aggregated_metrics = _unflatten_metrics(averaged_flat)

            suite_aggregations[spec] = {
                "metrics": aggregated_metrics,
                "tasks": tasks_included,
                "num_tasks": len(tasks_included),
                "aggregation": suite.aggregation.value,
            }

    return suite_aggregations

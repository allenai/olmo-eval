"""Data querying and extraction for metrics."""

from __future__ import annotations

from typing import Any

from olmo_eval.cli.metrics.config import METRICS, MetricInfo, QueryFilters
from olmo_eval.cli.metrics.utils import extract_metric_value


def ellipsis_center(text: str, max_len: int = 20) -> str:
    """Truncate text with ellipsis in the center, preserving prefix and suffix."""
    if len(text) <= max_len:
        return text
    side_len = (max_len - 1) // 2
    return f"{text[:side_len]}…{text[-side_len:]}"


def get_run_label(samples: list[Any], exp_id: str) -> str:
    """Generate a label for a run from its samples and experiment ID."""
    prefix = exp_id[:6]
    if not samples:
        return prefix

    sample = samples[0]
    parts = []
    if sample.model_name:
        parts.append(ellipsis_center(sample.model_name.split("/")[-1]))
    if sample.provider_kind:
        parts.append(sample.provider_kind)

    return f"({prefix}) {' / '.join(parts)}" if parts else prefix


def find_metrics_with_data(samples_by_exp: dict[str, list[Any]]) -> list[MetricInfo]:
    """Find which metrics have at least one data point in the samples."""
    result = []
    for key, (path, plot_name, table_name) in METRICS.items():
        for samples in samples_by_exp.values():
            if any(extract_metric_value(s, path) is not None for s in samples):
                result.append(MetricInfo(key, path, plot_name, table_name))
                break
    return result


def query_samples(session: Any, filters: QueryFilters) -> dict[str, list[Any]]:
    """Query inference samples with flexible filters, grouped by experiment_id."""
    from sqlalchemy import or_, select

    from olmo_eval.storage.backends.postgres.metrics_models import InferenceSample

    stmt = select(InferenceSample)

    # Apply filters (OR within each filter type, AND across types)
    filter_mappings = [
        (filters.experiment_ids, InferenceSample.experiment_id),
        (filters.experiment_groups, InferenceSample.experiment_group),
        (filters.model_names, InferenceSample.model_name),
        (filters.model_hashes, InferenceSample.model_hash),
        (filters.task_names, InferenceSample.task_name),
        (filters.task_hashes, InferenceSample.task_hash),
    ]

    for values, column in filter_mappings:
        if values:
            stmt = stmt.where(or_(*[column.startswith(v) for v in values]))

    stmt = stmt.order_by(InferenceSample.timestamp)
    samples = list(session.execute(stmt).scalars().all())

    # Group by experiment_id
    result: dict[str, list[Any]] = {}
    for sample in samples:
        result.setdefault(sample.experiment_id or "unknown", []).append(sample)
    return result


def extract_series_data(
    samples_by_exp: dict[str, list[Any]],
) -> dict[str, dict[str, list[float]]]:
    """Extract time series data for all metrics and experiments."""
    result: dict[str, dict[str, list[float]]] = {}
    for exp_id, samples in samples_by_exp.items():
        label = get_run_label(samples, exp_id)
        result[label] = {}
        for key, (path, _, _) in METRICS.items():
            values = [v for s in samples if (v := extract_metric_value(s, path)) is not None]
            if values:
                result[label][key] = values
    return result

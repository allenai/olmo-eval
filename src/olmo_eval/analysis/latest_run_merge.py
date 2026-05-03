"""Shared helpers for latest-run task-row merges."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LatestTaskRowInput:
    """Normalized task-row payload for latest-run merging."""

    experiment_pk: int
    task_name: str
    task_hash: str | None
    metrics: Any
    primary_metric: str | None
    num_instances: int | None = None


@dataclass(frozen=True, slots=True)
class LatestMergedTaskRow:
    """Merged task row after collapsing repeated runs by model hash."""

    experiment_pk: int
    task_name: str
    task_hash: str | None
    metrics: dict[str, dict[str, Any]]
    primary_metric: str | None
    num_instances: int | None = None


def merge_metric_values(
    merged_metrics: dict[str, dict[str, Any]],
    metrics: Any,
) -> None:
    """Merge scorer payloads without overwriting newer values already retained."""
    if not isinstance(metrics, dict):
        return
    for metric_name, scorer_values in metrics.items():
        if not isinstance(metric_name, str) or not isinstance(scorer_values, dict):
            continue
        merged_scorers = merged_metrics.setdefault(metric_name, {})
        for scorer_name, score in scorer_values.items():
            if not isinstance(scorer_name, str) or scorer_name in merged_scorers:
                continue
            merged_scorers[scorer_name] = score


def merge_latest_task_rows(
    *,
    task_rows: Sequence[LatestTaskRowInput],
    source_experiments: Sequence[Any],
    display_experiments: Sequence[Any],
) -> list[LatestMergedTaskRow]:
    """Collapse repeated runs to the latest compatible task row per model hash/task."""
    from olmo_eval.runners.processing.utils import extract_score_from_metrics

    source_experiment_by_pk = {
        int(experiment.id): experiment
        for experiment in source_experiments
        if getattr(experiment, "id", None) is not None
    }
    display_experiment_by_hash = {
        str(experiment.model_hash or ""): experiment
        for experiment in display_experiments
        if getattr(experiment, "model_hash", None)
    }
    display_order = {
        int(experiment.id): index for index, experiment in enumerate(display_experiments)
    }

    enriched_rows: list[tuple[int, str, str, float, int, LatestTaskRowInput]] = []
    for row in task_rows:
        source_experiment = source_experiment_by_pk.get(int(row.experiment_pk))
        if source_experiment is None:
            continue
        model_hash = str(getattr(source_experiment, "model_hash", "") or "")
        display_experiment = display_experiment_by_hash.get(model_hash)
        if display_experiment is None:
            continue
        source_timestamp = getattr(source_experiment, "timestamp", None)
        timestamp_value = (
            float(source_timestamp.timestamp()) if source_timestamp is not None else float("-inf")
        )
        resolved_name = str(row.task_name or "")
        resolved_hash = str(row.task_hash) if row.task_hash else None
        enriched_rows.append(
            (
                int(display_experiment.id),
                resolved_name,
                str(resolved_hash or ""),
                -timestamp_value,
                -int(row.experiment_pk),
                LatestTaskRowInput(
                    experiment_pk=int(row.experiment_pk),
                    task_name=resolved_name,
                    task_hash=resolved_hash,
                    metrics=row.metrics,
                    primary_metric=str(row.primary_metric) if row.primary_metric else None,
                    num_instances=row.num_instances,
                ),
            )
        )

    enriched_rows.sort()

    merged_rows_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for display_pk, task_name, _, _, _, row in enriched_rows:
        task_id = str(row.task_hash or task_name or "")
        merged_row = merged_rows_by_key.setdefault(
            (display_pk, task_id),
            {
                "experiment_pk": display_pk,
                "task_name": task_name,
                "task_hash": row.task_hash,
                "num_instances": 0,
                "num_instances_seen": False,
                "metrics": {},
                "primary_metric_candidates": [],
            },
        )
        if row.num_instances is not None:
            merged_row["num_instances"] = max(
                int(merged_row["num_instances"]),
                int(row.num_instances),
            )
            merged_row["num_instances_seen"] = True
        merge_metric_values(merged_row["metrics"], row.metrics)
        if row.primary_metric and row.primary_metric not in merged_row["primary_metric_candidates"]:
            merged_row["primary_metric_candidates"].append(row.primary_metric)

    merged_rows: list[LatestMergedTaskRow] = []
    for merged_row in merged_rows_by_key.values():
        merged_metrics = dict(merged_row["metrics"])
        primary_metric_candidates = list(merged_row["primary_metric_candidates"])
        primary_metric = next(
            (
                candidate
                for candidate in primary_metric_candidates
                if extract_score_from_metrics(merged_metrics, candidate) is not None
            ),
            primary_metric_candidates[0] if primary_metric_candidates else None,
        )
        merged_rows.append(
            LatestMergedTaskRow(
                experiment_pk=int(merged_row["experiment_pk"]),
                task_name=str(merged_row["task_name"]),
                task_hash=str(merged_row["task_hash"]) if merged_row["task_hash"] else None,
                num_instances=(
                    int(merged_row["num_instances"]) if merged_row["num_instances_seen"] else None
                ),
                metrics=merged_metrics,
                primary_metric=primary_metric,
            )
        )

    merged_rows.sort(
        key=lambda row: (
            display_order.get(int(row.experiment_pk), math.inf),
            row.task_name,
            row.task_hash or "",
        )
    )
    return merged_rows

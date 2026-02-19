"""Formatters for converting evaluation results to various output formats."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from typing import Any

from olmo_eval.cli.results.display import _build_model_task_scores
from olmo_eval.cli.results.transformers import group_experiments_by_model
from olmo_eval.storage.formatters import (
    ComparisonModelOutput,
    ComparisonOutput,
    ComparisonTaskOutput,
    ExperimentOutput,
    ExperimentsOutput,
    ExperimentTaskOutput,
    InstanceOutput,
    PaginationOutput,
)


def _filter_none_instances(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively remove 'instances' keys with None values from dict."""
    result = {}
    for k, v in d.items():
        if k == "instances" and v is None:
            continue
        elif isinstance(v, dict):
            result[k] = _filter_none_instances(v)
        elif isinstance(v, list):
            result[k] = [
                _filter_none_instances(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            result[k] = v
    return result


def experiments_to_dict(
    experiments: list[Any],
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    include_instances: bool = False,
) -> dict[str, Any]:
    """Convert experiments to dict with optional instances grouped by task.

    Args:
        experiments: List of experiment results.
        instances: Optional list of instance predictions to include.
        limit: The limit used for querying instances (for pagination metadata).
        include_instances: Whether user requested instances (controls key presence in output).

    Returns:
        Dict with experiments containing tasks containing instances.
    """
    # Group instances by task_hash
    instance_groups: dict[str, list[InstanceOutput]] = {}
    last_id: int | None = None
    if instances:
        for inst in instances:
            inst_id = inst.get("id")
            if inst_id is not None:
                last_id = inst_id
            key = inst.get("task_hash", "")
            if key not in instance_groups:
                instance_groups[key] = []
            instance_groups[key].append(
                InstanceOutput(
                    native_id=inst.get("native_id", ""),
                    metrics=inst.get("instance_metrics", {}),
                )
            )

    experiment_outputs = []
    for exp in experiments:
        tasks = []
        for t in exp.tasks:
            task_instances = instance_groups.get(t.task_hash) if t.task_hash else None
            tasks.append(
                ExperimentTaskOutput(
                    task_name=t.task_name,
                    task_hash=t.task_hash,
                    primary_metric=t.primary_metric,
                    num_instances=t.num_instances,
                    metrics=t.metrics,
                    instances=task_instances,
                )
            )

        experiment_outputs.append(
            ExperimentOutput(
                experiment_id=exp.experiment_id,
                model_name=exp.model_name,
                model_hash=exp.model_hash,
                backend_name=exp.backend_name,
                timestamp=exp.timestamp.isoformat() if exp.timestamp else None,
                experiment_name=exp.experiment_name,
                workspace=exp.workspace,
                author=exp.author,
                tags=exp.tags,
                git_ref=exp.git_ref,
                revision=exp.revision,
                s3_location=exp.s3_location,
                tasks=tasks,
            )
        )

    pagination = None
    if instances is not None:
        has_more = limit is not None and len(instances) >= limit
        pagination = PaginationOutput(last_id=last_id, has_more=has_more)

    output = ExperimentsOutput(experiments=experiment_outputs, pagination=pagination)
    result = asdict(output)

    # Remove instances keys if user didn't request them
    if not include_instances:
        result = _filter_none_instances(result)

    return result


def experiments_to_json(
    experiments: list[Any],
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    include_instances: bool = False,
) -> str:
    """Convert experiments to JSON string."""
    return json.dumps(
        experiments_to_dict(experiments, instances, limit, include_instances), indent=2
    )


def experiments_to_csv(experiments: list[Any]) -> None:
    """Write experiments to stdout as CSV."""
    writer = csv.writer(sys.stdout)
    writer.writerow(["experiment_id", "model_name", "backend_name", "timestamp", "task_count"])
    for exp in experiments:
        writer.writerow(
            [
                exp.experiment_id,
                exp.model_name,
                exp.backend_name or "",
                exp.timestamp.isoformat() if exp.timestamp else "",
                len(exp.tasks),
            ]
        )


def task_comparison_to_csv(experiments: list[Any], task_filter: set[str] | None = None) -> None:
    """Write task comparison matrix to stdout as CSV.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.
    """
    task_infos, model_scores = _build_model_task_scores(experiments, task_filter)

    if not task_infos:
        return

    writer = csv.writer(sys.stdout)

    # Header row - use task names from task_infos
    sorted_task_names = [t.task_name for t in task_infos]
    writer.writerow(["model"] + sorted_task_names)

    # Data rows
    for model_key in sorted(model_scores.keys()):
        scores = model_scores[model_key]
        row = [model_key]
        for task_info in task_infos:
            score = scores.get(task_info.column_key)
            row.append(f"{score:.4f}" if score is not None else "")
        writer.writerow(row)


def task_comparison_to_dict(
    experiments: list[Any],
    task_filter: set[str] | None = None,
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    include_instances: bool = False,
) -> dict[str, Any]:
    """Convert task comparison to dict with optional instances grouped by model-task.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.
        instances: Optional list of instance predictions to include.
        limit: The limit used for querying instances (for pagination metadata).
        include_instances: Whether user requested instances (controls key presence in output).

    Returns:
        Dict with models containing tasks containing scores and instances.
    """
    # Group instances by (model_hash, task_hash)
    instance_groups: dict[tuple[str, str], list[InstanceOutput]] = {}
    last_id: int | None = None
    if instances:
        for inst in instances:
            inst_id = inst.get("id")
            if inst_id is not None:
                last_id = inst_id
            key = (inst.get("model_hash", ""), inst.get("task_hash", ""))
            if key not in instance_groups:
                instance_groups[key] = []
            instance_groups[key].append(
                InstanceOutput(
                    native_id=inst.get("native_id", ""),
                    metrics=inst.get("instance_metrics", {}),
                )
            )

    grouped_models = group_experiments_by_model(experiments, task_filter)

    models = []
    for grouped in grouped_models:
        tasks = []
        for task_run in grouped.task_runs:
            key = (grouped.model_hash or "", task_run.task_hash or "")
            task_instances = instance_groups.get(key)

            tasks.append(
                ComparisonTaskOutput(
                    task_name=task_run.task_name,
                    task_hash=task_run.task_hash,
                    primary_metric=task_run.primary_metric,
                    timestamp=task_run.timestamp,
                    metrics=task_run.metrics,
                    instances=task_instances,
                )
            )

        models.append(
            ComparisonModelOutput(
                model_name=grouped.model_name,
                model_hash=grouped.model_hash,
                tasks=tasks,
            )
        )

    pagination = None
    if instances is not None:
        has_more = limit is not None and len(instances) >= limit
        pagination = PaginationOutput(last_id=last_id, has_more=has_more)

    output = ComparisonOutput(models=models, pagination=pagination)
    result = asdict(output)

    # Remove instances keys if user didn't request them
    if not include_instances:
        result = _filter_none_instances(result)

    return result


def task_comparison_to_json(
    experiments: list[Any],
    task_filter: set[str] | None = None,
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    include_instances: bool = False,
) -> str:
    """Convert task comparison to JSON string."""
    return json.dumps(
        task_comparison_to_dict(experiments, task_filter, instances, limit, include_instances),
        indent=2,
    )


def instances_to_json(instances: list[dict[str, Any]]) -> str:
    """Convert instances to JSON string."""
    return json.dumps(instances, indent=2)


def instances_to_csv(instances: list[dict[str, Any]]) -> None:
    """Write instances to stdout as CSV."""
    writer = csv.writer(sys.stdout)
    writer.writerow(["native_id", "task_name", "task_hash", "metrics"])
    for inst in instances:
        metrics_str = json.dumps(inst.get("instance_metrics", {}))
        writer.writerow(
            [
                inst.get("native_id", ""),
                inst.get("task_name", ""),
                inst.get("task_hash", ""),
                metrics_str,
            ]
        )

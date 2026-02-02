"""Data transformers for results processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _matches_prefix_filter(value: str, prefixes: set[str]) -> bool:
    """Check if value starts with any of the given prefixes."""
    return any(value.startswith(prefix) for prefix in prefixes)


@dataclass
class TaskRun:
    """A single task run with its metadata."""

    task_name: str
    task_hash: str | None
    primary_metric: str | None
    metrics: dict[str, dict[str, float]] | None
    timestamp: str | None
    num_instances: int | None = None


@dataclass
class GroupedModel:
    """Model with all its task runs grouped."""

    model_name: str
    model_hash: str | None
    task_runs: list[TaskRun]


def group_experiments_by_model(
    experiments: list[Any],
    task_filter: set[str] | None = None,
) -> list[GroupedModel]:
    """Group experiments by (model_name, model_hash), collecting all task runs.

    Each task run is kept separate even if it has the same task_hash
    (different experiments = different runs).

    Args:
        experiments: List of experiment results (EvalResult or similar).
        task_filter: Optional set of task names to include.

    Returns:
        List of GroupedModel with all task runs for each model.
    """
    model_tasks: dict[tuple[str, str | None], list[TaskRun]] = {}

    for exp in experiments:
        model_key = (exp.model_name, exp.model_hash)

        if model_key not in model_tasks:
            model_tasks[model_key] = []

        for task in exp.tasks:
            if task_filter and not _matches_prefix_filter(task.task_name, task_filter):
                continue

            model_tasks[model_key].append(
                TaskRun(
                    task_name=task.task_name,
                    task_hash=task.task_hash,
                    primary_metric=task.primary_metric,
                    metrics=task.metrics,
                    timestamp=exp.timestamp.isoformat() if exp.timestamp else None,
                    num_instances=getattr(task, "num_instances", None),
                )
            )

    return [
        GroupedModel(
            model_name=model_name,
            model_hash=model_hash,
            task_runs=task_runs,
        )
        for (model_name, model_hash), task_runs in model_tasks.items()
    ]

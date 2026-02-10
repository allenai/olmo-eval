"""Unified streaming formatters for evaluation results output."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import asdict, astuple, dataclass, field, fields
from itertools import groupby
from operator import attrgetter
from typing import IO, Any


@dataclass
class InstanceOutput:
    """Single instance prediction for output."""

    native_id: str
    metrics: dict[str, dict[str, float]]  # Nested: {metric: {scorer: value}}


@dataclass
class TaskOutput:
    """Task with its instances for output."""

    task_name: str
    task_hash: str
    instances: list[InstanceOutput] = field(default_factory=list)


@dataclass
class ModelOutput:
    """Model with its tasks for output."""

    model_name: str
    model_hash: str
    tasks: list[TaskOutput] = field(default_factory=list)


@dataclass
class PaginationOutput:
    """Pagination metadata."""

    last_id: int | None = None
    has_more: bool | None = None


@dataclass
class InstanceStreamOutput:
    """Top-level output for instance streaming."""

    models: list[ModelOutput]
    experiment_group: str | None = None
    pagination: PaginationOutput | None = None


@dataclass
class InstanceCSVRow:
    """Single row for CSV output - defines column order."""

    experiment_group: str
    model_name: str
    model_hash: str
    task_name: str
    task_hash: str
    native_id: str
    metrics: str  # JSON string


# --- Experiment query output dataclasses ---


@dataclass
class ExperimentTaskOutput:
    """Task result within an experiment.

    Note: primary_metric is in "metric:scorer" format. The primary_score can be
    extracted from metrics[metric][scorer] using the primary_metric identifier.
    """

    task_name: str
    task_hash: str | None
    primary_metric: str | None  # Format: "metric:scorer"
    num_instances: int | None
    metrics: dict[str, dict[str, float]] | None  # Nested: {metric: {scorer: value}}
    instances: list[InstanceOutput] | None = None


@dataclass
class ExperimentOutput:
    """Full experiment with metadata and tasks."""

    experiment_id: str
    model_name: str
    model_hash: str | None
    backend_name: str | None
    timestamp: str | None  # ISO format string
    experiment_name: str | None
    workspace: str | None
    author: str | None
    tags: list[str] | None
    git_ref: str | None
    revision: str | None
    s3_location: str | None
    tasks: list[ExperimentTaskOutput] = field(default_factory=list)


@dataclass
class ExperimentsOutput:
    """Top-level output for experiment queries."""

    experiments: list[ExperimentOutput]
    pagination: PaginationOutput | None = None


# --- Comparison query output dataclasses ---


@dataclass
class ComparisonTaskOutput:
    """Task result for comparison output (simpler than ExperimentTaskOutput).

    Note: primary_metric is in "metric:scorer" format.
    """

    task_name: str
    task_hash: str | None
    primary_metric: str | None  # Format: "metric:scorer"
    timestamp: str | None = None  # ISO format string (from parent experiment)
    metrics: dict[str, dict[str, float]] | None = None  # Nested: {metric: {scorer: value}}
    instances: list[InstanceOutput] | None = None


@dataclass
class ComparisonModelOutput:
    """Model with tasks for comparison output."""

    model_name: str
    model_hash: str | None
    tasks: list[ComparisonTaskOutput] = field(default_factory=list)


@dataclass
class ComparisonOutput:
    """Top-level output for task comparison queries."""

    models: list[ComparisonModelOutput]
    pagination: PaginationOutput | None = None


# Use field names for headers
CSV_HEADERS = [f.name for f in fields(InstanceCSVRow)]


def stream_instances_to_csv(
    instances: Iterator[Any],
    output: IO[str],
    experiment_group: str | None = None,
) -> None:
    """Stream instances directly to CSV. Memory-efficient for any size.

    Args:
        instances: Iterator of SQLAlchemy Row objects with instance and metadata fields.
        output: File-like object to write CSV output to.
        experiment_group: Optional experiment group to include in each row.
    """
    writer = csv.writer(output)
    writer.writerow(CSV_HEADERS)

    for row in instances:
        csv_row = InstanceCSVRow(
            experiment_group=experiment_group or getattr(row, "experiment_group", ""),
            model_name=row.model_name,
            model_hash=row.model_hash,
            task_name=row.task_name,
            task_hash=row.task_hash,
            native_id=row.native_id,
            metrics=json.dumps(row.instance_metrics),
        )
        writer.writerow(astuple(csv_row))


def stream_instances_to_nested_json(
    instances: Iterator[Any],
    output: IO[str],
    experiment_group: str | None = None,
) -> None:
    """Stream instances to nested JSON. Groups in single pass.

    Uses ordered grouping - assumes input is sorted by model_hash, task_hash.

    Args:
        instances: Iterator of SQLAlchemy Row objects sorted by (model_hash, task_hash, id).
        output: File-like object to write JSON output to.
        experiment_group: Optional experiment group to include in output.
    """
    models: list[ModelOutput] = []
    last_id: int | None = None

    for model_hash, model_group in groupby(instances, key=attrgetter("model_hash")):
        model_rows = list(model_group)
        model = ModelOutput(
            model_name=model_rows[0].model_name,
            model_hash=model_hash,
        )

        for task_hash, task_group in groupby(model_rows, key=attrgetter("task_hash")):
            task_rows = list(task_group)
            task = TaskOutput(
                task_name=task_rows[0].task_name,
                task_hash=task_hash,
                instances=[
                    InstanceOutput(
                        native_id=row.native_id,
                        metrics=row.instance_metrics,
                    )
                    for row in task_rows
                ],
            )
            model.tasks.append(task)
            last_id = task_rows[-1].id

        models.append(model)

    output_data = InstanceStreamOutput(
        models=models,
        experiment_group=experiment_group,
        pagination=PaginationOutput(last_id=last_id) if last_id else None,
    )

    json.dump(asdict(output_data), output, indent=2, default=str)

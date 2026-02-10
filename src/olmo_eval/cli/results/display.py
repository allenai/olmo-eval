"""Display functions for rendering evaluation results."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from olmo_eval.cli.results.transformers import _matches_prefix_filter
from olmo_eval.cli.utils import console, format_timestamp
from olmo_eval.runners.processing.utils import (
    extract_score_from_metrics,
    parse_metric_key,
)


def print_experiment_detail(experiment: Any) -> None:
    """Print detailed information about an experiment."""
    # Required fields (always shown)
    lines = [
        ("Experiment ID", experiment.experiment_id),
        ("Model", experiment.model_name),
        ("Backend", experiment.backend_name or "-"),
        ("Timestamp", format_timestamp(experiment.timestamp)),
    ]

    # Optional fields (shown only if truthy)
    optional = [
        ("Name", experiment.experiment_name),
        ("Model Hash", experiment.model_hash),
        ("Workspace", experiment.workspace),
        ("Author", experiment.author),
        ("Tags", ", ".join(experiment.tags) if experiment.tags else None),
        ("Git Ref", experiment.git_ref),
        ("Revision", experiment.revision),
    ]
    lines.extend((label, value) for label, value in optional if value)

    formatted = [f"[bold]{label}:[/bold] {value}" for label, value in lines]
    console.print(Panel("\n".join(formatted), title="Experiment Details", expand=False))


def print_task_results_table(tasks: list[Any], task_filter: set[str] | None = None) -> None:
    """Print a table of task results."""
    table = Table(title="Task Results")
    table.add_column("Task", style="cyan")
    table.add_column("Primary Metric", style="dim")
    table.add_column("Score", justify="right", style="green")

    for task in tasks:
        # Apply filter if provided
        if task_filter and not _matches_prefix_filter(task.task_name, task_filter):
            continue

        # Extract primary score from nested metrics
        primary_score = extract_score_from_metrics(task.metrics, task.primary_metric)
        score_str = f"{primary_score:.4f}" if primary_score is not None else "-"

        table.add_row(
            task.task_name,
            task.primary_metric or "-",
            score_str,
        )

    console.print(table)


@dataclass
class TaskDisplayInfo:
    """Display information for a task column."""

    task_name: str
    task_hash: str
    primary_metric: str | None
    scorer: str | None

    @property
    def column_key(self) -> tuple[str, str]:
        """Return the unique key for this task (name, hash)."""
        return (self.task_name, self.task_hash)

    @property
    def column_header(self) -> str:
        """Build the column header with metric/scorer info."""
        # Build metric/scorer part - escape brackets for Rich markup
        if self.primary_metric and self.scorer:
            metric_text = f"[{self.primary_metric}:{self.scorer}]"
            metric_part = f" [dim]{escape(metric_text)}[/dim]"
        elif self.primary_metric:
            metric_part = f" [dim]{escape(f'[{self.primary_metric}]')}[/dim]"
        else:
            metric_part = ""

        # Build hash part
        short_hash = self.task_hash[:4] if self.task_hash else ""
        hash_part = f" [dim]({short_hash})[/dim]" if short_hash else ""

        return f"{self.task_name}{metric_part}{hash_part}"


def _build_model_task_scores(
    experiments: list[Any], task_filter: set[str] | None = None
) -> tuple[list[TaskDisplayInfo], dict[str, dict[tuple[str, str], float | None]]]:
    """Build model-task score mapping from experiments.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.

    Returns:
        Tuple of (task_infos, model_scores) where:
        - task_infos: List of TaskDisplayInfo for each unique (task_name, task_hash)
        - model_scores: Maps model_key -> (task_name, task_hash) -> score
    """
    # Track unique tasks by (task_name, task_hash)
    task_info_map: dict[tuple[str, str], TaskDisplayInfo] = {}
    model_scores: dict[str, dict[tuple[str, str], float | None]] = {}

    for exp in experiments:
        model_key = exp.model_name
        if exp.model_hash:
            model_key += f" [dim]({exp.model_hash[:4]})[/dim]"

        if model_key not in model_scores:
            model_scores[model_key] = {}

        for task in exp.tasks:
            if task_filter and not _matches_prefix_filter(task.task_name, task_filter):
                continue

            task_hash = task.task_hash or ""
            task_key = (task.task_name, task_hash)

            # Extract metric and scorer from primary_metric "metric:scorer" format
            metric_name = None
            scorer_name = None
            parsed = parse_metric_key(task.primary_metric) if task.primary_metric else None
            if parsed:
                metric_name, scorer_name = parsed
            elif task.primary_metric:
                metric_name = task.primary_metric

            # Build task display info if not already seen
            if task_key not in task_info_map:
                task_info_map[task_key] = TaskDisplayInfo(
                    task_name=task.task_name,
                    task_hash=task_hash,
                    primary_metric=metric_name,
                    scorer=scorer_name,
                )

            # Extract primary score from nested metrics
            primary_score = extract_score_from_metrics(task.metrics, task.primary_metric)
            model_scores[model_key][task_key] = primary_score

    # Sort by (task_name, task_hash) for consistent ordering
    sorted_task_infos = sorted(task_info_map.values(), key=lambda t: t.column_key)

    return sorted_task_infos, model_scores


def print_task_comparison_matrix(
    experiments: list[Any], task_filter: set[str] | None = None
) -> None:
    """Print a comparison matrix with models as rows and tasks as columns.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.
    """
    task_infos, model_scores = _build_model_task_scores(experiments, task_filter)

    if not task_infos:
        console.print("[dim]No matching tasks found.[/dim]")
        return

    # Create the comparison table
    table = Table(title="Results")
    table.add_column("Model", style="cyan")

    for task_info in task_infos:
        table.add_column(task_info.column_header, justify="right")

    # Add rows for each model
    for model_key in sorted(model_scores.keys()):
        scores = model_scores[model_key]
        row = [model_key]
        for task_info in task_infos:
            score = scores.get(task_info.column_key)
            if score is not None:
                row.append(f"{score:.4f}")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


def print_experiment_summary(experiments: list[Any]) -> None:
    """Print a unified experiment summary grouping by experiment_id.

    Shows shared experiment details once, then lists models and tasks.
    """
    # Group experiments by experiment_id
    by_exp_id: dict[str, list[Any]] = defaultdict(list)
    for exp in experiments:
        by_exp_id[exp.experiment_id].append(exp)

    for exp_id, exp_group in by_exp_id.items():
        first_exp = exp_group[0]

        # Build unified summary table
        table = Table(
            title=f"Experiment: {exp_id}",
            show_header=False,
            box=None,
            padding=(0, 2),
            collapse_padding=True,
        )
        table.add_column("Field", style="bold", width=12)
        table.add_column("Value")

        # Global experiment fields
        if first_exp.experiment_name:
            table.add_row("Name", first_exp.experiment_name)
        if first_exp.experiment_group:
            table.add_row("Group", first_exp.experiment_group)
        if first_exp.workspace:
            table.add_row("Workspace", first_exp.workspace)
        if first_exp.author:
            table.add_row("Author", first_exp.author)
        if first_exp.timestamp:
            table.add_row("Timestamp", format_timestamp(first_exp.timestamp))
        if first_exp.git_ref:
            table.add_row("Git Ref", first_exp.git_ref)
        if first_exp.revision and first_exp.revision != "unknown":
            table.add_row("Revision", first_exp.revision)
        if first_exp.tags:
            table.add_row("Tags", ", ".join(first_exp.tags))
        if first_exp.s3_location:
            table.add_row("S3 Location", first_exp.s3_location)

        # Models section
        table.add_row("", "")  # Spacer
        table.add_row("Models", f"[dim]({len(exp_group)} total)[/dim]")
        for exp in exp_group:
            hash_str = f"[dim]({exp.model_hash[:4]})[/dim]" if exp.model_hash else ""
            table.add_row("", f"  {exp.model_name} {hash_str}")

        # Tasks section - collect unique tasks across all models
        tasks_seen: dict[str, str] = {}  # task_name -> task_hash
        for exp in exp_group:
            for task in exp.tasks:
                if task.task_name not in tasks_seen:
                    tasks_seen[task.task_name] = task.task_hash or ""

        if tasks_seen:
            table.add_row("", "")  # Spacer
            table.add_row("Tasks", f"[dim]({len(tasks_seen)} total)[/dim]")
            for task_name in sorted(tasks_seen.keys()):
                task_hash = tasks_seen[task_name]
                hash_str = f"[dim]({task_hash[:4]})[/dim]" if task_hash else ""
                table.add_row("", f"  {task_name} {hash_str}")

        console.print(table)
        console.print()


def print_experiments_table(experiments: list[Any], task_filter: set[str] | None) -> None:
    """Print experiments in table format with details."""
    # Use unified summary view for experiment queries
    print_experiment_summary(experiments)

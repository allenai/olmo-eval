"""Display functions for rendering evaluation results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rich.panel import Panel
from rich.table import Table

from olmo_eval.cli.utils import console, format_timestamp


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
        if task_filter and task.task_name not in task_filter:
            continue

        score_str = f"{task.primary_score:.4f}" if task.primary_score is not None else "-"

        table.add_row(
            task.task_name,
            task.primary_metric or "-",
            score_str,
        )

    console.print(table)


def _build_model_task_scores(
    experiments: list[Any], task_filter: set[str] | None = None
) -> tuple[list[str], dict[str, dict[str, float | None]], dict[str, str]]:
    """Build model-task score mapping from experiments.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.

    Returns:
        Tuple of (sorted_tasks, model_scores, task_hashes) where model_scores maps
        model_key -> task_name -> score, and task_hashes maps task_name -> short_hash.
    """
    all_tasks: set[str] = set()
    model_scores: dict[str, dict[str, float | None]] = {}
    task_hashes: dict[str, str] = {}

    for exp in experiments:
        model_key = exp.model_name
        if exp.model_hash:
            model_key += f" [dim]({exp.model_hash[-4:]})[/dim]"

        if model_key not in model_scores:
            model_scores[model_key] = {}

        for task in exp.tasks:
            if task_filter and task.task_name not in task_filter:
                continue
            all_tasks.add(task.task_name)
            # Keep the latest score if we see duplicates
            model_scores[model_key][task.task_name] = task.primary_score
            # Store task hash (use latest if multiple)
            if task.task_hash:
                task_hashes[task.task_name] = task.task_hash[-4:]

    return sorted(all_tasks), model_scores, task_hashes


def print_task_comparison_matrix(
    experiments: list[Any], task_filter: set[str] | None = None
) -> None:
    """Print a comparison matrix with models as rows and tasks as columns.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.
    """
    sorted_tasks, model_scores, task_hashes = _build_model_task_scores(experiments, task_filter)

    if not sorted_tasks:
        console.print("[dim]No matching tasks found.[/dim]")
        return

    # Create the comparison table
    table = Table(title="Results")
    table.add_column("Model", style="cyan")

    for task_name in sorted_tasks:
        # Include short hash in column header if available (dimmed)
        short_hash = task_hashes.get(task_name)
        header = f"{task_name} [dim]({short_hash})[/dim]" if short_hash else task_name
        table.add_column(header, justify="right")

    # Add rows for each model
    for model_key in sorted(model_scores.keys()):
        scores = model_scores[model_key]
        row = [model_key]
        for task_name in sorted_tasks:
            score = scores.get(task_name)
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
            hash_str = f"[dim]({exp.model_hash[:8]})[/dim]" if exp.model_hash else ""
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
                hash_str = f"[dim]({task_hash[:8]})[/dim]" if task_hash else ""
                table.add_row("", f"  {task_name} {hash_str}")

        console.print(table)
        console.print()


def print_experiments_table(experiments: list[Any], task_filter: set[str] | None) -> None:
    """Print experiments in table format with details."""
    # Use unified summary view for experiment queries
    print_experiment_summary(experiments)

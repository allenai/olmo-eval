"""CLI commands for querying and displaying evaluation results."""

from __future__ import annotations

import csv
import functools
import json
import sys
from datetime import datetime
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from olmo_eval.cli.utils import console


def db_options(func: Any) -> Any:
    """Decorator that adds common database connection options to a command."""

    @click.option(
        "--db-host",
        envvar="OLMO_EVAL_DB_HOST",
        default="localhost",
        help="Database host.",
    )
    @click.option(
        "--db-port",
        envvar="OLMO_EVAL_DB_PORT",
        default=5432,
        type=int,
        help="Database port.",
    )
    @click.option(
        "--db-name",
        envvar="OLMO_EVAL_DB_NAME",
        default="olmo_eval",
        help="Database name.",
    )
    @click.option(
        "--db-user",
        envvar="OLMO_EVAL_DB_USER",
        default="postgres",
        help="Database user.",
    )
    @click.option(
        "--db-password-env",
        default="OLMO_EVAL_DB_PASSWORD",
        help="Environment variable containing database password.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


def get_database_session(
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password_env: str,
) -> Any:
    """Create and initialize a DatabaseSession.

    Returns:
        Initialized DatabaseSession instance.

    Raises:
        SystemExit: If psycopg is not installed.
    """
    try:
        from olmo_eval.storage.db.session import DatabaseSession
    except ImportError:
        console.print(
            "[red]Error:[/red] Database support requires psycopg. "
            "Install with: pip install psycopg[binary]"
        )
        raise SystemExit(1) from None

    db = DatabaseSession(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password_env=db_password_env,
    )
    db.initialize()
    return db


def format_timestamp(ts: datetime | None) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def print_experiments_table(experiments: list[Any]) -> None:
    """Print a table of experiments."""
    table = Table(title="Evaluation Results")
    table.add_column("Experiment ID", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Backend", style="dim")
    table.add_column("Timestamp")
    table.add_column("Tasks", justify="right")

    for exp in experiments:
        table.add_row(
            exp.experiment_id,
            exp.model_name,
            exp.backend_name or "-",
            format_timestamp(exp.timestamp),
            str(len(exp.tasks)),
        )

    console.print(table)


def print_experiment_detail(experiment: Any) -> None:
    """Print detailed information about an experiment."""
    # Header panel with metadata
    lines = [
        f"[bold]Experiment ID:[/bold] {experiment.experiment_id}",
        f"[bold]Model:[/bold] {experiment.model_name}",
        f"[bold]Backend:[/bold] {experiment.backend_name or '-'}",
        f"[bold]Timestamp:[/bold] {format_timestamp(experiment.timestamp)}",
    ]

    if experiment.experiment_name:
        lines.append(f"[bold]Name:[/bold] {experiment.experiment_name}")
    if experiment.model_hash:
        lines.append(f"[bold]Model Hash:[/bold] {experiment.model_hash}")
    if experiment.workspace:
        lines.append(f"[bold]Workspace:[/bold] {experiment.workspace}")
    if experiment.author:
        lines.append(f"[bold]Author:[/bold] {experiment.author}")
    if experiment.tags:
        lines.append(f"[bold]Tags:[/bold] {', '.join(experiment.tags)}")
    if experiment.git_ref:
        lines.append(f"[bold]Git Ref:[/bold] {experiment.git_ref}")
    if experiment.revision:
        lines.append(f"[bold]Revision:[/bold] {experiment.revision}")

    console.print(Panel("\n".join(lines), title="Experiment Details", expand=False))


def print_task_results_table(tasks: list[Any], task_filter: tuple[str, ...] | None = None) -> None:
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


def print_instances_table(instances: list[dict[str, Any]]) -> None:
    """Print a table of instance predictions."""
    table = Table(title="Instance Predictions")
    table.add_column("Doc", justify="right", style="dim")
    table.add_column("Native ID", style="cyan")
    table.add_column("Task")
    table.add_column("Metrics")

    for inst in instances:
        # Format metrics as key: value pairs
        metrics = inst.get("instance_metrics", {})
        if metrics:
            metrics_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
        else:
            metrics_str = "-"

        table.add_row(
            str(inst.get("doc_id", "-")),
            str(inst.get("native_id", "-")),
            inst.get("task_name", "-"),
            metrics_str,
        )

    console.print(table)


def experiments_to_json(experiments: list[Any]) -> str:
    """Convert experiments to JSON string."""
    data = []
    for exp in experiments:
        data.append({
            "experiment_id": exp.experiment_id,
            "model_name": exp.model_name,
            "model_hash": exp.model_hash,
            "backend_name": exp.backend_name,
            "timestamp": exp.timestamp.isoformat() if exp.timestamp else None,
            "experiment_name": exp.experiment_name,
            "workspace": exp.workspace,
            "author": exp.author,
            "tags": exp.tags,
            "git_ref": exp.git_ref,
            "revision": exp.revision,
            "s3_location": exp.s3_location,
            "tasks": [
                {
                    "task_name": t.task_name,
                    "task_hash": t.task_hash,
                    "primary_metric": t.primary_metric,
                    "primary_score": t.primary_score,
                    "num_instances": t.num_instances,
                    "metrics": t.metrics,
                }
                for t in exp.tasks
            ],
        })
    return json.dumps(data, indent=2)


def experiments_to_csv(experiments: list[Any]) -> None:
    """Write experiments to stdout as CSV."""
    writer = csv.writer(sys.stdout)
    writer.writerow(["experiment_id", "model_name", "backend_name", "timestamp", "task_count"])
    for exp in experiments:
        writer.writerow([
            exp.experiment_id,
            exp.model_name,
            exp.backend_name or "",
            exp.timestamp.isoformat() if exp.timestamp else "",
            len(exp.tasks),
        ])


def instances_to_json(instances: list[dict[str, Any]]) -> str:
    """Convert instances to JSON string."""
    return json.dumps(instances, indent=2)


def instances_to_csv(instances: list[dict[str, Any]]) -> None:
    """Write instances to stdout as CSV."""
    writer = csv.writer(sys.stdout)
    writer.writerow(["doc_id", "native_id", "task_name", "experiment_id", "model_hash", "metrics"])
    for inst in instances:
        metrics_str = json.dumps(inst.get("instance_metrics", {}))
        writer.writerow([
            inst.get("doc_id", ""),
            inst.get("native_id", ""),
            inst.get("task_name", ""),
            inst.get("experiment_id", ""),
            inst.get("model_hash", ""),
            metrics_str,
        ])


@click.group()
def results() -> None:
    """Query and display evaluation results."""
    pass


@results.command()
@click.argument("experiment_id")
@click.option(
    "--instances/--no-instances",
    default=False,
    help="Include instance-level predictions.",
)
@click.option(
    "--task",
    "-t",
    "task_filter",
    multiple=True,
    help="Filter tasks to display (can specify multiple).",
)
@click.option(
    "--limit",
    "-n",
    default=100,
    type=int,
    help="Limit instances when --instances is used.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format.",
)
@db_options
def get(
    experiment_id: str,
    instances: bool,
    task_filter: tuple[str, ...],
    limit: int,
    output_format: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password_env: str,
) -> None:
    """Get results for a specific experiment.

    EXPERIMENT_ID is the unique identifier of the experiment.

    Example: olmo-eval results get abc123def456
    """
    db = get_database_session(db_host, db_port, db_name, db_user, db_password_env)

    try:
        from olmo_eval.storage.db.repository import ExperimentRepository, InstancePredictionRepository

        with db.session() as session:
            repo = ExperimentRepository(session)
            experiment = repo.get(experiment_id)

            if experiment is None:
                console.print(f"[red]Error:[/red] Experiment '{experiment_id}' not found")
                raise SystemExit(1)

            # Handle output formats
            if output_format == "json":
                print(experiments_to_json([experiment]))
                return

            if output_format == "csv":
                experiments_to_csv([experiment])
                return

            # Table format
            print_experiment_detail(experiment)
            console.print()
            print_task_results_table(experiment.tasks, task_filter if task_filter else None)

            # Optionally show instances
            if instances:
                instance_repo = InstancePredictionRepository(session)
                task_names = list(task_filter) if task_filter else None
                instance_data = instance_repo.get_instances(
                    experiment_id=experiment_id,
                    task_name=task_names,
                    limit=limit,
                )

                if instance_data:
                    console.print()
                    print_instances_table(instance_data)
                else:
                    console.print("\n[dim]No instance predictions found.[/dim]")
    finally:
        db.dispose()


@results.command()
@click.option(
    "--model",
    "-m",
    "model_names",
    multiple=True,
    help="Model name(s) to query (can specify multiple).",
)
@click.option(
    "--model-hash",
    "-h",
    "model_hashes",
    multiple=True,
    help="Model hash(es) to query (can specify multiple).",
)
@click.option(
    "--task",
    "-t",
    "task_names",
    multiple=True,
    required=True,
    help="Task name(s) to display (can specify multiple).",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format.",
)
@db_options
def query(
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    output_format: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password_env: str,
) -> None:
    """Query evaluation results across models and tasks.

    Displays a table with models/hashes as rows and task scores as columns.
    Uses the most recent experiment for each model.

    Examples:
        olmo-eval results query -m llama3.1-8b -m qwen2.5-72b -t mmlu -t gsm8k
        olmo-eval results query -h abc123 -h def456 -t mmlu -t gsm8k
        olmo-eval results query -m olmo-2-13b -t suite:core -t suite:overall
    """
    if not model_names and not model_hashes:
        raise click.UsageError("At least one --model or --model-hash is required")

    db = get_database_session(db_host, db_port, db_name, db_user, db_password_env)

    try:
        from olmo_eval.storage.db.repository import ExperimentRepository

        with db.session() as session:
            repo = ExperimentRepository(session)

            # Each row: (display_label, scores_dict, model_name, model_hash)
            rows: list[tuple[str, dict[str, float | None], str, str | None]] = []
            # Collect task hashes for column headers
            task_hashes: dict[str, str | None] = {task: None for task in task_names}

            def collect_task_hashes(exp: Any) -> None:
                for t in exp.tasks:
                    if t.task_name in task_hashes and t.task_hash and not task_hashes[t.task_name]:
                        task_hashes[t.task_name] = t.task_hash

            # Query by model names - find all distinct configs for each model
            for model_name in model_names:
                experiments = repo.query(model_name=model_name, limit=1000)
                if not experiments:
                    console.print(f"[yellow]Warning:[/yellow] No experiments found for model '{model_name}'")
                    rows.append((model_name, {task: None for task in task_names}, model_name, None))
                else:
                    latest_by_hash: dict[str | None, Any] = {}
                    for exp in experiments:
                        h = exp.model_hash
                        if h not in latest_by_hash:
                            latest_by_hash[h] = exp
                        collect_task_hashes(exp)
                    for exp in latest_by_hash.values():
                        task_scores = {t.task_name: t.primary_score for t in exp.tasks}
                        scores = {task: task_scores.get(task) for task in task_names}
                        rows.append((model_name, scores, exp.model_name, exp.model_hash))

            # Query by model hashes
            for model_hash in model_hashes:
                experiments = repo.query(model_hash=model_hash, latest=True)
                if not experiments:
                    console.print(f"[yellow]Warning:[/yellow] No experiments found for hash '{model_hash}'")
                    rows.append((model_hash, {task: None for task in task_names}, "", model_hash))
                else:
                    exp = experiments[0]
                    collect_task_hashes(exp)
                    task_scores = {t.task_name: t.primary_score for t in exp.tasks}
                    scores = {task: task_scores.get(task) for task in task_names}
                    rows.append((model_hash, scores, exp.model_name, exp.model_hash))

            if not rows:
                console.print("[dim]No results found.[/dim]")
                return

            # Handle output formats
            if output_format == "json":
                # JSON: use model_name + hash suffix as key to handle multiple configs
                data = {}
                for _, scores, model_name, model_hash in rows:
                    hash_suffix = model_hash[-6:] if model_hash else ""
                    key = f"{model_name} {hash_suffix}" if hash_suffix else (model_name or "unknown")
                    data[key] = {"model_hash": model_hash, "scores": scores}
                print(json.dumps(data, indent=2))
                return

            if output_format == "csv":
                writer = csv.writer(sys.stdout)
                writer.writerow(["model", "model_hash"] + list(task_names))
                for _, scores, model_name, model_hash in rows:
                    row = [model_name, model_hash or ""]
                    for task in task_names:
                        score = scores.get(task)
                        row.append(f"{score:.4f}" if score is not None else "")
                    writer.writerow(row)
                return

            max_name_len = max((len(r[2]) for r in rows if r[2]), default=0)
            max_task_len = max(len(t) for t in task_names)

            table = Table(title="Results")
            table.add_column("Model", style="cyan", no_wrap=True)

            for task in task_names:
                task_hash = task_hashes.get(task)
                if task_hash:
                    padded_task = task.ljust(max_task_len)
                    col_name = f"{padded_task} [dim]{task_hash[-6:]}[/dim]"
                else:
                    col_name = task
                table.add_column(col_name, justify="right", style="green")

            for _, scores, model_name, model_hash in rows:
                if model_name and model_hash:
                    padded_name = model_name.ljust(max_name_len)
                    display = f"{padded_name} [dim]{model_hash[-6:]}[/dim]"
                elif model_name:
                    display = model_name
                elif model_hash:
                    display = f"[dim]{model_hash[-6:]}[/dim]"
                else:
                    display = "-"

                row = [display]
                for task in task_names:
                    score = scores.get(task)
                    row.append(f"{score:.4f}" if score is not None else "-")
                table.add_row(*row)

            console.print(table)
    finally:
        db.dispose()


@results.command()
@click.option(
    "--experiment",
    "-e",
    "experiment_id",
    help="Filter by experiment ID.",
)
@click.option(
    "--model",
    "-m",
    "model_name",
    help="Filter by model name.",
)
@click.option(
    "--model-hash",
    help="Filter by model config hash.",
)
@click.option(
    "--task",
    "-t",
    "task_name",
    required=True,
    help="Task name (required).",
)
@click.option(
    "--limit",
    "-n",
    default=100,
    type=int,
    help="Maximum instances to return.",
)
@click.option(
    "--offset",
    default=0,
    type=int,
    help="Skip first N instances.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format.",
)
@db_options
def instances(
    experiment_id: str | None,
    model_name: str | None,
    model_hash: str | None,
    task_name: str,
    limit: int,
    offset: int,
    output_format: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password_env: str,
) -> None:
    """Get instance-level predictions.

    At least one of --experiment, --model, or --model-hash is required along with --task.

    Examples:
        olmo-eval results instances --task mmlu --model llama3.1-8b
        olmo-eval results instances --task mmlu --experiment abc123
    """
    # Validate that at least one identifier is provided
    if not any([experiment_id, model_name, model_hash]):
        raise click.UsageError(
            "At least one of --experiment, --model, or --model-hash is required"
        )

    db = get_database_session(db_host, db_port, db_name, db_user, db_password_env)

    try:
        from olmo_eval.storage.db.queries import QueryHelper

        with db.session() as session:
            helper = QueryHelper(session)

            instance_data = helper.get_model_task_instances(
                task_name=task_name,
                model_name=model_name,
                model_hash=model_hash,
                experiment_id=experiment_id,
                limit=limit,
                offset=offset,
            )

            if not instance_data:
                console.print("[dim]No instances found matching filters.[/dim]")
                return

            # Handle output formats
            if output_format == "json":
                print(instances_to_json(instance_data))
                return

            if output_format == "csv":
                instances_to_csv(instance_data)
                return

            # Table format
            print_instances_table(instance_data)
            console.print(f"\n[dim]Showing {len(instance_data)} instance(s)[/dim]")
    finally:
        db.dispose()

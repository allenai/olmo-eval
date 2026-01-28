"""CLI commands for querying and displaying evaluation results."""

from __future__ import annotations

import csv
import functools
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from olmo_eval.cli.utils import console


def s3_options(func: Any) -> Any:
    """Decorator that adds common S3 connection options to a command."""

    @click.option(
        "--s3-endpoint-url",
        envvar="S3_ENDPOINT_URL",
        default=None,
        help="S3 endpoint URL (for LocalStack or S3-compatible services).",
    )
    @click.option(
        "--s3-region",
        envvar="AWS_REGION",
        default="us-east-1",
        help="AWS region.",
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return wrapper


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
        "--db-password",
        envvar="OLMO_EVAL_DB_PASSWORD",
        default="postgres",
        help="Database password.",
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
    db_password: str,
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
        password=db_password,
    )
    db.initialize()
    return db


def format_timestamp(ts: datetime | None) -> str:
    """Format a timestamp for display."""
    if ts is None:
        return "-"
    return ts.strftime("%Y-%m-%d %H:%M:%S")


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


def experiments_to_dict(
    experiments: list[Any],
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert experiments to dict with optional instances grouped by task.

    Args:
        experiments: List of experiment results.
        instances: Optional list of instance predictions to include.
        limit: The limit used for querying instances (for pagination metadata).

    Returns:
        Dict with experiments containing tasks containing instances.
    """
    # Group instances by task_hash
    instance_groups: dict[str, list[dict[str, Any]]] = {}
    last_id: int | None = None
    if instances:
        for inst in instances:
            # Track last_id for pagination
            inst_id = inst.get("id")
            if inst_id is not None:
                last_id = inst_id
            # Use task_hash as key since we're grouping within experiments
            key = inst.get("task_hash", "")
            if key not in instance_groups:
                instance_groups[key] = []
            # Only include native_id and instance_metrics in grouped output
            instance_groups[key].append(
                {
                    "native_id": inst.get("native_id"),
                    "instance_metrics": inst.get("instance_metrics"),
                }
            )

    data = []
    for exp in experiments:
        tasks = []
        for t in exp.tasks:
            task_entry: dict[str, Any] = {
                "task_name": t.task_name,
                "task_hash": t.task_hash,
                "primary_metric": t.primary_metric,
                "primary_score": t.primary_score,
                "num_instances": t.num_instances,
                "metrics": t.metrics,
            }
            # Add instances for this task if available
            if t.task_hash and t.task_hash in instance_groups:
                task_entry["instances"] = instance_groups[t.task_hash]
            tasks.append(task_entry)

        data.append(
            {
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
                "tasks": tasks,
            }
        )

    result: dict[str, Any] = {"experiments": data}

    # Add pagination metadata if instances were queried
    if instances is not None:
        has_more = limit is not None and len(instances) >= limit
        result["pagination"] = {
            "last_id": last_id,
            "has_more": has_more,
        }

    return result


def experiments_to_json(
    experiments: list[Any],
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> str:
    """Convert experiments to JSON string."""
    return json.dumps(experiments_to_dict(experiments, instances, limit), indent=2)


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
    sorted_tasks, model_scores, _ = _build_model_task_scores(experiments, task_filter)

    if not sorted_tasks:
        return

    writer = csv.writer(sys.stdout)

    # Header row
    writer.writerow(["model"] + sorted_tasks)

    # Data rows
    for model_key in sorted(model_scores.keys()):
        scores = model_scores[model_key]
        row = [model_key]
        for task_name in sorted_tasks:
            score = scores.get(task_name)
            row.append(f"{score:.4f}" if score is not None else "")
        writer.writerow(row)


def task_comparison_to_dict(
    experiments: list[Any],
    task_filter: set[str] | None = None,
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convert task comparison to dict with optional instances grouped by model-task.

    Args:
        experiments: List of experiment results.
        task_filter: Optional set of task names to include.
        instances: Optional list of instance predictions to include.
        limit: The limit used for querying instances (for pagination metadata).

    Returns:
        Dict with models containing tasks containing scores and instances.
    """
    # Group instances by (model_hash, task_hash)
    instance_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    last_id: int | None = None
    if instances:
        for inst in instances:
            # Track last_id for pagination
            inst_id = inst.get("id")
            if inst_id is not None:
                last_id = inst_id
            key = (inst.get("model_hash", ""), inst.get("task_hash", ""))
            if key not in instance_groups:
                instance_groups[key] = []
            # Only include native_id and instance_metrics in grouped output
            instance_groups[key].append(
                {
                    "native_id": inst.get("native_id"),
                    "instance_metrics": inst.get("instance_metrics"),
                }
            )

    # Build output structured by model
    output: dict[str, Any] = {"models": []}

    for exp in experiments:
        model_entry = {
            "model_name": exp.model_name,
            "model_hash": exp.model_hash,
            "tasks": [],
        }

        for task in exp.tasks:
            if task_filter and task.task_name not in task_filter:
                continue

            task_entry: dict[str, Any] = {
                "task_name": task.task_name,
                "task_hash": task.task_hash,
                "primary_metric": task.primary_metric,
                "primary_score": task.primary_score,
            }

            # Add instances for this model-task pair if available
            key = (exp.model_hash or "", task.task_hash or "")
            if key in instance_groups:
                task_entry["instances"] = instance_groups[key]

            model_entry["tasks"].append(task_entry)

        output["models"].append(model_entry)

    # Add pagination metadata if instances were queried
    if instances is not None:
        has_more = limit is not None and len(instances) >= limit
        output["pagination"] = {
            "last_id": last_id,
            "has_more": has_more,
        }

    return output


def task_comparison_to_json(
    experiments: list[Any],
    task_filter: set[str] | None = None,
    instances: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> str:
    """Convert task comparison to JSON string."""
    return json.dumps(task_comparison_to_dict(experiments, task_filter, instances, limit), indent=2)


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


@click.group()
def results() -> None:
    """Query and display evaluation results."""
    pass


def _download_s3_files(
    experiment: Any,
    task_filter: tuple[str, ...],
    download_metrics: bool,
    download_predictions: bool,
    download_requests: bool,
    output_dir: str,
    s3_endpoint_url: str | None,
    s3_region: str,
) -> None:
    """Download files from S3 for an experiment.

    Uses the actual S3 paths stored in the database (s3_metrics_key, s3_predictions_key)
    rather than constructing paths from conventions.

    Args:
        experiment: The experiment ORM object.
        task_filter: Task names to filter (empty means all).
        download_metrics: Whether to download metrics.json.
        download_predictions: Whether to download predictions files.
        download_requests: Whether to download requests files.
        output_dir: Directory to save files.
        s3_endpoint_url: S3 endpoint URL (for LocalStack).
        s3_region: AWS region.
    """
    import boto3

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=s3_endpoint_url,
        region_name=s3_region,
    )

    downloaded_files: list[str] = []

    def parse_s3_uri(s3_uri: str) -> tuple[str, str] | None:
        """Parse s3://bucket/key into (bucket, key)."""
        if not s3_uri or not s3_uri.startswith("s3://"):
            return None
        path = s3_uri[5:]  # Remove 's3://'
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        return parts[0], parts[1]

    def download_file(s3_uri: str, label: str) -> str | None:
        """Download a file from S3 URI."""
        parsed = parse_s3_uri(s3_uri)
        if not parsed:
            console.print(f"[yellow]Warning:[/yellow] Invalid S3 URI for {label}: {s3_uri}")
            return None

        bucket, key = parsed
        # Use just the filename for local path to avoid deeply nested directories
        filename = Path(key).name
        local_file = output_path / experiment.experiment_id / filename
        local_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            s3_client.download_file(bucket, key, str(local_file))
            console.print(f"[green]Downloaded:[/green] {local_file}")
            return str(local_file)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to download {s3_uri}: {e}")
            return None

    # Download metrics.json from experiment's s3_location
    if download_metrics and experiment.s3_location:
        parsed = parse_s3_uri(experiment.s3_location.rstrip("/") + "/metrics.json")
        if parsed:
            bucket, key = parsed
            local_file = output_path / experiment.experiment_id / "metrics.json"
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3_client.download_file(bucket, key, str(local_file))
                console.print(f"[green]Downloaded:[/green] {local_file}")
                downloaded_files.append(str(local_file))
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to download metrics.json: {e}")

    # Download predictions files using paths stored in database
    tasks_to_download = experiment.tasks
    if task_filter:
        tasks_to_download = [t for t in tasks_to_download if t.task_name in task_filter]

    for task in tasks_to_download:
        if download_predictions and task.s3_predictions_key:
            result = download_file(task.s3_predictions_key, f"{task.task_name} predictions")
            if result:
                downloaded_files.append(result)

        # For requests, derive from predictions path (same directory, different filename)
        if download_requests and task.s3_predictions_key:
            # Replace predictions filename with requests filename
            requests_uri = task.s3_predictions_key.replace("predictions.jsonl", "requests.jsonl")
            result = download_file(requests_uri, f"{task.task_name} requests")
            if result:
                downloaded_files.append(result)

    if not downloaded_files:
        console.print("[yellow]No files were downloaded.[/yellow]")


@results.command(name="get", hidden=True)
@click.argument("experiment_id")
@click.option("--task", "-t", "task_filter", multiple=True)
@click.option(
    "--format", "-f", "output_format", type=click.Choice(["table", "json", "csv"]), default="table"
)
@db_options
@click.pass_context
def get_deprecated(
    ctx: click.Context,
    experiment_id: str,
    task_filter: tuple[str, ...],
    output_format: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """[DEPRECATED] Use 'olmo-eval results query --experiment <id>' instead."""
    console.print(
        "[yellow]Warning:[/yellow] 'olmo-eval results get' is deprecated.\n"
        f"Use: olmo-eval results query --experiment {experiment_id}"
        + (f" --task {' --task '.join(task_filter)}" if task_filter else "")
        + (f" --format {output_format}" if output_format != "table" else "")
        + "\n"
    )
    ctx.invoke(
        query,
        experiment_ids=(experiment_id,),
        model_names=(),
        model_hashes=(),
        task_names=task_filter,
        task_hash=None,
        instances=False,
        limit=100,
        after_id=None,
        output_format=output_format,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
    )


@results.command()
@click.option(
    "--experiment",
    "-e",
    "experiment_ids",
    multiple=True,
    help="Experiment ID(s) to query (can specify multiple).",
)
@click.option(
    "--model",
    "-m",
    "model_names",
    multiple=True,
    help="Model name(s) to query (can specify multiple).",
)
@click.option(
    "--model-hash",
    "-M",
    "model_hashes",
    multiple=True,
    help="Model hash(es) to query (can specify multiple).",
)
@click.option(
    "--task",
    "-t",
    "task_names",
    multiple=True,
    help="Task name(s) to filter (can specify multiple).",
)
@click.option(
    "--task-hash",
    "-T",
    help="Task hash to filter by (exact match).",
)
@click.option(
    "--instances/--no-instances",
    default=False,
    help="Include instance-level predictions (requires --format json or csv).",
)
@click.option(
    "--limit",
    "-n",
    default=100,
    type=int,
    help="Maximum results to return (applies to instances).",
)
@click.option(
    "--after-id",
    "after_id",
    default=None,
    type=int,
    help="Return instances after this ID (for keyset pagination).",
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
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    task_hash: str | None,
    instances: bool,
    limit: int,
    after_id: int | None,
    output_format: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Query evaluation results with flexible filters.

    Filter by experiment, model, model-hash, task, or task-hash.
    Use --instances with --format json or csv to include instance-level predictions.

    Examples:
        # Get experiment by ID
        olmo-eval results query --experiment exp_001

        # Compare models on tasks
        olmo-eval results query -m llama3.1-8b -m qwen2.5-72b -t mmlu -t gsm8k

        # Get instances for a task
        olmo-eval results query --task mmlu --instances --format csv

        # Get instances by experiment
        olmo-eval results query --experiment exp_001 --task mmlu --instances
    """
    # Validate at least one filter is provided
    if not any([experiment_ids, model_names, model_hashes, task_names, task_hash]):
        raise click.UsageError(
            "At least one filter is required: "
            "--experiment, --model, --model-hash, --task, or --task-hash"
        )

    db = get_database_session(db_host, db_port, db_name, db_user, db_password)

    try:
        from olmo_eval.storage.db.queries import QueryHelper
        from olmo_eval.storage.db.repository import ExperimentRepository

        with db.session() as session:
            helper = QueryHelper(session)
            repo = ExperimentRepository(session)

            all_experiments: list[Any] = []

            # Query by experiment IDs
            for exp_id in experiment_ids:
                exps = helper.get_by_experiment_id(exp_id)
                if not exps:
                    console.print(
                        f"[yellow]Warning:[/yellow] No experiments found with "
                        f"experiment_id='{exp_id}'"
                    )
                all_experiments.extend(exps)

            # Query by model names
            for model_name in model_names:
                exps = repo.query(model_name=model_name, limit=1000)
                if not exps:
                    console.print(
                        f"[yellow]Warning:[/yellow] No experiments found with "
                        f"model_name='{model_name}'"
                    )
                all_experiments.extend(exps)

            # Query by model hashes
            for mhash in model_hashes:
                exps = repo.query(model_hash=mhash, latest=True)
                if not exps:
                    console.print(
                        f"[yellow]Warning:[/yellow] No experiments found with model_hash='{mhash}'"
                    )
                all_experiments.extend(exps)

            # Determine if this is a comparison query (no experiment filter)
            # Show matrix for task-only, model-only, or model+task queries
            # Only show experiment details when querying by experiment ID
            comparison_query = not experiment_ids

            # If only task filter provided (no model filters), query by task
            if task_names and not model_names and not model_hashes:
                for tname in task_names:
                    exps = repo.query(task_name=tname, limit=100)
                    if not exps:
                        console.print(
                            f"[yellow]Warning:[/yellow] No experiments found with "
                            f"task_name='{tname}'"
                        )
                    all_experiments.extend(exps)

            if not all_experiments:
                console.print("[dim]No results found.[/dim]")
                return

            # Filter tasks if task_names specified
            task_filter = set(task_names) if task_names else None

            # Query instance-level predictions if requested
            instance_data: list[dict[str, Any]] = []
            if instances:
                task_name_filter = list(task_names) if task_names else None

                if experiment_ids:
                    for exp_id in experiment_ids:
                        instance_data.extend(
                            helper.get_instances_by_experiment_id(
                                experiment_id=exp_id,
                                task_name=task_name_filter,
                                limit=limit,
                                after_id=after_id,
                            )
                        )
                elif task_hash:
                    instance_data = helper.instance_repo.get_instances_by_task(
                        task_hash=task_hash,
                        task_name=task_name_filter,
                        limit=limit,
                        after_id=after_id,
                    )
                elif task_names and not model_names and not model_hashes:
                    instance_data = helper.instance_repo.get_instances_by_task(
                        task_name=task_name_filter,
                        limit=limit,
                        after_id=after_id,
                    )
                elif model_names or model_hashes:
                    if not task_names:
                        raise click.UsageError(
                            "--task is required when querying instances by model"
                        )
                    task_list = list(task_names)
                    instance_data = helper.get_instances_by_model(
                        task_name=task_list[0] if len(task_list) == 1 else task_list,
                        model_name=model_names[0] if model_names else None,
                        model_hash=model_hashes[0] if model_hashes else None,
                        limit=limit,
                        after_id=after_id,
                    )

            # Handle output formats
            if comparison_query:
                # Comparison query (task-only or model-only): use comparison matrix format
                if output_format == "json":
                    output = task_comparison_to_dict(
                        all_experiments,
                        task_filter,
                        instance_data if instances else None,
                        limit if instances else None,
                    )
                    print(json.dumps(output, indent=2, default=str))
                    return
                if output_format == "csv":
                    task_comparison_to_csv(all_experiments, task_filter)
                    if instances and instance_data:
                        console.print("\n[bold]Instance Predictions[/bold]")
                        instances_to_csv(instance_data)
                    return
            else:
                # Experiment query: use standard experiment format
                if output_format == "json":
                    output = experiments_to_dict(
                        all_experiments,
                        instance_data if instances else None,
                        limit if instances else None,
                    )
                    print(json.dumps(output, indent=2, default=str))
                    return
                if output_format == "csv":
                    experiments_to_csv(all_experiments)
                    if instances and instance_data:
                        console.print("\n[bold]Instance Predictions[/bold]")
                        instances_to_csv(instance_data)
                    return

            # Table format
            if comparison_query:
                # Comparison query: show matrix (models as rows, tasks as columns)
                print_task_comparison_matrix(all_experiments, task_filter)
            else:
                # Experiment/model query: show experiment details
                if len(all_experiments) > 1:
                    console.print(f"[bold]Found {len(all_experiments)} experiment(s)[/bold]\n")

                for i, experiment in enumerate(all_experiments):
                    if len(all_experiments) > 1:
                        n = len(all_experiments)
                        console.print(f"[bold cyan]--- Experiment {i + 1}/{n} ---[/bold cyan]")

                    print_experiment_detail(experiment)
                    console.print()
                    print_task_results_table(experiment.tasks, task_filter)

                    if i < len(all_experiments) - 1:
                        console.print()

            # Show instance summary for table format
            if instances:
                if instance_data:
                    console.print(
                        f"\n[yellow]Found {len(instance_data)} instance(s). "
                        f"Use --format json or --format csv to include them.[/yellow]"
                    )
                else:
                    console.print("\n[dim]No instance predictions found.[/dim]")
    finally:
        db.dispose()

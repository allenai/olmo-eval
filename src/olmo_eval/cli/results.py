"""CLI commands for querying and displaying evaluation results."""

from __future__ import annotations

import csv
import enum
import functools
import json
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.table import Table

from olmo_eval.cli.utils import console, format_timestamp
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


class FilterType(enum.Enum):
    """Filter types for experiment queries."""

    EXPERIMENT_ID = "experiment_id"
    MODEL_NAME = "model_name"
    MODEL_HASH = "model_hash"
    TASK_NAME = "task_name"


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
        from olmo_eval.storage.db.session import (
            get_database_session as _get_database_session,
        )

        return _get_database_session(db_host, db_port, db_name, db_user, db_password)
    except ImportError:
        console.print(
            "[red]Error:[/red] Database support requires psycopg. "
            "Install with: pip install psycopg[binary]"
        )
        raise SystemExit(1) from None


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
                    instance_metrics=inst.get("instance_metrics", {}),
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
                    primary_score=t.primary_score,
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
    return asdict(output)


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
                    instance_metrics=inst.get("instance_metrics", {}),
                )
            )

    models = []
    for exp in experiments:
        tasks = []
        for task in exp.tasks:
            if task_filter and task.task_name not in task_filter:
                continue

            key = (exp.model_hash or "", task.task_hash or "")
            task_instances = instance_groups.get(key)

            tasks.append(
                ComparisonTaskOutput(
                    task_name=task.task_name,
                    task_hash=task.task_hash,
                    primary_metric=task.primary_metric,
                    primary_score=task.primary_score,
                    instances=task_instances,
                )
            )

        models.append(
            ComparisonModelOutput(
                model_name=exp.model_name,
                model_hash=exp.model_hash,
                tasks=tasks,
            )
        )

    pagination = None
    if instances is not None:
        has_more = limit is not None and len(instances) >= limit
        pagination = PaginationOutput(last_id=last_id, has_more=has_more)

    output = ComparisonOutput(models=models, pagination=pagination)
    return asdict(output)


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
    "--experiment-group",
    "-G",
    help="Filter by experiment group (for cross-model analysis).",
)
@click.option(
    "--instances/--no-instances",
    default=False,
    help="Include instance-level predictions (requires --format json or csv).",
)
@click.option(
    "--limit",
    "-n",
    default=None,
    type=int,
    help="Maximum instances to return (default: no limit).",
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
    experiment_group: str | None,
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

    Filter by experiment, model, model-hash, task, task-hash, or experiment-group.
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

        # Get instances for an experiment group (cross-model analysis)
        olmo-eval results query -G my-benchmark --instances --format json
    """
    filters = [experiment_ids, model_names, model_hashes, task_names, task_hash, experiment_group]
    if not any(filters):
        raise click.UsageError(
            "At least one filter is required: "
            "--experiment, --model, --model-hash, --task, --task-hash, or --experiment-group"
        )

    db = get_database_session(db_host, db_port, db_name, db_user, db_password)

    try:
        from olmo_eval.storage.db.queries import QueryHelper
        from olmo_eval.storage.db.repository import ExperimentRepository

        with db.session() as session:
            helper = QueryHelper(session)
            repo = ExperimentRepository(session)

            # Experiment group with instances uses streaming (early return)
            if experiment_group and instances:
                _stream_experiment_group_instances(
                    session, experiment_group, model_hashes, task_hash, output_format
                )
                return

            # Fetch experiments based on filters
            all_experiments = _query_experiments(
                helper, repo, experiment_ids, model_names, model_hashes, task_names, task_hash
            )
            if not all_experiments:
                console.print("[dim]No results found.[/dim]")
                return

            # Comparison mode: no experiment IDs specified
            is_comparison = not experiment_ids

            # Filter experiments by model_hashes if specified
            if model_hashes:
                model_hash_set = set(model_hashes)
                all_experiments = [
                    exp for exp in all_experiments if exp.model_hash in model_hash_set
                ]

            # Filter tasks within experiments by task_hash or task_names
            if task_hash or task_names:
                task_name_set = set(task_names) if task_names else None
                for exp in all_experiments:
                    exp.tasks = [
                        t
                        for t in exp.tasks
                        if (task_hash is None or t.task_hash == task_hash)
                        and (task_name_set is None or t.task_name in task_name_set)
                    ]

            task_filter = set(task_names) if task_names else None

            # Fetch instances if requested
            instance_data = (
                _query_instances(
                    helper,
                    experiment_ids,
                    model_names,
                    model_hashes,
                    task_names,
                    task_hash,
                    limit,
                    after_id,
                )
                if instances
                else []
            )

            # Output results
            _output_results(
                all_experiments,
                instance_data,
                is_comparison,
                task_filter,
                output_format,
                instances,
                limit,
            )
    finally:
        db.dispose()


def _stream_experiment_group_instances(
    session: Any,
    experiment_group: str,
    model_hashes: tuple[str, ...],
    task_hash: str | None,
    output_format: str,
) -> None:
    """Stream instances for an experiment group directly to output."""
    from olmo_eval.storage.db.repository import InstancePredictionRepository
    from olmo_eval.storage.formatters import (
        stream_instances_to_csv,
        stream_instances_to_nested_json,
    )

    instance_repo = InstancePredictionRepository(session)
    instance_stream = instance_repo.stream_instances_with_metadata(
        experiment_group=experiment_group,
        model_hashes=list(model_hashes) if model_hashes else None,
        task_hashes=[task_hash] if task_hash else None,
    )

    if output_format == "csv":
        stream_instances_to_csv(instance_stream, sys.stdout, experiment_group)
    elif output_format == "json":
        stream_instances_to_nested_json(instance_stream, sys.stdout, experiment_group)
    else:
        console.print(
            "[yellow]Note:[/yellow] Use --format json or --format csv "
            "with --experiment-group --instances for output."
        )


def _query_experiments(
    helper: Any,
    repo: Any,
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    task_hash: str | None = None,
) -> list[Any]:
    """Query experiments based on provided filters."""
    results: list[Any] = []

    def query_with_warning(
        items: tuple[str, ...], query_fn: Callable[[str], list[Any]], filter_type: FilterType
    ) -> None:
        for item in items:
            exps = query_fn(item)
            if not exps:
                msg = f"No experiments found with {filter_type.value}='{item}'"
                console.print(f"[yellow]Warning:[/yellow] {msg}")
            results.extend(exps)

    # Query by each filter type
    query_with_warning(experiment_ids, helper.get_by_experiment_id, FilterType.EXPERIMENT_ID)
    query_with_warning(model_names, lambda x: repo.query(model_name=x), FilterType.MODEL_NAME)
    query_with_warning(
        model_hashes, lambda x: repo.query(model_hash=x, latest=True), FilterType.MODEL_HASH
    )

    # Task-only query (no model filters)
    if task_names and not model_names and not model_hashes:
        query_with_warning(task_names, lambda x: repo.query(task_name=x), FilterType.TASK_NAME)

    # Task hash query (if no results yet from other filters)
    if task_hash and not results:
        exps = repo.query(task_hash=task_hash)
        if not exps:
            console.print(
                f"[yellow]Warning:[/yellow] No experiments found with task_hash='{task_hash}'"
            )
        results.extend(exps)

    return results


def _query_instances(
    helper: Any,
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    task_hash: str | None,
    limit: int,
    after_id: int | None,
) -> list[dict[str, Any]]:
    """Query instance-level predictions based on filters."""
    return helper.query_instances(
        experiment_ids=list(experiment_ids) or None,
        model_names=list(model_names) or None,
        model_hashes=list(model_hashes) or None,
        task_names=list(task_names) or None,
        task_hash=task_hash,
        limit=limit,
        after_id=after_id,
    )


def _output_results(
    experiments: list[Any],
    instance_data: list[dict[str, Any]],
    is_comparison: bool,
    task_filter: set[str] | None,
    output_format: str,
    include_instances: bool,
    limit: int,
) -> None:
    """Output query results in the requested format."""
    instances_for_output = instance_data if include_instances else None
    limit_for_output = limit if include_instances else None

    # JSON output
    if output_format == "json":
        if is_comparison:
            output = task_comparison_to_dict(
                experiments, task_filter, instances_for_output, limit_for_output
            )
        else:
            output = experiments_to_dict(experiments, instances_for_output, limit_for_output)
        print(json.dumps(output, indent=2, default=str))
        return

    # CSV output
    if output_format == "csv":
        if is_comparison:
            task_comparison_to_csv(experiments, task_filter)
        else:
            experiments_to_csv(experiments)
        if include_instances and instance_data:
            console.print("\n[bold]Instance Predictions[/bold]")
            instances_to_csv(instance_data)
        return

    # Table output
    if is_comparison:
        print_task_comparison_matrix(experiments, task_filter)
    else:
        _print_experiments_table(experiments, task_filter)

    # Instance summary for table format
    if include_instances:
        if instance_data:
            console.print(
                f"\n[yellow]Found {len(instance_data)} instance(s). "
                f"Use --format json or --format csv to include them.[/yellow]"
            )
        else:
            console.print("\n[dim]No instance predictions found.[/dim]")


def _print_experiment_summary(experiments: list[Any]) -> None:
    """Print a unified experiment summary grouping by experiment_id.

    Shows shared experiment details once, then lists models and tasks.
    """
    from collections import defaultdict

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


def _print_experiments_table(experiments: list[Any], task_filter: set[str] | None) -> None:
    """Print experiments in table format with details."""
    # Use unified summary view for experiment queries
    _print_experiment_summary(experiments)

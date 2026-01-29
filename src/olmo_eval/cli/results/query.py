"""Query command for evaluation results."""

from __future__ import annotations

import enum
import json
import sys
from collections.abc import Callable
from typing import Any

import click

from olmo_eval.cli.results.display import (
    print_experiments_table,
    print_task_comparison_matrix,
)
from olmo_eval.cli.results.formatters import (
    experiments_to_csv,
    experiments_to_dict,
    instances_to_csv,
    task_comparison_to_csv,
    task_comparison_to_dict,
)
from olmo_eval.cli.results.options import db_options, get_database_session
from olmo_eval.cli.utils import console


class FilterType(enum.Enum):
    """Filter types for experiment queries."""

    EXPERIMENT_ID = "experiment_id"
    MODEL_NAME = "model_name"
    MODEL_HASH = "model_hash"
    TASK_NAME = "task_name"


@click.command()
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
    help="Filter by experiment group.",
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
        from olmo_eval.storage.backends.postgres.queries import QueryHelper
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

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
                helper,
                repo,
                experiment_ids,
                model_names,
                model_hashes,
                task_names,
                task_hash,
                experiment_group=experiment_group,
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
    from olmo_eval.storage.backends.postgres.repository import InstancePredictionRepository
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
    experiment_group: str | None = None,
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

    # Experiment group query (standalone filter)
    if experiment_group:
        exps = repo.query(experiment_group=experiment_group)
        if not exps:
            msg = f"No experiments found with experiment_group='{experiment_group}'"
            console.print(f"[yellow]Warning:[/yellow] {msg}")
        results.extend(exps)

    # Query by each filter type
    query_with_warning(experiment_ids, helper.get_by_experiment_id, FilterType.EXPERIMENT_ID)
    query_with_warning(model_names, lambda x: repo.query(model_name=x), FilterType.MODEL_NAME)
    query_with_warning(
        model_hashes, lambda x: repo.query(model_hash=x, latest=True), FilterType.MODEL_HASH
    )

    # Task-only query (no model filters)
    if task_names and not model_names and not model_hashes and not experiment_group:
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
                experiments, task_filter, instances_for_output, limit_for_output, include_instances
            )
        else:
            output = experiments_to_dict(
                experiments, instances_for_output, limit_for_output, include_instances
            )
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
        print_experiments_table(experiments, task_filter)

    # Instance summary for table format
    if include_instances:
        if instance_data:
            console.print(
                f"\n[yellow]Found {len(instance_data)} instance(s). "
                f"Use --format json or --format csv to include them.[/yellow]"
            )
        else:
            console.print("\n[dim]No instance predictions found.[/dim]")

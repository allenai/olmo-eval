"""Plot command for visualizing inference metrics in the terminal."""

from __future__ import annotations

import click

from olmo_eval.cli.metrics.app import print_stats_table, run_plot_app
from olmo_eval.cli.metrics.config import METRICS, METRICS_DB_NAME, DbConfig, QueryFilters
from olmo_eval.cli.metrics.data import extract_series_data, query_samples
from olmo_eval.cli.results.options import db_options, get_database_session
from olmo_eval.cli.utils import console


@click.command()
@click.option("-e", "--experiment", "experiment_ids", multiple=True, help="Experiment ID(s).")
@click.option("-G", "--experiment-group", "experiment_groups", multiple=True, help="Exp group.")
@click.option("-m", "--model", "model_names", multiple=True, help="Model name prefix(es).")
@click.option("-M", "--model-hash", "model_hashes", multiple=True, help="Model hash prefix(es).")
@click.option("-t", "--task", "task_names", multiple=True, help="Task name prefix(es).")
@click.option("-T", "--task-hash", "task_hashes", multiple=True, help="Task hash prefix(es).")
@click.option("--metric", type=click.Choice(list(METRICS.keys())), help="Single metric focus.")
@click.option("--stats-only", is_flag=True, help="Show only statistics table, no plots.")
@click.option("-r", "--refresh", "refresh_interval", type=int, default=10, help="Refresh interval.")
@db_options
def plot(
    experiment_ids: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    task_hashes: tuple[str, ...],
    metric: str | None,
    stats_only: bool,
    refresh_interval: int | None,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Plot inference metrics for evaluation runs.

    Shows how provider performance metrics evolve over the course of an evaluation run.

    \b
    Interactive controls:
      - Mouse scroll to zoom, drag to pan
      - Press 'r' to reset zoom, 'q' to quit
      - Press 's' to cycle sort, 'S' to toggle direction
      - Press Enter on a row to toggle series visibility

    \b
    Examples:
        olmo-eval metrics plot -G my-benchmark
        olmo-eval metrics plot -m OLMo-3 --metric throughput
    """
    filters = QueryFilters(
        experiment_ids=experiment_ids,
        experiment_groups=experiment_groups,
        model_names=model_names,
        model_hashes=model_hashes,
        task_names=task_names,
        task_hashes=task_hashes,
    )

    has_filter = any(
        [experiment_ids, experiment_groups, model_names, model_hashes, task_names, task_hashes]
    )
    if not has_filter:
        raise click.UsageError("At least one filter is required: -e, -G, -m, -M, -t, or -T")

    with console.status("[bold blue]Starting application..."):
        db = get_database_session(db_host, db_port, METRICS_DB_NAME, db_user, db_password)
        try:
            with db.session() as session:
                samples_by_exp = query_samples(session, filters)
        finally:
            db.dispose()

    if not samples_by_exp:
        console.print("[dim]No metrics found for the specified filter(s).[/dim]")
        return

    if stats_only:
        print_stats_table(samples_by_exp)
    else:
        db_config = DbConfig(host=db_host, port=db_port, user=db_user, password=db_password)
        run_plot_app(
            extract_series_data(samples_by_exp),
            samples_by_exp,
            metric,
            refresh_interval,
            filters,
            db_config,
        )

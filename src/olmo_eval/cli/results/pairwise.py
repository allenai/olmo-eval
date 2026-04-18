"""CLI command for pairwise model comparison."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from olmo_eval.cli.results.options import db_options, get_database_session
from olmo_eval.cli.utils import console


@click.command()
@click.option(
    "--experiment",
    "-e",
    "experiment_ids",
    multiple=True,
    help="Experiment ID(s) to compare (can specify multiple).",
)
@click.option(
    "--model",
    "-m",
    "model_names",
    multiple=True,
    help="Model name prefix(es) to compare.",
)
@click.option(
    "--model-hash",
    "-M",
    "model_hashes",
    multiple=True,
    help="Model hash prefix(es) to compare.",
)
@click.option(
    "--experiment-group",
    "-G",
    "experiment_groups",
    multiple=True,
    help="Experiment group prefix(es) to compare.",
)
@click.option(
    "--task",
    "-t",
    "task_name",
    default=None,
    help="Task name to compare on (provide --task or --task-hash).",
)
@click.option(
    "--task-hash",
    "-T",
    "task_hash",
    default=None,
    help="Task hash prefix to filter by (provide --task or --task-hash).",
)
@click.option(
    "--suite",
    "-S",
    "suite_name",
    default=None,
    help="Suite name (e.g. olmobase:math) — pools instances across all suite tasks.",
)
@click.option(
    "--metric",
    "metric",
    default=None,
    help="Metric in 'metric:scorer' format. Defaults to the task's primary_metric.",
)
@click.option(
    "--margin",
    default=0.0,
    type=float,
    help="Tie threshold for continuous metrics (default: 0.0).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    default=None,
    type=click.Path(),
    help="Save plot to file (e.g., matrix.png).",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["plot", "json", "csv"]),
    default="plot",
    help="Output format (default: plot).",
)
@db_options
def pairwise(
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    task_name: str | None,
    task_hash: str | None,
    suite_name: str | None,
    metric: str | None,
    margin: float,
    output_path: str | None,
    output_format: str,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Compute pairwise win/loss/tie comparison between models.

    Compare instance-level scores across 2+ experiments on a single task.
    Requires the [analysis] extra for plot output: pip install olmo-eval-internal[analysis]

    Examples:

        olmo-eval results pairwise -G my-benchmark -t mmlu

        olmo-eval results pairwise -m llama3 -m qwen2.5 -t gsm8k

        olmo-eval results pairwise -e exp_abc -e exp_def -t mmlu -f json

        olmo-eval results pairwise -G my-benchmark -t mmlu -o comparison.png
    """
    if not any([experiment_ids, model_names, model_hashes, experiment_groups]):
        raise click.UsageError(
            "At least one filter is required: "
            "--experiment, --model, --model-hash, or --experiment-group"
        )
    scope_count = sum(bool(x) for x in (task_name, task_hash, suite_name))
    if scope_count != 1:
        raise click.UsageError(
            "Provide exactly one of --task, --task-hash, or --suite to scope the comparison."
        )

    from olmo_eval.analysis.pairwise import compute_pairwise

    with console.status("[bold blue]Computing pairwise comparison..."):
        db = get_database_session(db_host, db_port, db_name, db_user, db_password)
        try:
            with db.session() as session:
                try:
                    result = compute_pairwise(
                        session=session,
                        task_name=task_name,
                        metric=metric,
                        margin=margin,
                        experiment_ids=list(experiment_ids) or None,
                        model_names=list(model_names) or None,
                        model_hashes=list(model_hashes) or None,
                        task_hash=task_hash,
                        experiment_groups=list(experiment_groups) or None,
                        suite_name=suite_name,
                    )
                except ValueError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise SystemExit(1) from None
        finally:
            db.dispose()

    if output_format == "json":
        _output_json(result)
    elif output_format == "csv":
        _output_csv(result)
    else:
        _output_plot(result, output_path)


def _output_json(result: Any) -> None:
    """Serialize PairwiseResult to JSON on stdout."""
    from olmo_eval.analysis.pairwise import get_win_rate

    n = len(result.models)
    data: dict[str, Any] = {
        "task_name": result.task_name,
        "suite_name": result.suite_name,
        "task_names": list(result.task_names),
        "metric": result.metric,
        "margin": result.margin,
        "instance_count": result.instance_count,
        "models": [{"label": m.label} for m in result.models],
        "pairs": [
            {
                "model_a": result.models[p.index_a].label,
                "model_b": result.models[p.index_b].label,
                "wins_a": p.wins_a,
                "wins_b": p.wins_b,
                "ties": p.ties,
                "win_rate_a": p.win_rate_a,
                "win_rate_b": p.win_rate_b,
            }
            for p in result.pairs
        ],
        "matrix": {
            result.models[i].label: {
                result.models[j].label: get_win_rate(result.pairs, i, j) for j in range(n) if j != i
            }
            for i in range(n)
        },
    }
    print(json.dumps(data, indent=2))


def _output_csv(result: Any) -> None:
    """Output NxN win-rate matrix as CSV on stdout."""
    import csv

    from olmo_eval.analysis.pairwise import get_win_rate

    n = len(result.models)
    writer = csv.writer(sys.stdout)
    labels = [m.label for m in result.models]
    writer.writerow([""] + labels)
    for i in range(n):
        row = [labels[i]]
        for j in range(n):
            if i == j:
                row.append("-")
            else:
                wr = get_win_rate(result.pairs, i, j)
                row.append(f"{wr:.4f}")
        writer.writerow(row)


def _output_plot(result: Any, output_path: str | None) -> None:
    """Render the pairwise matrix plot."""
    try:
        from olmo_eval.analysis.pairwise_plot import plot_pairwise_matrix
    except ImportError:
        console.print(
            "[red]Error:[/red] matplotlib is required for plot output. "
            "Install with: pip install olmo-eval-internal[analysis]"
        )
        raise SystemExit(1) from None

    if result.suite_name:
        title = f"{result.suite_name} ({len(result.task_names)} tasks) — {result.metric}"
    else:
        title = f"{result.task_name} — {result.metric}"
    plot_pairwise_matrix(result, title=title, save_path=output_path)
    if output_path:
        console.print(f"[green]Saved plot to {output_path}[/green]")
    else:
        import matplotlib.pyplot as plt

        plt.show()

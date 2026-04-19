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
@click.option(
    "--all",
    "keep_all",
    is_flag=True,
    default=False,
    help="Include every matched experiment as its own row (default: dedupe "
    "to the most recent per model+hash).",
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
    keep_all: bool,
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

    with console.status("[bold blue]Computing pairwise results..."):
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
                        keep_all=keep_all,
                    )
                except ValueError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise SystemExit(1) from None
        finally:
            db.dispose()

    matched = result.n_experiments_matched
    dropped = result.n_experiments_dropped
    if keep_all:
        console.print(
            f"[dim]Compared all {len(result.models)} experiments (-- all; no dedupe).[/dim]"
        )
    elif dropped:
        console.print(
            f"[dim]Compared {len(result.models)} unique model(s) from "
            f"{matched} matched experiments "
            f"({dropped} re-run(s) dropped; pass --all to keep them).[/dim]"
        )
    else:
        console.print(
            f"[dim]Compared {len(result.models)} model(s) from "
            f"{matched} matched experiment(s).[/dim]"
        )

    if output_format == "json":
        _output_json(result, output_path)
    elif output_format == "csv":
        _output_csv(result, output_path)
    else:
        _output_plot(result, output_path)


def _short_label(meta: Any) -> str:
    """Single-line display label suitable for matrix keys / CSV cells."""
    hash_short = (meta.model_hash or "")[:8]
    if meta.model_name and hash_short:
        return f"{meta.model_name} ({hash_short})"
    return meta.label.replace("\n", " ")


def _output_json(result: Any, output_path: str | None) -> None:
    """Serialize PairwiseResult to JSON — analysis-friendly schema.

    Writes to ``output_path`` if set, else stdout.
    """
    from olmo_eval.analysis.pairwise import get_win_rate

    n = len(result.models)
    labels = [_short_label(m) for m in result.models]

    data: dict[str, Any] = {
        "task_name": result.task_name,
        "suite_name": result.suite_name,
        "task_names": list(result.task_names),
        "metric": result.metric,
        "margin": result.margin,
        "instance_count": result.instance_count,
        "n_experiments_matched": result.n_experiments_matched,
        "n_experiments_dropped": result.n_experiments_dropped,
        "models": [
            {
                "label": labels[i],
                "model_name": m.model_name,
                "model_hash": m.model_hash,
                "timestamp": m.timestamp,
            }
            for i, m in enumerate(result.models)
        ],
        "pairs": [
            {
                "model_a": labels[p.index_a],
                "model_b": labels[p.index_b],
                "index_a": p.index_a,
                "index_b": p.index_b,
                "wins_a": p.wins_a,
                "wins_b": p.wins_b,
                "ties": p.ties,
                "n_contested": p.wins_a + p.wins_b,
                "win_rate_a": p.win_rate_a,
                "win_rate_b": p.win_rate_b,
                "se": p.se,
                "var_paired_diff": p.var_paired_diff,
                "var_marginal_sum": p.var_marginal_sum,
            }
            for p in result.pairs
        ],
        "matrix": {
            labels[i]: {labels[j]: get_win_rate(result.pairs, i, j) for j in range(n) if j != i}
            for i in range(n)
        },
    }
    payload = json.dumps(data, indent=2)
    if output_path:
        with open(output_path, "w") as f:
            f.write(payload)
            f.write("\n")
        console.print(f"[green]Saved JSON to {output_path}[/green]")
    else:
        print(payload)


def _output_csv(result: Any, output_path: str | None) -> None:
    """Output pairwise stats as a long-format CSV (one row per pair).

    Columns: model_a, model_b, wins_a, wins_b, ties, n_contested, win_rate_a,
    win_rate_b, se, var_paired_diff, var_marginal_sum. Writes to
    ``output_path`` if set, else stdout.
    """
    import csv

    labels = [_short_label(m) for m in result.models]

    def _write_rows(writer: Any) -> None:
        writer.writerow(
            [
                "model_a",
                "model_b",
                "wins_a",
                "wins_b",
                "ties",
                "n_contested",
                "win_rate_a",
                "win_rate_b",
                "se",
                "var_paired_diff",
                "var_marginal_sum",
            ]
        )
        for p in result.pairs:
            writer.writerow(
                [
                    labels[p.index_a],
                    labels[p.index_b],
                    p.wins_a,
                    p.wins_b,
                    p.ties,
                    p.wins_a + p.wins_b,
                    f"{p.win_rate_a:.6f}",
                    f"{p.win_rate_b:.6f}",
                    f"{p.se:.6f}",
                    f"{p.var_paired_diff:.6f}",
                    f"{p.var_marginal_sum:.6f}",
                ]
            )

    if output_path:
        with open(output_path, "w", newline="") as f:
            _write_rows(csv.writer(f))
        console.print(f"[green]Saved CSV to {output_path}[/green]")
    else:
        _write_rows(csv.writer(sys.stdout))


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

"""CLI entrypoint for the results viewer and its pairwise exports."""

from __future__ import annotations

import json
import sys
from typing import Any

import click

from olmo_eval.cli.results.options import db_options, get_database_session
from olmo_eval.cli.utils import console


@click.command(name="viewer")
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
    "--exclude-model",
    "exclude_model_names",
    multiple=True,
    help="Model name prefix(es) to exclude from the comparison.",
)
@click.option(
    "--exclude-model-hash",
    "exclude_model_hashes",
    multiple=True,
    help="Model hash prefix(es) to exclude from the comparison.",
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
    help="Exact task name to compare on — full variant/regime, e.g. "
    "'humaneval:3shot:pass_at_1'. For prefix/fuzzy lookup use 'results query'; "
    "for grouping use --suite.",
)
@click.option(
    "--task-hash",
    "-T",
    "task_hash",
    default=None,
    help="Task hash prefix to filter by. Must resolve to a single task name.",
)
@click.option(
    "--exclude-task",
    "exclude_task_names",
    multiple=True,
    help="Exact task name(s) to exclude from the scoped comparison.",
)
@click.option(
    "--exclude-task-hash",
    "exclude_task_hashes",
    multiple=True,
    help="Task hash prefix(es) to exclude from the scoped comparison.",
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
    help="Save JSON / CSV to a file. The browser viewer does not use --output.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default=None,
    help="Optional export format. Omit to launch the local results viewer.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="Bind address for viewer mode (default: 127.0.0.1).",
)
@click.option(
    "--port",
    default=8765,
    type=int,
    help="Listen port for viewer mode (default: 8765).",
)
@click.option(
    "--repeated-runs/--latest-only",
    "keep_all",
    default=False,
    help="Keep repeated runs as distinct rows instead of collapsing to the latest "
    "run per model hash.",
)
@click.option(
    "--require-full-coverage/--no-require-full-coverage",
    "require_full_coverage",
    default=True,
    help="Drop models that lack full coverage of the suite's tasks "
    "(suite scope only; default: enabled).",
)
@db_options
def viewer(
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    exclude_model_names: tuple[str, ...],
    exclude_model_hashes: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    task_name: str | None,
    task_hash: str | None,
    exclude_task_names: tuple[str, ...],
    exclude_task_hashes: tuple[str, ...],
    suite_name: str | None,
    metric: str | None,
    margin: float,
    output_path: str | None,
    output_format: str | None,
    host: str,
    port: int,
    keep_all: bool,
    require_full_coverage: bool,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Launch the results viewer or dump viewer data as JSON / CSV.

    Examples:

        olmo-eval results viewer

        olmo-eval results viewer -G my-benchmark -S multipl_e:pass_at_1

        olmo-eval results viewer -G my-benchmark -S multipl_e:pass_at_1 -f json -o matrix.json

        olmo-eval results viewer -m model-a -m model-b -t gsm8k:olmo3base -f csv
    """
    if output_format is None:
        if output_path is not None:
            raise click.UsageError("--output is not supported when launching the results viewer.")
        if experiment_ids or model_names or model_hashes:
            raise click.UsageError(
                "The results viewer starts from experiment-group discovery. "
                "Use --experiment-group to open a specific group, or switch to "
                "-f json/csv for direct filtered dumps."
            )
        if task_hash is not None:
            raise click.UsageError(
                "The results viewer does not support --task-hash. "
                "Use --task or --suite to open a specific view."
            )
        if exclude_model_names or exclude_model_hashes or exclude_task_names or exclude_task_hashes:
            raise click.UsageError(
                "The results viewer does not take exclude flags. "
                "Use the in-browser filters to narrow what you see."
            )
        if len(experiment_groups) > 1:
            raise click.UsageError(
                "The results viewer accepts at most one --experiment-group seed value."
            )
        scope_count = sum(bool(x) for x in (task_name, suite_name))
        if scope_count > 1:
            raise click.UsageError(
                "Provide at most one of --task or --suite to open the initial viewer state."
            )
        _serve_html_browser(
            db_host=db_host,
            db_port=db_port,
            db_name=db_name,
            db_user=db_user,
            db_password=db_password,
            host=host,
            port=port,
            initial_group=experiment_groups[0] if experiment_groups else None,
            initial_scope_key=_initial_scope_key(task_name=task_name, suite_name=suite_name),
            margin=margin,
            keep_all=keep_all,
            require_full_coverage=require_full_coverage,
        )
        return

    _run_pairwise_dump(
        experiment_ids=experiment_ids,
        model_names=model_names,
        model_hashes=model_hashes,
        exclude_model_names=exclude_model_names,
        exclude_model_hashes=exclude_model_hashes,
        experiment_groups=experiment_groups,
        task_name=task_name,
        task_hash=task_hash,
        exclude_task_names=exclude_task_names,
        exclude_task_hashes=exclude_task_hashes,
        suite_name=suite_name,
        metric=metric,
        margin=margin,
        output_path=output_path,
        output_format=output_format,
        keep_all=keep_all,
        require_full_coverage=require_full_coverage,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
    )


def _run_pairwise_dump(
    *,
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    exclude_model_names: tuple[str, ...],
    exclude_model_hashes: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    task_name: str | None,
    task_hash: str | None,
    exclude_task_names: tuple[str, ...],
    exclude_task_hashes: tuple[str, ...],
    suite_name: str | None,
    metric: str | None,
    margin: float,
    output_path: str | None,
    output_format: str,
    keep_all: bool,
    require_full_coverage: bool,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Compute pairwise data and dump it as JSON or CSV."""
    if not any([experiment_ids, model_names, model_hashes, experiment_groups]):
        raise click.UsageError(
            "At least one filter is required for dumps: "
            "--experiment, --model, --model-hash, or --experiment-group"
        )
    scope_count = sum(bool(x) for x in (task_name, task_hash, suite_name))
    if scope_count != 1:
        raise click.UsageError(
            "Provide exactly one of --task, --task-hash, or --suite to scope the dump."
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
                        exclude_model_names=list(exclude_model_names) or None,
                        exclude_model_hashes=list(exclude_model_hashes) or None,
                        task_hash=task_hash,
                        exclude_task_names=list(exclude_task_names) or None,
                        exclude_task_hashes=list(exclude_task_hashes) or None,
                        experiment_groups=list(experiment_groups) or None,
                        suite_name=suite_name,
                        keep_all=keep_all,
                        require_full_coverage=require_full_coverage,
                    )
                except ValueError as e:
                    console.print(f"[red]Error:[/red] {e}")
                    raise SystemExit(1) from None
        finally:
            db.dispose()

    if result.filtered_models:
        from rich.table import Table

        console.print(
            f"[yellow]Filtered {len(result.filtered_models)} model(s) "
            "lacking full suite coverage:[/yellow]"
        )
        table = Table(show_lines=True)
        table.add_column("Model", style="cyan", no_wrap=False)
        table.add_column("Hash", style="dim")
        table.add_column("Missing tasks", justify="right")
        table.add_column("Short instances", justify="right")
        table.add_column("Gaps")
        for fm in result.filtered_models:
            gaps: list[str] = [f"- {t} (missing)" for t in fm.missing_tasks]
            gaps.extend(
                f"- {t} ({have}/{expected})" for t, have, expected in fm.instance_shortfalls
            )
            table.add_row(
                fm.model_name,
                (fm.model_hash or "")[:8],
                str(len(fm.missing_tasks)),
                str(len(fm.instance_shortfalls)),
                "\n".join(gaps),
            )
        console.print(table)

    matched = result.n_experiments_matched
    dropped = result.n_experiments_dropped
    if keep_all:
        console.print(
            f"[dim]Compared all {len(result.models)} experiments (repeated runs enabled).[/dim]"
        )
    elif dropped:
        console.print(
            f"[dim]Compared {len(result.models)} unique model(s) from "
            f"{matched} matched experiments "
            f"({dropped} repeated run(s) collapsed to the latest per model hash).[/dim]"
        )
    else:
        console.print(
            f"[dim]Compared {len(result.models)} model(s) from "
            f"{matched} matched experiment(s).[/dim]"
        )

    if output_format == "json":
        _output_json(result, output_path)
    else:
        _output_csv(result, output_path)


def _short_label(meta: Any) -> str:
    """Collapse a model label to one line for text outputs."""
    hash_short = (meta.model_hash or "")[:8]
    if meta.model_name and hash_short:
        return f"{meta.model_name} ({hash_short})"
    return meta.label.replace("\n", " ")


def _output_json(result: Any, output_path: str | None) -> None:
    """Write the pairwise result as JSON."""
    from olmo_eval.analysis.pairwise import get_win_rate
    from olmo_eval.analysis.pairwise_metrics import build_task_display_entries

    n = len(result.models)
    labels = [_short_label(m) for m in result.models]
    task_entries = build_task_display_entries(
        tuple(result.task_names),
        tuple(getattr(result, "task_hashes", ()) or ()),
    )

    data: dict[str, Any] = {
        "metadata": {
            "scope_name": result.task_name,
            "scope_kind": "suite" if result.suite_name else "task",
            "suite_name": result.suite_name,
            "contributing_task_names": list(result.task_names),
            "contributing_task_hashes": list(getattr(result, "task_hashes", ()) or ()),
            "contributing_tasks": [
                {
                    "task_id": task_entry.id,
                    "task_name": task_entry.task_name,
                    "task_hash": task_entry.task_hash,
                    "task_label": task_entry.label,
                    "task_full_label": task_entry.full_label,
                    "metric_key": (
                        result.task_metric_keys[task_idx]
                        if task_idx < len(result.task_metric_keys)
                        else None
                    ),
                }
                for task_idx, task_entry in enumerate(task_entries)
            ],
            "metric_name": result.metric,
            "tie_margin": result.margin,
            "shared_instance_count": result.instance_count,
            "score_display_format": result.score_display_format,
            "score_unit": result.score_unit,
            "score_higher_is_better": result.higher_is_better,
            "matched_experiment_count": result.n_experiments_matched,
            "dropped_experiment_count": result.n_experiments_dropped,
        },
        "models": [
            {
                "model_index": i,
                "display_label": labels[i],
                "model_name": m.model_name,
                "model_hash": m.model_hash,
                "timestamp": m.timestamp,
                "total_cost": result.model_costs[i] if i < len(result.model_costs) else None,
                "shared_instance_mean_score": (
                    result.model_shared_scores[i] if i < len(result.model_shared_scores) else None
                ),
                "task_metric_keys_by_task_id": {
                    task_entry.id: (
                        result.task_metric_keys[task_idx]
                        if task_idx < len(result.task_metric_keys)
                        else None
                    )
                    for task_idx, task_entry in enumerate(task_entries)
                },
                "task_metric_keys_by_task_name": {
                    task_name: (
                        result.task_metric_keys[task_idx]
                        if task_idx < len(result.task_metric_keys)
                        else None
                    )
                    for task_idx, task_name in enumerate(result.task_names)
                },
                "task_scores_by_task_id": {
                    task_entry.id: (
                        result.model_task_scores[i][task_idx]
                        if i < len(result.model_task_scores)
                        and task_idx < len(result.model_task_scores[i])
                        else None
                    )
                    for task_idx, task_entry in enumerate(task_entries)
                },
                "task_scores_by_task_name": {
                    task_name: (
                        result.model_task_scores[i][task_idx]
                        if i < len(result.model_task_scores)
                        and task_idx < len(result.model_task_scores[i])
                        else None
                    )
                    for task_idx, task_name in enumerate(result.task_names)
                },
            }
            for i, m in enumerate(result.models)
        ],
        "pairwise_comparisons": [
            {
                "model_a_index": p.index_a,
                "model_b_index": p.index_b,
                "model_a_label": labels[p.index_a],
                "model_b_label": labels[p.index_b],
                "wins_model_a": p.wins_a,
                "wins_model_b": p.wins_b,
                "tie_count": p.ties,
                "contested_instance_count": p.wins_a + p.wins_b,
                "win_rate_model_a": p.win_rate_a,
                "win_rate_model_b": p.win_rate_b,
                "win_rate_standard_error": p.se,
                "probability_model_a_beats_model_b": p.prob_a_gt_b,
                "p_value": p.p_value,
                "paired_difference_variance": p.var_paired_diff,
                "marginal_sum_variance": p.var_marginal_sum,
            }
            for p in result.pairs
        ],
        "win_rate_matrix_by_model_label": {
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
    """Write one CSV row per pair."""
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


def _initial_scope_key(*, task_name: str | None, suite_name: str | None) -> str | None:
    """Translate optional CLI seed flags into the browser's scope-key format."""
    if suite_name:
        return f"suite::{suite_name}"
    if task_name:
        return f"task::{task_name}"
    return None


def _serve_html_browser(
    *,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    host: str,
    port: int,
    initial_group: str | None,
    initial_scope_key: str | None,
    margin: float,
    keep_all: bool,
    require_full_coverage: bool,
) -> None:
    """Start the local results viewer server for interactive exploration."""
    from olmo_eval.cli.results.viewer_server import serve_results_viewer

    db = get_database_session(db_host, db_port, db_name, db_user, db_password)
    url = f"http://{host}:{port}"
    console.print(f"[green]Starting results viewer at {url}[/green]")
    console.print("[dim]Press Ctrl+C to stop the viewer.[/dim]")
    try:
        serve_results_viewer(
            db=db,
            host=host,
            port=port,
            initial_group=initial_group,
            initial_scope_key=initial_scope_key,
            margin=margin,
            keep_all=keep_all,
            require_full_coverage=require_full_coverage,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Results viewer stopped.[/dim]")
    finally:
        db.dispose()

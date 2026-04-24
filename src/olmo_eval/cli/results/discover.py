"""Discovery commands for results data."""

from __future__ import annotations

import click

from olmo_eval.cli.results.options import db_options, get_database_session
from olmo_eval.cli.utils import console


@click.command()
@click.option(
    "--experiment",
    "-e",
    "experiment_ids",
    multiple=True,
    help="Experiment ID(s) to scope coverage to.",
)
@click.option(
    "--model",
    "-m",
    "model_names",
    multiple=True,
    help="Model name prefix(es) to scope coverage to.",
)
@click.option(
    "--model-hash",
    "-M",
    "model_hashes",
    multiple=True,
    help="Model hash prefix(es) to scope coverage to.",
)
@click.option(
    "--experiment-group",
    "-G",
    "experiment_groups",
    multiple=True,
    help="Experiment group prefix(es) to scope coverage to.",
)
@click.option(
    "--filter",
    "-f",
    "name_filter",
    default="",
    help="Substring filter applied to suite names.",
)
@click.option(
    "--min-coverage",
    default=0.0,
    type=float,
    help="Hide suites with fractional coverage below this value (0-1).",
)
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Include suites with zero coverage (hidden by default).",
)
@db_options
def suites(
    experiment_ids: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    name_filter: str,
    min_coverage: float,
    show_all: bool,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """List suite coverage within the selected experiment scope."""
    from sqlalchemy import distinct, or_, select

    from olmo_eval.evals.suites.registry import get_suite, list_suites
    from olmo_eval.storage.backends.postgres.models import Experiment, TaskResult

    with console.status("[bold blue]Scanning suites..."):
        db = get_database_session(db_host, db_port, db_name, db_user, db_password)
        try:
            with db.session() as session:
                stmt = select(distinct(TaskResult.task_name)).join(
                    Experiment, Experiment.id == TaskResult.experiment_pk
                )
                if experiment_ids:
                    stmt = stmt.where(Experiment.experiment_id.in_(experiment_ids))
                if model_names:
                    stmt = stmt.where(
                        or_(*[Experiment.model_name.startswith(n) for n in model_names])
                    )
                if model_hashes:
                    stmt = stmt.where(
                        or_(*[Experiment.model_hash.startswith(h) for h in model_hashes])
                    )
                if experiment_groups:
                    stmt = stmt.where(
                        or_(*[Experiment.experiment_group.startswith(g) for g in experiment_groups])
                    )
                present_tasks: set[str] = set(session.execute(stmt).scalars().all())
        finally:
            db.dispose()

    rows: list[tuple[str, int, int, float]] = []
    for name in list_suites():
        if name_filter and name_filter.lower() not in name.lower():
            continue
        suite = get_suite(name)
        expanded = suite.expanded_tasks
        total = len(expanded)
        covered = sum(1 for t in expanded if t in present_tasks)
        ratio = covered / total if total else 0.0
        if not show_all and covered == 0:
            continue
        if ratio < min_coverage:
            continue
        rows.append((name, covered, total, ratio))

    rows.sort(key=lambda r: (-r[3], r[0]))

    if not rows:
        console.print("[dim]No suites matched.[/dim]")
        return

    from rich.table import Table

    scope_bits = []
    if experiment_groups:
        scope_bits.append(f"groups={list(experiment_groups)}")
    if model_names:
        scope_bits.append(f"models={list(model_names)}")
    if model_hashes:
        scope_bits.append(f"hashes={list(model_hashes)}")
    if experiment_ids:
        scope_bits.append(f"experiments={list(experiment_ids)}")
    title = "Suite coverage" + (f" ({', '.join(scope_bits)})" if scope_bits else "")

    table = Table(title=title)
    table.add_column("Suite", style="cyan")
    table.add_column("Covered", justify="right", style="green")
    table.add_column("Total", justify="right", style="dim")
    table.add_column("Coverage", justify="right")
    for name, covered, total, ratio in rows:
        pct_color = "green" if ratio >= 0.99 else "yellow" if ratio >= 0.5 else "red"
        table.add_row(
            name,
            str(covered),
            str(total),
            f"[{pct_color}]{ratio:.0%}[/{pct_color}]",
        )
    console.print(table)


@click.command()
@click.option(
    "--filter",
    "-f",
    "name_filter",
    default="",
    help="Substring filter applied to group names.",
)
@click.option(
    "--limit",
    "-n",
    default=50,
    type=int,
    help="Maximum rows to show (default: 50).",
)
@db_options
def groups(
    name_filter: str,
    limit: int,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """List experiment groups with summary counts."""
    from sqlalchemy import distinct, func, select

    from olmo_eval.storage.backends.postgres.models import Experiment, TaskResult

    with console.status("[bold blue]Scanning experiment groups..."):
        db = get_database_session(db_host, db_port, db_name, db_user, db_password)
        try:
            with db.session() as session:
                stmt = (
                    select(
                        Experiment.experiment_group,
                        func.count(distinct(Experiment.id)).label("experiments"),
                        func.count(distinct(Experiment.model_hash)).label("models"),
                        func.max(Experiment.timestamp).label("most_recent"),
                    )
                    .group_by(Experiment.experiment_group)
                    .order_by(func.max(Experiment.timestamp).desc())
                )
                if name_filter:
                    stmt = stmt.where(Experiment.experiment_group.ilike(f"%{name_filter}%"))
                stmt = stmt.limit(limit)
                group_rows = session.execute(stmt).all()

                if not group_rows:
                    console.print("[dim]No experiment groups matched.[/dim]")
                    return

                task_counts_stmt = (
                    select(
                        Experiment.experiment_group,
                        func.count(distinct(TaskResult.task_name)).label("tasks"),
                    )
                    .join(TaskResult, Experiment.id == TaskResult.experiment_pk)
                    .group_by(Experiment.experiment_group)
                )
                if name_filter:
                    task_counts_stmt = task_counts_stmt.where(
                        Experiment.experiment_group.ilike(f"%{name_filter}%")
                    )
                task_count_map: dict[str, int] = {
                    g: n for g, n in session.execute(task_counts_stmt).all()
                }
        finally:
            db.dispose()

    from rich.table import Table

    table = Table(title="Experiment groups")
    table.add_column("Group", style="cyan")
    table.add_column("Experiments", justify="right")
    table.add_column("Models", justify="right")
    table.add_column("Tasks", justify="right")
    table.add_column("Most recent", style="dim")
    for row in group_rows:
        group_name, n_experiments, n_models, most_recent = row
        n_tasks = task_count_map.get(group_name, 0)
        ts = most_recent.strftime("%Y-%m-%d %H:%M") if most_recent else ""
        table.add_row(group_name, str(n_experiments), str(n_models), str(n_tasks), ts)
    console.print(table)


@click.command()
@click.argument("group_name")
@click.option(
    "--top",
    "-n",
    default=20,
    type=int,
    help="Maximum rows per section (default: 20).",
)
@click.option(
    "--min-coverage",
    default=0.0,
    type=float,
    help="Hide suites with fractional coverage below this value (0-1).",
)
@db_options
def group(
    group_name: str,
    top: int,
    min_coverage: float,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Show models, tasks, and suite coverage for one experiment group."""
    from sqlalchemy import distinct, func, select

    from olmo_eval.evals.suites.registry import get_suite, list_suites
    from olmo_eval.storage.backends.postgres.models import Experiment, TaskResult

    with console.status(f"[bold blue]Loading group '{group_name}'..."):
        db = get_database_session(db_host, db_port, db_name, db_user, db_password)
        try:
            with db.session() as session:
                summary = session.execute(
                    select(
                        func.count(distinct(Experiment.id)).label("experiments"),
                        func.count(distinct(Experiment.model_hash)).label("models"),
                        func.min(Experiment.timestamp).label("first_ts"),
                        func.max(Experiment.timestamp).label("last_ts"),
                    ).where(Experiment.experiment_group == group_name)
                ).one()

                n_experiments, n_models, first_ts, last_ts = summary
                if not n_experiments:
                    console.print(f"[red]No experiments found in group '{group_name}'.[/red]")
                    return

                model_rows = session.execute(
                    select(
                        Experiment.model_name,
                        Experiment.model_hash,
                        func.max(Experiment.timestamp).label("last_ts"),
                        func.count(distinct(TaskResult.task_name)).label("tasks"),
                    )
                    .join(TaskResult, Experiment.id == TaskResult.experiment_pk, isouter=True)
                    .where(Experiment.experiment_group == group_name)
                    .group_by(Experiment.model_name, Experiment.model_hash)
                    .order_by(func.max(Experiment.timestamp).desc())
                ).all()

                task_rows = session.execute(
                    select(
                        TaskResult.task_name,
                        func.count(distinct(Experiment.model_hash)).label("models"),
                        func.max(TaskResult.primary_metric).label("metric"),
                    )
                    .join(Experiment, Experiment.id == TaskResult.experiment_pk)
                    .where(Experiment.experiment_group == group_name)
                    .group_by(TaskResult.task_name)
                    .order_by(TaskResult.task_name)
                ).all()

                present_tasks = {t for t, _, _ in task_rows}
        finally:
            db.dispose()

    from rich.table import Table

    ts_range = ""
    if first_ts and last_ts:
        ts_range = f"  {first_ts.strftime('%Y-%m-%d')} → {last_ts.strftime('%Y-%m-%d')}"

    console.print(
        f"\n[bold cyan]{group_name}[/bold cyan]  "
        f"[dim]{n_experiments} experiments, {n_models} models, "
        f"{len(present_tasks)} tasks[/dim][dim]{ts_range}[/dim]"
    )

    models_table = Table(title=f"Models ({len(model_rows)})")
    models_table.add_column("Model", style="cyan")
    models_table.add_column("Hash", style="dim")
    models_table.add_column("Tasks", justify="right")
    models_table.add_column("Most recent", style="dim")
    for model_name, model_hash, last_seen, n_tasks in model_rows[:top]:
        hash_short = model_hash[:8] if model_hash else ""
        ts = last_seen.strftime("%Y-%m-%d %H:%M") if last_seen else ""
        models_table.add_row(model_name, hash_short, str(n_tasks), ts)
    if len(model_rows) > top:
        models_table.caption = f"[dim]…{len(model_rows) - top} more (use --top)[/dim]"
    console.print(models_table)

    tasks_table = Table(title=f"Tasks ({len(task_rows)})")
    tasks_table.add_column("Task", style="cyan")
    tasks_table.add_column("Models", justify="right")
    tasks_table.add_column("Primary metric", style="dim")
    for task_name, n_task_models, metric in task_rows[:top]:
        tasks_table.add_row(task_name, str(n_task_models), metric or "")
    if len(task_rows) > top:
        tasks_table.caption = f"[dim]…{len(task_rows) - top} more (use --top)[/dim]"
    console.print(tasks_table)

    suite_rows: list[tuple[str, int, int, float]] = []
    for name in list_suites():
        suite = get_suite(name)
        expanded = suite.expanded_tasks
        total = len(expanded)
        covered = sum(1 for t in expanded if t in present_tasks)
        ratio = covered / total if total else 0.0
        if covered == 0:
            continue
        if ratio < min_coverage:
            continue
        suite_rows.append((name, covered, total, ratio))
    suite_rows.sort(key=lambda r: (-r[3], r[0]))

    suites_table = Table(title=f"Suites with coverage ({len(suite_rows)})")
    suites_table.add_column("Suite", style="cyan")
    suites_table.add_column("Covered", justify="right", style="green")
    suites_table.add_column("Total", justify="right", style="dim")
    suites_table.add_column("Coverage", justify="right")
    for name, covered, total, ratio in suite_rows[:top]:
        pct_color = "green" if ratio >= 0.99 else "yellow" if ratio >= 0.5 else "red"
        suites_table.add_row(
            name,
            str(covered),
            str(total),
            f"[{pct_color}]{ratio:.0%}[/{pct_color}]",
        )
    if len(suite_rows) > top:
        suites_table.caption = f"[dim]…{len(suite_rows) - top} more (use --top)[/dim]"
    console.print(suites_table)

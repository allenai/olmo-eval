"""Beaker group management commands."""

from datetime import UTC

import click
from rich.table import Table

from olmo_eval.cli.utils import console


def _get_launcher() -> "BeakerLauncher":  # type: ignore[name-defined]  # noqa: F821
    """Get BeakerLauncher instance, handling import errors."""
    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None
    return BeakerLauncher()


def _get_beaker_group(launcher: "BeakerLauncher", group_name: str) -> "BeakerGroup":  # type: ignore[name-defined]  # noqa: F821
    """Get a Beaker group by name, handling errors."""
    try:
        from beaker.exceptions import BeakerGroupNotFound

        return launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None


@click.group()
def group() -> None:
    """Manage Beaker groups."""
    pass


@group.command(name="info")
@click.argument("group_name")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    help="Output format (csv exports raw metrics from Beaker)",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed task info")
@click.option("--wait", "-w", is_flag=True, help="Wait for all experiments to complete")
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    help="Seconds between status checks when waiting",
)
def group_info(
    group_name: str, output_format: str, verbose: bool, wait: bool, poll_interval: int
) -> None:
    """Get detailed info about a Beaker group."""
    import json as json_module

    launcher = _get_launcher()
    beaker_group = _get_beaker_group(launcher, group_name)

    if wait:
        import time

        console.print(f"[dim]Waiting for experiments in '{group_name}' to complete...[/dim]")
        while True:
            status = launcher.get_group_status(beaker_group)
            running = status.get("running", 0) + status.get("pending", 0)

            if running == 0:
                break

            console.print(
                f"[dim]  {status.get('succeeded', 0)} succeeded, "
                f"{status.get('running', 0)} running, "
                f"{status.get('pending', 0)} pending, "
                f"{status.get('failed', 0)} failed[/dim]"
            )
            time.sleep(poll_interval)

        console.print("[green]All experiments completed.[/green]\n")

    status = launcher.get_group_status(beaker_group)
    experiments = launcher.get_group_experiments(beaker_group)
    group_url = launcher.get_group_url(beaker_group)

    if output_format == "csv":
        try:
            csv_data = launcher.export_group_metrics(beaker_group)
            click.echo(csv_data)
        except Exception as e:
            from beaker import BeakerWorkloadStatus

            console.print(f"[yellow]Warning:[/yellow] Could not export metrics: {e}")
            click.echo("experiment_id,name,status")
            for exp in experiments:
                workload = launcher.beaker.workload.get(exp.id)
                click.echo(f"{exp.id},{exp.name},{BeakerWorkloadStatus(workload.status).name}")

    elif output_format == "json":
        from beaker import BeakerWorkloadStatus

        exp_data = []
        for exp in experiments:
            workload = launcher.beaker.workload.get(exp.id)
            status_enum = BeakerWorkloadStatus(workload.status)
            exp_info = {
                "id": exp.id,
                "name": exp.name,
                "status": status_enum.name,
                "url": launcher.experiment_url(exp),
            }

            if verbose:
                try:
                    task_list = []
                    for task in exp.tasks:
                        task_status = (
                            BeakerWorkloadStatus(task.status).name if task.status else "unknown"
                        )
                        task_list.append({"id": task.id, "name": task.name, "status": task_status})
                    exp_info["tasks"] = task_list
                except Exception:
                    pass

            exp_data.append(exp_info)

        data = {
            "group": group_name,
            "group_id": beaker_group.id,
            "url": group_url,
            "status": status,
            "total_experiments": len(experiments),
            "experiments": exp_data,
        }
        click.echo(json_module.dumps(data, indent=2))
    else:
        console.print(f"\n[bold]Group:[/bold] {group_name}")
        console.print(f"[bold]ID:[/bold] {beaker_group.id}")
        console.print(f"[bold]URL:[/bold] {group_url}")
        console.print()

        total = sum(status.values())
        console.print(
            f"[bold]Status Summary:[/bold] {total} experiment(s)\n"
            f"  [green]\u2713 {status.get('succeeded', 0)} succeeded[/green]\n"
            f"  [yellow]\u25cf {status.get('running', 0)} running[/yellow]\n"
            f"  [dim]\u25cb {status.get('pending', 0)} pending[/dim]\n"
            f"  [red]\u2717 {status.get('failed', 0)} failed[/red]\n"
            f"  [red]\u2298 {status.get('canceled', 0)} canceled[/red]"
        )
        console.print()

        if experiments:
            from beaker import BeakerWorkloadStatus

            table = Table(title="Experiments")
            table.add_column("Name", style="cyan")
            table.add_column("Status")
            if verbose:
                table.add_column("Tasks")
            table.add_column("URL", style="dim")

            for exp in experiments:
                workload = launcher.beaker.workload.get(exp.id)
                status_str = BeakerWorkloadStatus(workload.status).name
                status_style = {
                    "succeeded": "[green]succeeded[/green]",
                    "failed": "[red]failed[/red]",
                    "running": "[yellow]running[/yellow]",
                    "canceled": "[red]canceled[/red]",
                }.get(status_str.lower(), f"[dim]{status_str}[/dim]")

                if verbose:
                    try:
                        task_info = []
                        for task in exp.tasks:
                            task_status = (
                                BeakerWorkloadStatus(task.status).name if task.status else "unknown"
                            )
                            task_info.append(f"{task.name}: {task_status}")
                        task_str = "\n".join(task_info) if task_info else "-"
                    except Exception:
                        task_str = "-"

                    table.add_row(exp.name, status_style, task_str, launcher.experiment_url(exp))
                else:
                    table.add_row(exp.name, status_style, launcher.experiment_url(exp))

            console.print(table)
        else:
            console.print("[dim]No experiments in group.[/dim]")


@group.command(name="cancel")
@click.argument("group_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def group_cancel(group_name: str, yes: bool) -> None:
    """Cancel all active experiments in a Beaker group."""
    launcher = _get_launcher()
    beaker_group = _get_beaker_group(launcher, group_name)

    status = launcher.get_group_status(beaker_group)
    active_count = status.get("running", 0) + status.get("pending", 0)

    if active_count == 0:
        console.print(f"[yellow]No active experiments in group '{group_name}'[/yellow]")
        console.print(
            f"Status: {status.get('succeeded', 0)} succeeded, "
            f"{status.get('failed', 0)} failed, "
            f"{status.get('canceled', 0)} canceled"
        )
        return

    console.print(f"[bold]Group:[/bold] {group_name}")
    console.print(
        f"[bold]Active experiments:[/bold] {active_count} "
        f"({status.get('running', 0)} running, {status.get('pending', 0)} pending)"
    )

    if not yes and not click.confirm(f"Cancel all {active_count} active experiment(s)?"):
        console.print("[dim]Cancelled.[/dim]")
        return

    console.print(f"\n[yellow]Canceling {active_count} experiment(s)...[/yellow]")
    result = launcher.cancel_group(beaker_group)

    console.print(
        f"\n[bold]Results:[/bold]\n"
        f"  [green]\u2713 {result.get('canceled', 0)} canceled[/green]\n"
        f"  [dim]\u25cb {result.get('skipped', 0)} skipped (already completed)[/dim]"
    )
    if result.get("failed", 0) > 0:
        console.print(f"  [red]\u2717 {result.get('failed', 0)} failed to cancel[/red]")


@group.command(name="list")
@click.option("--workspace", "-w", required=True, help="Beaker workspace to list groups from")
@click.option("--limit", "-n", type=int, default=20, help="Number of groups to show")
@click.option("--search", "-s", help="Search by name or description")
@click.option("--mine/--all", default=True, help="Show only my groups (default) or all groups")
def group_list(workspace: str, limit: int, search: str | None, mine: bool) -> None:
    """List Beaker groups."""
    launcher = _get_launcher()
    workspace_obj = launcher.beaker.workspace.get(workspace) if workspace else None

    current_user_id = None
    if mine:
        try:
            current_user_id = launcher.beaker.user.get(launcher.beaker.user_name).id
        except Exception:
            console.print(
                "[yellow]Warning: Could not get current user, showing all groups[/yellow]"
            )

    try:
        fetch_limit = limit * 5 if mine and current_user_id else limit
        all_groups = list(
            launcher.beaker.group.list(
                workspace=workspace_obj,
                name_or_description=search,
                limit=fetch_limit,
            )
        )

        if mine and current_user_id:
            groups = [g for g in all_groups if g.author_id == current_user_id][:limit]
        else:
            groups = all_groups[:limit]
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    if not groups:
        console.print("[dim]No groups found.[/dim]")
        return

    workspace_names: dict[str, str] = {}

    RUNNING_STATUSES = {1, 2, 3, 4, 5, 6, 10}
    SUCCEEDED_STATUS = 8
    FAILED_STATUS = 9

    table = Table(title="Beaker Groups")
    table.add_column("Name", style="cyan")
    table.add_column("Workspace", style="dim")
    table.add_column("Experiments", justify="right")
    table.add_column("Status")
    table.add_column("Created", style="dim")

    for grp in groups:
        try:
            task_metrics = list(launcher.beaker.group.list_task_metrics(grp))

            experiments: dict[str, int] = {}
            for tm in task_metrics:
                exp_id = tm.experiment_id
                if exp_id not in experiments:
                    experiments[exp_id] = tm.task_status
                elif tm.task_status == FAILED_STATUS:
                    experiments[exp_id] = FAILED_STATUS
                elif tm.task_status in RUNNING_STATUSES and experiments[exp_id] == SUCCEEDED_STATUS:
                    experiments[exp_id] = tm.task_status

            exp_count = len(experiments)

            if exp_count > 0:
                succeeded = sum(1 for s in experiments.values() if s == SUCCEEDED_STATUS)
                failed = sum(1 for s in experiments.values() if s == FAILED_STATUS)
                running = sum(1 for s in experiments.values() if s in RUNNING_STATUSES)
                status_str = (
                    f"[green]{succeeded}[/green]/[yellow]{running}[/yellow]/[red]{failed}[/red]"
                )
            else:
                status_str = "[dim]empty[/dim]"

            created_str = "-"
            if grp.created and grp.created.seconds:
                from datetime import datetime

                created_dt = datetime.fromtimestamp(grp.created.seconds, tz=UTC)
                created_str = created_dt.strftime("%Y-%m-%d %H:%M")

            workspace_name = "-"
            if grp.workspace_id:
                if grp.workspace_id not in workspace_names:
                    try:
                        ws = launcher.beaker.workspace.get(grp.workspace_id)
                        workspace_names[grp.workspace_id] = ws.name
                    except Exception:
                        workspace_names[grp.workspace_id] = grp.workspace_id
                workspace_name = workspace_names[grp.workspace_id]

            table.add_row(grp.full_name, workspace_name, str(exp_count), status_str, created_str)
        except Exception:
            table.add_row(grp.full_name, "-", "?", "[dim]error[/dim]", "-")

    console.print(table)

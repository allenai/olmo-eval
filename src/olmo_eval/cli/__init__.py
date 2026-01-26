"""olmo-eval CLI entry point."""

import click
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.evals.tasks  # noqa: F401 - triggers task registration
from olmo_eval.cli.beaker import beaker
from olmo_eval.cli.results import results
from olmo_eval.cli.run import run
from olmo_eval.cli.utils import console
from olmo_eval.core import get_model_presets
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.evals.tasks import list_regimes, list_tasks, list_variants


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    pass


# Register command groups
main.add_command(run)
main.add_command(beaker)
main.add_command(results)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def tasks(filter: str) -> None:
    """List all available tasks in the registry."""
    task_names = list_tasks()
    variants = list_variants()
    regimes = list_regimes()

    if not task_names:
        console.print("[dim]No tasks registered.[/dim]")
        return

    table = Table(title="Available Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Variants", style="green")
    table.add_column("Regimes", style="dim")

    for name in task_names:
        if filter.lower() in name.lower():
            task_variants = variants.get(name, [])
            task_regimes = regimes.get(name, [])
            variant_str = ", ".join(task_variants) if task_variants else "-"
            regime_str = ", ".join(task_regimes) if task_regimes else "-"
            table.add_row(name, variant_str, regime_str)

    console.print(table)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def models(filter: str) -> None:
    """List available model presets."""
    table = Table(title="Model Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Model", style="dim")

    for name, cfg in sorted(get_model_presets().items()):
        if filter.lower() in name.lower():
            table.add_row(name, cfg.model)

    console.print(table)


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def suites(filter: str) -> None:
    """List available task suites (task groups)."""
    table = Table(title="Task Suites")
    table.add_column("Suite", style="cyan")
    table.add_column("Tasks", style="dim")
    table.add_column("Aggregation", style="yellow")

    for name in list_suites():
        if filter.lower() in name.lower():
            suite = get_suite(name)
            task_count = len(suite.expanded_tasks)
            table.add_row(name, f"{task_count} tasks", suite.aggregation.value)

    console.print(table)


@main.command(name="suite-info")
@click.argument("suite_name")
def suite_info(suite_name: str) -> None:
    """Show tasks and regimes in a suite.

    SUITE_NAME is the name of the suite to inspect.

    Example: olmo-eval suite-info core
    """
    try:
        suite = get_suite(suite_name)
    except KeyError:
        console.print(f"[red]Error:[/red] Suite '{suite_name}' not found")
        console.print(f"\n[dim]Available suites: {', '.join(list_suites())}[/dim]")
        raise SystemExit(1) from None

    console.print(f"\n[bold cyan]Suite:[/bold cyan] {suite.name}")
    if suite.description:
        console.print(f"[dim]{suite.description}[/dim]")
    console.print(f"[bold]Aggregation:[/bold] {suite.aggregation.value}")
    console.print()

    table = Table(title=f"Tasks in '{suite_name}'")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Task", style="cyan")
    table.add_column("Regime", style="yellow")

    for idx, task_spec in enumerate(suite.expanded_tasks, 1):
        if "::" in task_spec:
            task_name, regime = task_spec.split("::", 1)
        else:
            task_name = task_spec
            regime = "(default)"
        table.add_row(str(idx), task_name, regime)

    console.print(table)
    console.print(f"\n[dim]Total: {len(suite.expanded_tasks)} tasks[/dim]")


if __name__ == "__main__":
    main()

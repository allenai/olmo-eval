"""Suite inspection commands."""

import click
from rich.table import Table

from olmo_eval.cli.utils import console


@click.group()
def suite() -> None:
    """Suite inspection commands."""
    pass


@suite.command()
@click.argument("suite_name")
def inspect(suite_name: str) -> None:
    """Inspect the expanded tasks in a suite.

    SUITE_NAME is the name of the suite to inspect.

    Examples:
        olmo-eval suite inspect mmlu
        olmo-eval suite inspect gpqa
    """
    from olmo_eval.evals.suites import get_suite, list_suites

    try:
        suite_obj = get_suite(suite_name)
    except KeyError:
        console.print(f"[red]Error:[/red] Suite '{suite_name}' not found")
        console.print(f"\n[dim]Available suites: {', '.join(list_suites())}[/dim]")
        raise SystemExit(1) from None

    console.print(f"\n[bold cyan]Suite:[/bold cyan] {suite_obj.name}")
    if suite_obj.description:
        console.print(f"[dim]{suite_obj.description}[/dim]")
    console.print(f"[bold]Aggregation:[/bold] {suite_obj.aggregation.value}")
    console.print()

    table = Table(title=f"Tasks in '{suite_name}'")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Task", style="cyan")
    table.add_column("Variant", style="yellow")

    for idx, task_spec in enumerate(suite_obj.expanded_tasks, 1):
        if ":" in task_spec:
            task_name, variant = task_spec.split(":", 1)
        else:
            task_name = task_spec
            variant = "(default)"
        table.add_row(str(idx), task_name, variant)

    console.print(table)
    console.print(f"\n[dim]Total: {len(suite_obj.expanded_tasks)} tasks[/dim]")

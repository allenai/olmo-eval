"""olmo-eval CLI entry point."""

import click
from rich.console import Console
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers task registration
from olmo_eval.tasks import list_tasks

console = Console()


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    pass


@main.command()
def hello() -> None:
    """Print a hello world message."""
    table = Table(title="Hello World")
    table.add_column("Message", style="cyan")
    table.add_column("Status", style="green")
    table.add_row("Hello from olmo-eval!", "OK")
    console.print(table)


@main.command()
def tasks() -> None:
    """List all available tasks in the registry."""
    task_names = list_tasks()
    if not task_names:
        console.print("[dim]No tasks registered.[/dim]")
        return
    table = Table(title="Available Tasks")
    table.add_column("Task", style="cyan")
    for name in task_names:
        table.add_row(name)
    console.print(table)


if __name__ == "__main__":
    main()

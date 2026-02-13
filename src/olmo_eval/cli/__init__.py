"""olmo-eval CLI entry point."""

# Suppress noisy third-party library output BEFORE any imports.
# Must be at the very top to take effect before transformers/datasets load.
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BAR", "1")

import click
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.evals.tasks  # noqa: F401 - triggers task registration
from olmo_eval.cli.beaker import beaker
from olmo_eval.cli.results import results
from olmo_eval.cli.run import run
from olmo_eval.cli.task import task
from olmo_eval.cli.utils import console
from olmo_eval.common.constants import get_model_presets
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.evals.tasks.common import list_regimes, list_tasks, list_variants
from olmo_eval.harness.backends import BACKEND_REGISTRY
from olmo_eval.harness.presets import get_harness_preset, list_harness_presets


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    pass


# Register command groups
main.add_command(run)
main.add_command(beaker)
main.add_command(results)
main.add_command(task)


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


def _format_value(value: object) -> str:
    """Format a config value for display."""
    from dataclasses import fields, is_dataclass

    if value is None:
        return ""
    if is_dataclass(value):
        parts = []
        for f in fields(value):
            v = getattr(value, f.name)
            formatted = _format_value(v)
            if formatted:
                parts.append(f"{f.name}={formatted}")
        return "\n".join(parts) if parts else ""
    if isinstance(value, dict):
        if not value:
            return ""
        return "\n".join(f"{k}={v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        formatted_items = [_format_value(v) for v in value]
        return "\n".join(f for f in formatted_items if f)
    if isinstance(value, bool):
        return str(value) if value else ""
    return str(value) if value else ""


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def models(filter: str) -> None:
    """List available model presets."""
    from dataclasses import fields

    from rich.panel import Panel

    presets = get_model_presets()
    if not presets:
        console.print("[dim]No model presets registered.[/dim]")
        return

    # Get field names from first config
    first_cfg = next(iter(presets.values()))
    field_names = [f.name for f in fields(first_cfg)]
    max_field_width = max(len(f) for f in field_names)

    for name, cfg in sorted(presets.items()):
        if filter.lower() in name.lower():
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Field", style="dim", width=max_field_width)
            table.add_column("Value", overflow="fold")

            for f in fields(cfg):
                value = getattr(cfg, f.name)
                formatted = _format_value(value)
                if formatted:
                    table.add_row(f.name, formatted)

            console.print(Panel(table, title=f"[cyan]{name}[/cyan]", title_align="left"))
            console.print()


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def harnesses(filter: str) -> None:
    """List available harness presets."""
    from rich.panel import Panel
    from rich.pretty import Pretty

    preset_names = list_harness_presets()
    if not preset_names:
        console.print("[dim]No harness presets registered.[/dim]")
        return

    for name in sorted(preset_names):
        if filter.lower() not in name.lower():
            continue

        cfg = get_harness_preset(name)
        console.print(
            Panel(
                Pretty(cfg, expand_all=True),
                title=f"[bold]{name}[/bold]",
                border_style="cyan",
            )
        )
        console.print()


@main.command()
@click.option("--filter", "-f", default="", help="Filter by name substring")
def backends(filter: str) -> None:
    """List available harness backends."""
    table = Table(title="Harness Backends")
    table.add_column("Backend", style="cyan")
    table.add_column("Required Extra", style="yellow")

    for name, cls in sorted(BACKEND_REGISTRY.items()):
        if filter.lower() in name.lower():
            extras = ", ".join(cls.required_extras) if cls.required_extras else "-"
            table.add_row(name, extras)

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
        if ":" in task_spec:
            task_name, variant = task_spec.split(":", 1)
        else:
            task_name = task_spec
            variant = "(default)"
        table.add_row(str(idx), task_name, variant)

    console.print(table)
    console.print(f"\n[dim]Total: {len(suite.expanded_tasks)} tasks[/dim]")


if __name__ == "__main__":
    main()

"""olmo-eval CLI entry point."""

import click
from rich.console import Console
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.tasks  # noqa: F401 - triggers task registration
from olmo_eval.core import get_model_presets
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.tasks import list_tasks
from olmo_eval.tasks.registry import list_regimes

console = Console()


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    pass


@main.command()
@click.option(
    "--model", "-m", required=True, help="Model name or preset (e.g., llama3.1-8b)"
)
@click.option("--task", "-t", multiple=True, required=True, help="Task spec or suite")
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--output-dir", "-o", default="./results", help="Output directory")
@click.option("--num-shots", type=int, help="Override num_fewshot for all tasks")
@click.option("--limit", type=int, help="Override instance limit for all tasks")
@click.option(
    "--backend", type=click.Choice(["hf", "vllm", "litellm"]), help="Override backend"
)
@click.option("--dry-run", is_flag=True, help="Print config and exit without running")
def run(
    model: str,
    task: tuple[str, ...],
    config: str | None,
    output_dir: str,
    num_shots: int | None,
    limit: int | None,
    backend: str | None,
    dry_run: bool,
) -> None:
    """Run evaluation on specified tasks."""
    from olmo_eval.runner import EvalRunner, ValidationError

    runner = EvalRunner(
        model_name=model,
        task_specs=list(task),
        output_dir=output_dir,
        num_shots_override=num_shots,
        limit_override=limit,
        backend_override=backend,
    )

    # Validate inputs before running (applies to both dry-run and actual runs)
    try:
        runner.validate()
    except ValidationError as e:
        console.print(f"[red]Validation error:[/red]\n{e}")
        raise SystemExit(1)

    if dry_run:
        runner.print_config()
    else:
        runner.run()


@main.command()
def tasks() -> None:
    """List all available tasks in the registry."""
    task_names = list_tasks()
    regimes = list_regimes()

    if not task_names:
        console.print("[dim]No tasks registered.[/dim]")
        return

    table = Table(title="Available Tasks")
    table.add_column("Task", style="cyan")
    table.add_column("Regimes", style="dim")

    for name in task_names:
        task_regimes = regimes.get(name, [])
        regime_str = ", ".join(task_regimes) if task_regimes else "-"
        table.add_row(name, regime_str)

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


if __name__ == "__main__":
    main()

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
@click.option("--model", "-m", required=True, help="Model name or preset (e.g., llama3.1-8b)")
@click.option("--task", "-t", multiple=True, required=True, help="Task spec or suite")
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--output-dir", "-o", default="./results", help="Output directory")
@click.option("--num-shots", type=int, help="Override num_fewshot for all tasks")
@click.option("--limit", type=int, help="Override instance limit for all tasks")
@click.option("--backend", type=click.Choice(["hf", "vllm", "litellm"]), help="Override backend")
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
        raise SystemExit(1) from None

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


@main.command()
@click.option(
    "--config",
    "-f",
    type=click.Path(exists=True),
    help="YAML config file (CLI args override config values)",
)
@click.option("--name", "-n", help="Experiment name")
@click.option(
    "--model",
    "-m",
    multiple=True,
    help="Model name or preset (can specify multiple)",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    help="Task name with optional @priority suffix (e.g., mmlu, mmlu@high)",
)
@click.option("--cluster", "-c", default=None, help="Cluster alias (h100, a100, aus) or full name")
@click.option("--gpus", default=None, type=int, help="Number of GPUs")
@click.option(
    "--priority",
    default=None,
    type=click.Choice(["low", "normal", "high", "urgent"]),
    help="Job priority",
)
@click.option("--preemptible/--no-preemptible", default=None, help="Allow preemption")
@click.option("--timeout", default=None, help="Job timeout (e.g., 24h, 30m)")
@click.option("--retries", type=int, help="Number of retries on failure")
@click.option("--workspace", help="Beaker workspace")
@click.option("--budget", help="Beaker budget")
@click.option("--dry-run", is_flag=True, help="Print spec without launching")
def launch(
    config: str | None,
    name: str | None,
    model: tuple[str, ...],
    task: tuple[str, ...],
    cluster: str | None,
    gpus: int | None,
    priority: str | None,
    preemptible: bool | None,
    timeout: str | None,
    retries: int | None,
    workspace: str | None,
    budget: str | None,
    dry_run: bool,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.

    Examples:

        olmo-eval launch -n "eval-llama3" -m llama3.1-8b -t mmlu

        olmo-eval launch -n "eval-suite" -m llama3.1-8b -t mmlu -t gsm8k -t arc

        olmo-eval launch -n "eval-70b" -m llama3.1-70b -t mmlu --cluster h100 --gpus 4

        # Multiple models (creates separate experiments per model)
        olmo-eval launch -n "eval-compare" -m llama3.1-8b -m olmo-2-7b -t mmlu -t gsm8k

        # Per-task priorities (creates separate experiments per priority level)
        olmo-eval launch -n "eval-mixed" -m llama3.1-8b -t "mmlu@high" -t "gsm8k@normal"

        # From YAML config file
        olmo-eval launch -f eval_config.yaml

        # Config file with CLI overrides
        olmo-eval launch -f eval_config.yaml --gpus 4 --priority high
    """
    from collections import defaultdict

    try:
        from olmo_eval.launch import (
            BeakerJobConfig,
            BeakerLauncher,
            LaunchConfig,
            parse_task_with_priority,
        )
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    from olmo_eval.core.constants.infrastructure import (
        BEAKER_DEFAULT_BUDGET,
        BEAKER_DEFAULT_WORKSPACE,
    )

    # Load config from file if provided
    if config:
        try:
            cfg = LaunchConfig.from_yaml(config)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"[red]Config error:[/red] {e}")
            raise SystemExit(1) from None

        # Use config values as defaults, CLI args override
        name = name or cfg.name
        model = model if model else tuple(cfg.models)
        task = task if task else tuple(cfg.tasks)
        cluster = cluster if cluster is not None else cfg.cluster
        gpus = gpus if gpus is not None else cfg.gpus
        priority = priority if priority is not None else cfg.priority
        preemptible = preemptible if preemptible is not None else cfg.preemptible
        timeout = timeout if timeout is not None else cfg.timeout
        retries = retries if retries is not None else cfg.retries
        workspace = workspace or cfg.workspace
        budget = budget or cfg.budget

    # Apply defaults for values not set by config or CLI
    cluster = cluster or "h100"
    gpus = gpus if gpus is not None else 1
    priority = priority or "normal"
    preemptible = preemptible if preemptible is not None else True
    timeout = timeout or "24h"

    # Validate required fields
    if not name:
        console.print("[red]Error:[/red] --name/-n is required (or set 'name' in config)")
        raise SystemExit(1)
    if not model:
        console.print("[red]Error:[/red] --model/-m is required (or set 'models' in config)")
        raise SystemExit(1)
    if not task:
        console.print("[red]Error:[/red] --task/-t is required (or set 'tasks' in config)")
        raise SystemExit(1)

    # Parse tasks and group by priority
    tasks_by_priority: dict[str, list[str]] = defaultdict(list)
    try:
        for t in task:
            task_name, task_priority = parse_task_with_priority(t, default_priority=priority)
            tasks_by_priority[task_priority].append(task_name)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    launcher = BeakerLauncher(workspace=workspace or BEAKER_DEFAULT_WORKSPACE)
    multiple_models = len(model) > 1
    multiple_priorities = len(tasks_by_priority) > 1

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Launch one experiment per model and priority level
    for model_name in model:
        # Get short model name for experiment naming (last part after /)
        short_model = model_name.split("/")[-1].lower()

        for task_priority, task_list in tasks_by_priority.items():
            # Build experiment name with model and/or priority suffix as needed
            exp_name = name
            if multiple_models:
                exp_name = f"{exp_name}-{short_model}"
            if multiple_priorities:
                exp_name = f"{exp_name}-{task_priority}"

            # Build command with this model and priority's tasks
            command = ["olmo-eval", "run", "-m", model_name]
            for t in task_list:
                command.extend(["-t", t])

            job_config = BeakerJobConfig(
                name=exp_name,
                command=command,
                cluster=cluster,
                num_gpus=gpus,
                priority=task_priority,
                preemptible=preemptible,
                timeout=timeout,
                retries=retries,
                workspace=workspace or BEAKER_DEFAULT_WORKSPACE,
                budget=budget or BEAKER_DEFAULT_BUDGET,
            )

            if dry_run:
                if multiple_models or multiple_priorities:
                    console.print()  # Add spacing between multiple experiments
                launcher.launch(job_config, dry_run=True)
            else:
                experiment = launcher.launch(job_config)
                if experiment:
                    console.print(f"[green]Launched:[/green] {launcher.experiment_url(experiment)}")


if __name__ == "__main__":
    main()

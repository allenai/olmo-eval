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
@click.option(
    "--storage-backend",
    type=click.Choice(["file", "s3", "postgres"]),
    default=None,
    help="Storage backend for results (default: legacy file output)",
)
@click.option(
    "--storage-config",
    type=click.Path(exists=True),
    help="YAML config file for storage backend",
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
    storage_backend: str | None,
    storage_config: str | None,
    dry_run: bool,
) -> None:
    """Run evaluation on specified tasks."""
    from olmo_eval.runner import EvalRunner, ValidationError

    # Set up storage backend if specified
    storage = None
    if storage_backend:
        from olmo_eval.storage import get_backend

        # Load storage config if provided
        storage_kwargs: dict = {}
        if storage_config:
            from omegaconf import DictConfig, OmegaConf

            cfg = OmegaConf.load(storage_config)
            # Get backend-specific config section
            if isinstance(cfg, DictConfig):
                backend_cfg = cfg.get(storage_backend, {})
                storage_kwargs = OmegaConf.to_container(backend_cfg, resolve=True) or {}  # type: ignore
            else:
                console.print("[red]Error:[/red] Storage config must be a YAML dict, not a list")
                raise SystemExit(1)

        # Add default output_dir for file backend
        if storage_backend == "file" and "output_dir" not in storage_kwargs:
            storage_kwargs["output_dir"] = output_dir

        try:
            storage = get_backend(storage_backend, **storage_kwargs)
        except ImportError as e:
            console.print(f"[red]Storage backend error:[/red] {e}")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"[red]Failed to initialize storage backend:[/red] {e}")
            raise SystemExit(1) from None

    runner = EvalRunner(
        model_name=model,
        task_specs=list(task),
        output_dir=output_dir,
        num_shots_override=num_shots,
        limit_override=limit,
        backend_override=backend,
        storage=storage,
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
@click.option("--group", "-g", help="Add experiments to this Beaker group (creates if needed)")
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
    group: str | None,
    dry_run: bool,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.
    Use --group/-g to organize experiments into a Beaker group for result aggregation.

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

        # With grouping for result aggregation
        olmo-eval launch -n "benchmark" --group "benchmark-2024" -m llama3.1-8b -t mmlu -t gsm8k
    """
    from collections import defaultdict

    try:
        from olmo_eval.launch import (
            BeakerJobConfig,
            BeakerLauncher,
            LaunchConfig,
            ModelConfig,
            parse_model_config,
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

    # Track which CLI args were explicitly set (vs using defaults)
    cli_cluster = cluster
    cli_gpus = gpus
    cli_priority = priority
    cli_preemptible = preemptible
    cli_timeout = timeout

    # Load config from file if provided
    cfg: LaunchConfig | None = None
    model_configs: list[ModelConfig] = []

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
        task = task if task else tuple(cfg.tasks)
        retries = retries if retries is not None else cfg.retries
        workspace = workspace or cfg.workspace
        budget = budget or cfg.budget

        # Get model configs from file (with per-model resource overrides)
        if not model:
            model_configs = cfg.get_model_configs()
        else:
            # CLI models override config file models
            model_configs = [parse_model_config(m) for m in model]

        # Set defaults from config (will be overridden by per-model or CLI)
        cluster = cluster if cluster is not None else cfg.cluster
        gpus = gpus if gpus is not None else cfg.gpus
        priority = priority if priority is not None else cfg.priority
        preemptible = preemptible if preemptible is not None else cfg.preemptible
        timeout = timeout if timeout is not None else cfg.timeout
    else:
        # No config file - use CLI models
        model_configs = [parse_model_config(m) for m in model] if model else []

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
    if not model_configs:
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
    multiple_models = len(model_configs) > 1
    multiple_priorities = len(tasks_by_priority) > 1

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Track launched experiments for grouping
    launched_experiments: list[str] = []

    # Launch one experiment per model and priority level
    for model_cfg in model_configs:
        model_name = model_cfg.name

        # Get effective resources for this model (per-model overrides merged with defaults)
        if cfg is not None:
            model_resources = cfg.get_model_resources(model_cfg)
        else:
            # No config file - use ModelConfig values or defaults
            model_resources = {
                "gpus": model_cfg.gpus if model_cfg.gpus is not None else gpus,
                "cluster": model_cfg.cluster if model_cfg.cluster is not None else cluster,
                "priority": model_cfg.priority if model_cfg.priority is not None else priority,
                "preemptible": (
                    model_cfg.preemptible if model_cfg.preemptible is not None else preemptible
                ),
                "timeout": model_cfg.timeout if model_cfg.timeout is not None else timeout,
                "shared_memory": model_cfg.shared_memory,
            }

        # CLI args always override per-model config
        # Cast values from model_resources dict to expected types
        effective_cluster: str = (
            cli_cluster if cli_cluster is not None else str(model_resources["cluster"])
        )
        res_gpus = model_resources["gpus"]
        effective_gpus: int = (
            cli_gpus if cli_gpus is not None else (int(res_gpus) if res_gpus else 1)
        )
        effective_preemptible: bool = (
            cli_preemptible if cli_preemptible is not None else bool(model_resources["preemptible"])
        )
        effective_timeout: str = (
            cli_timeout if cli_timeout is not None else str(model_resources["timeout"])
        )
        res_shared_memory = model_resources.get("shared_memory")
        effective_shared_memory: str = str(res_shared_memory) if res_shared_memory else "10GiB"

        # Get short model name for experiment naming (last part after /)
        short_model = model_name.split("/")[-1].lower()

        for task_priority, task_list in tasks_by_priority.items():
            # CLI priority override applies to task priorities too
            effective_priority = cli_priority if cli_priority is not None else task_priority

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
                cluster=effective_cluster,
                num_gpus=effective_gpus,
                priority=effective_priority,
                preemptible=effective_preemptible,
                timeout=effective_timeout,
                shared_memory=effective_shared_memory,
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
                    launched_experiments.append(experiment.id)

    # Add experiments to group if specified
    if group and launched_experiments and not dry_run:
        try:
            beaker_group = launcher.get_or_create_group(
                name=group,
                workspace=workspace or BEAKER_DEFAULT_WORKSPACE,
            )
            launcher.add_experiments_to_group(beaker_group, launched_experiments)
            console.print(
                f"[blue]Group:[/blue] Added {len(launched_experiments)} experiment(s) to '{group}'"
            )
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to add experiments to group: {e}")


@main.command()
@click.option("--group", "-g", required=True, help="Beaker group name")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    help="Output format",
)
@click.option("--wait", is_flag=True, help="Wait for all experiments to complete")
@click.option(
    "--poll-interval",
    type=int,
    default=30,
    help="Seconds between status checks when waiting",
)
def results(
    group: str,
    output_format: str,
    wait: bool,
    poll_interval: int,
) -> None:
    """Show results from a Beaker group.

    Displays status and metrics for all experiments in a Beaker group.
    Use --wait to block until all experiments complete.

    Examples:

        # Show status table
        olmo-eval results --group "benchmark-2024"

        # Export as CSV
        olmo-eval results --group "benchmark-2024" --format csv > results.csv

        # Wait for completion then show results
        olmo-eval results --group "benchmark-2024" --wait
    """
    import json as json_module
    import time

    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    # Try to get the group
    try:
        from beaker.exceptions import BeakerGroupNotFound

        beaker_group = launcher.beaker.group.get(group)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Wait for completion if requested
    if wait:
        console.print(f"[dim]Waiting for experiments in '{group}' to complete...[/dim]")
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

    # Get status summary
    status = launcher.get_group_status(beaker_group)
    experiments = launcher.get_group_experiments(beaker_group)

    if output_format == "csv":
        # Export raw metrics CSV from Beaker
        try:
            csv_data = launcher.export_group_metrics(beaker_group)
            click.echo(csv_data)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not export metrics: {e}")
            # Fall back to basic experiment info
            click.echo("experiment_id,name,status")
            for exp in experiments:
                workload = launcher.beaker.workload.get(exp.id)
                click.echo(f"{exp.id},{exp.name},{workload.status.name}")

    elif output_format == "json":
        # Export as JSON
        data = {
            "group": group,
            "status": status,
            "experiments": [
                {
                    "id": exp.id,
                    "name": exp.name,
                    "status": launcher.beaker.workload.get(exp.id).status.name,
                    "url": launcher.experiment_url(exp),
                }
                for exp in experiments
            ],
        }
        click.echo(json_module.dumps(data, indent=2))

    else:
        # Table format (default)
        console.print(f"[bold]Group:[/bold] {group}")
        console.print(
            f"[bold]Status:[/bold] "
            f"[green]{status.get('succeeded', 0)} succeeded[/green], "
            f"[yellow]{status.get('running', 0)} running[/yellow], "
            f"[dim]{status.get('pending', 0)} pending[/dim], "
            f"[red]{status.get('failed', 0)} failed[/red]"
        )
        console.print()

        if experiments:
            table = Table(title="Experiments")
            table.add_column("Name", style="cyan")
            table.add_column("Status")
            table.add_column("URL", style="dim")

            for exp in experiments:
                workload = launcher.beaker.workload.get(exp.id)
                status_str = workload.status.name
                status_style = {
                    "succeeded": "[green]succeeded[/green]",
                    "failed": "[red]failed[/red]",
                    "running": "[yellow]running[/yellow]",
                    "canceled": "[red]canceled[/red]",
                }.get(status_str.lower(), f"[dim]{status_str}[/dim]")

                table.add_row(
                    exp.name,
                    status_style,
                    launcher.experiment_url(exp),
                )

            console.print(table)
        else:
            console.print("[dim]No experiments in group.[/dim]")


if __name__ == "__main__":
    main()

"""olmo-eval CLI entry point."""

from datetime import UTC

import click
from rich.console import Console
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.evals.tasks  # noqa: F401 - triggers task registration
from olmo_eval.core import get_model_presets
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.evals.tasks import list_tasks
from olmo_eval.evals.tasks.registry import list_regimes

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
@click.option(
    "--async",
    "use_async",
    is_flag=True,
    help="Use async runner for parallel task execution",
)
@click.option(
    "--num-workers",
    type=int,
    default=None,
    help="Number of workers for async mode (default: auto-detect from GPUs)",
)
@click.option(
    "--gpus-per-worker",
    type=int,
    default=1,
    help="Number of GPUs each worker uses (default: 1)",
)
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
    use_async: bool,
    num_workers: int | None,
    gpus_per_worker: int,
) -> None:
    """Run evaluation on specified tasks."""
    from olmo_eval.runners.sequential import EvalRunner, ValidationError

    # Warning for num-workers without async
    if num_workers is not None and not use_async:
        console.print("[yellow]Warning:[/yellow] --num-workers has no effect without --async flag")

    if gpus_per_worker != 1 and not use_async:
        console.print(
            "[yellow]Warning:[/yellow] --gpus-per-worker has no effect without --async flag"
        )

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

    # Choose runner based on --async flag
    if use_async:
        from olmo_eval.runners.parallel import AsyncEvalRunner

        console.print("[bold cyan]Using AsyncEvalRunner[/bold cyan] - parallel execution enabled")

        runner = AsyncEvalRunner(
            model_name=model,
            task_specs=list(task),
            output_dir=output_dir,
            num_shots_override=num_shots,
            limit_override=limit,
            backend_override=backend,
            storage=storage,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
        )
    else:
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

    # Header with suite info
    console.print(f"\n[bold cyan]Suite:[/bold cyan] {suite.name}")
    if suite.description:
        console.print(f"[dim]{suite.description}[/dim]")
    console.print(f"[bold]Aggregation:[/bold] {suite.aggregation.value}")
    console.print()

    # Table of tasks
    table = Table(title=f"Tasks in '{suite_name}'")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Task", style="cyan")
    table.add_column("Regime", style="yellow")

    for idx, task_spec in enumerate(suite.expanded_tasks, 1):
        # Parse task::regime format
        if "::" in task_spec:
            task_name, regime = task_spec.split("::", 1)
        else:
            task_name = task_spec
            regime = "(default)"
        table.add_row(str(idx), task_name, regime)

    console.print(table)
    console.print(f"\n[dim]Total: {len(suite.expanded_tasks)} tasks[/dim]")


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
@click.option(
    "--backends", "-b", multiple=True, help="Backends to install at runtime (e.g., vllm==0.13.0)"
)
@click.option("--async", "use_async", is_flag=True, help="Enable parallel task execution")
@click.option("--num-workers", type=int, help="Number of workers for async mode")
@click.option("--gpus-per-worker", type=int, default=1, help="GPUs per worker for async mode")
@click.option(
    "--flash-attn",
    type=click.Choice(["2", "3", "none"]),
    default=None,
    help="Flash Attention version to install (2, 3, or none)",
)
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
    backends: tuple[str, ...],
    use_async: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    flash_attn: str | None,
    dry_run: bool,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.
    Use --group/-g to organize experiments into a Beaker group for result aggregation.
    Use --backends/-b to install inference backends at runtime (e.g., vllm, transformers).
    Use --flash-attn to install Flash Attention at runtime (2 for FA2, 3 for FA3).

    Examples:

        olmo-eval launch -n "eval-llama3" -m llama3.1-8b -t mmlu

        olmo-eval launch -n "eval-suite" -m llama3.1-8b -t mmlu -t gsm8k -t arc

        olmo-eval launch -n "eval-70b" -m llama3.1-70b -t mmlu --cluster h100 --gpus 4

        # Multiple models (creates separate experiments per model)
        olmo-eval launch -n "eval-compare" -m llama3.1-8b -m olmo-2-7b -t mmlu -t gsm8k

        # Per-task priorities (creates separate experiments per priority level)
        olmo-eval launch -n "eval-mixed" -m llama3.1-8b -t "mmlu@high" -t "gsm8k@normal"

        # Install backends at runtime
        olmo-eval launch -n "eval-vllm" -m llama3.1-8b -t mmlu -b vllm==0.13.0

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
        backends = backends if backends else (tuple(cfg.backends) if cfg.backends else ())
        retries = retries if retries is not None else cfg.retries
        workspace = workspace or cfg.workspace
        budget = budget or cfg.budget

        # Flash Attention: CLI overrides config
        if flash_attn is None and cfg.flash_attn is not None:
            flash_attn = str(cfg.flash_attn)

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
        use_async = use_async or cfg.use_async
        num_workers = num_workers if num_workers is not None else cfg.num_workers
        gpus_per_worker = gpus_per_worker if gpus_per_worker != 1 else cfg.gpus_per_worker
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

    # Validate all tasks exist before launching to Beaker
    from olmo_eval.core.configs import validate_tasks

    all_task_specs = [t for tasks in tasks_by_priority.values() for t in tasks]
    valid_tasks, invalid_tasks = validate_tasks(all_task_specs)

    if invalid_tasks:
        console.print("[red]Error:[/red] The following tasks/suites do not exist:")
        for inv in invalid_tasks:
            console.print(f"  - {inv}")
        console.print("\nUse 'olmo-eval list tasks' to see available tasks.")
        console.print("Use 'olmo-eval list suites' to see available suites.")
        raise SystemExit(1)

    launcher = BeakerLauncher(workspace=workspace or BEAKER_DEFAULT_WORKSPACE)
    multiple_models = len(model_configs) > 1
    multiple_priorities = len(tasks_by_priority) > 1

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Always use groups for all experiments
    # Auto-generate group name from experiment name if not specified
    from datetime import datetime

    effective_group = group or f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    console.print(f"[blue]Group:[/blue] {effective_group}")

    # Pre-create the group so it exists when experiments reference it
    if not dry_run:
        try:
            beaker_group = launcher.get_or_create_group(
                name=effective_group,
                workspace=workspace or BEAKER_DEFAULT_WORKSPACE,
            )
            group_url = launcher.get_group_url(beaker_group)
            console.print(f"[blue]Group URL:[/blue] {group_url}")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to create group: {e}")
            effective_group = None  # Fall back to no group if creation fails

    # Track launched experiments
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
                "cluster": (model_cfg.cluster if model_cfg.cluster is not None else cluster),
                "priority": (model_cfg.priority if model_cfg.priority is not None else priority),
                "preemptible": (
                    model_cfg.preemptible if model_cfg.preemptible is not None else preemptible
                ),
                "timeout": (model_cfg.timeout if model_cfg.timeout is not None else timeout),
                "shared_memory": model_cfg.shared_memory,
                "backend": model_cfg.backend,
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

            # Add async flags if enabled
            model_use_async = model_resources.get("use_async", False)
            model_num_workers = model_resources.get("num_workers")
            model_gpus_per_worker = model_resources.get("gpus_per_worker", 1)

            if model_use_async:
                command.append("--async")
                if model_num_workers is not None:
                    command.extend(["--num-workers", str(model_num_workers)])
                if model_gpus_per_worker and model_gpus_per_worker != 1:
                    command.extend(["--gpus-per-worker", str(model_gpus_per_worker)])

            # Determine the backend this model will use at runtime
            # First check for explicit backend override in config, then get from model config
            from olmo_eval.core.configs import get_model_config as get_runtime_model_config
            from olmo_eval.core.constants.infrastructure import BACKEND_DEPENDENCIES

            config_backend = model_resources.get("backend")  # Explicit override from launch config
            if config_backend:
                runtime_backend = config_backend
            else:
                # Get the backend from model config (preset or default)
                runtime_model_config = get_runtime_model_config(model_name)
                runtime_backend = runtime_model_config.backend

            # CLI backends override auto-detected backend dependency
            if backends:
                effective_backends = list(backends)
            else:
                # Auto-install the backend dependency with version spec
                backend_dep = BACKEND_DEPENDENCIES.get(runtime_backend)
                effective_backends = [backend_dep] if backend_dep else []

            # Convert flash_attn string to int (None, 2, or 3)
            effective_flash_attn: int | None = None
            if flash_attn is not None and flash_attn != "none":
                effective_flash_attn = int(flash_attn)

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
                backends=effective_backends,
                flash_attn=effective_flash_attn,
                group=effective_group,
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

    # Summary of launched experiments
    if launched_experiments and not dry_run:
        console.print(f"\n[bold]Launched {len(launched_experiments)} experiment(s)[/bold]")


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


@main.group()
def group() -> None:
    """Manage Beaker groups.

    Commands for viewing group status, getting detailed task info,
    and bulk operations like canceling all experiments.
    """
    pass


@group.command(name="info")
@click.argument("group_name")
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed task info")
def group_info(group_name: str, output_format: str, verbose: bool) -> None:
    """Get detailed info about a Beaker group.

    Shows status of all experiments and tasks in the group.

    Examples:

        olmo-eval group info my-experiment-group

        olmo-eval group info my-experiment-group --verbose

        olmo-eval group info my-experiment-group --format json
    """
    import json as json_module

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

        beaker_group = launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Get status summary
    status = launcher.get_group_status(beaker_group)
    experiments = launcher.get_group_experiments(beaker_group)
    group_url = launcher.get_group_url(beaker_group)

    if output_format == "json":
        # Build detailed experiment data
        exp_data = []
        for exp in experiments:
            workload = launcher.beaker.workload.get(exp.id)
            exp_info = {
                "id": exp.id,
                "name": exp.name,
                "status": workload.status.name,
                "url": launcher.experiment_url(exp),
            }

            # Add task-level details if verbose
            if verbose:
                try:
                    tasks = list(launcher.beaker.experiment.tasks(exp))
                    task_list = []
                    for task in tasks:
                        job = launcher.beaker.job.get(task.latest_job) if task.latest_job else None
                        task_list.append(
                            {
                                "id": task.id,
                                "name": task.name,
                                "status": job.status.current if job else "unknown",
                                "exit_code": job.status.exit_code if job and job.status else None,
                            }
                        )
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
        # Table format
        console.print(f"\n[bold]Group:[/bold] {group_name}")
        console.print(f"[bold]ID:[/bold] {beaker_group.id}")
        console.print(f"[bold]URL:[/bold] {group_url}")
        console.print()

        # Status summary
        total = sum(status.values())
        console.print(
            f"[bold]Status Summary:[/bold] {total} experiment(s)\n"
            f"  [green]✓ {status.get('succeeded', 0)} succeeded[/green]\n"
            f"  [yellow]● {status.get('running', 0)} running[/yellow]\n"
            f"  [dim]○ {status.get('pending', 0)} pending[/dim]\n"
            f"  [red]✗ {status.get('failed', 0)} failed[/red]\n"
            f"  [red]⊘ {status.get('canceled', 0)} canceled[/red]"
        )
        console.print()

        if experiments:
            table = Table(title="Experiments")
            table.add_column("Name", style="cyan")
            table.add_column("Status")
            if verbose:
                table.add_column("Tasks")
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

                if verbose:
                    # Get task-level details
                    try:
                        tasks = list(launcher.beaker.experiment.tasks(exp))
                        task_info = []
                        for task in tasks:
                            job = (
                                launcher.beaker.job.get(task.latest_job)
                                if task.latest_job
                                else None
                            )
                            task_status = job.status.current if job else "unknown"
                            task_info.append(f"{task.name}: {task_status}")
                        task_str = "\n".join(task_info) if task_info else "-"
                    except Exception:
                        task_str = "-"

                    table.add_row(
                        exp.name,
                        status_style,
                        task_str,
                        launcher.experiment_url(exp),
                    )
                else:
                    table.add_row(
                        exp.name,
                        status_style,
                        launcher.experiment_url(exp),
                    )

            console.print(table)
        else:
            console.print("[dim]No experiments in group.[/dim]")


@group.command(name="cancel")
@click.argument("group_name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
def group_cancel(group_name: str, yes: bool) -> None:
    """Cancel all active experiments in a Beaker group.

    Stops all running and pending experiments. Completed experiments are skipped.

    Examples:

        olmo-eval group cancel my-experiment-group

        olmo-eval group cancel my-experiment-group --yes
    """
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

        beaker_group = launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Get current status to show what will be affected
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

    # Confirm cancellation
    console.print(f"[bold]Group:[/bold] {group_name}")
    console.print(
        f"[bold]Active experiments:[/bold] {active_count} "
        f"({status.get('running', 0)} running, {status.get('pending', 0)} pending)"
    )

    if not yes and not click.confirm(f"Cancel all {active_count} active experiment(s)?"):
        console.print("[dim]Cancelled.[/dim]")
        return

    # Perform cancellation
    console.print(f"\n[yellow]Canceling {active_count} experiment(s)...[/yellow]")
    result = launcher.cancel_group(beaker_group)

    # Show results
    console.print(
        f"\n[bold]Results:[/bold]\n"
        f"  [green]✓ {result.get('canceled', 0)} canceled[/green]\n"
        f"  [dim]○ {result.get('skipped', 0)} skipped (already completed)[/dim]"
    )
    if result.get("failed", 0) > 0:
        console.print(f"  [red]✗ {result.get('failed', 0)} failed to cancel[/red]")


@group.command(name="list")
@click.option("--workspace", "-w", help="Filter by workspace")
@click.option("--limit", "-n", type=int, default=20, help="Number of groups to show")
@click.option("--search", "-s", help="Search by name or description")
@click.option("--mine/--all", default=True, help="Show only my groups (default) or all groups")
def group_list(workspace: str | None, limit: int, search: str | None, mine: bool) -> None:
    """List Beaker groups.

    Shows recent groups with their status summaries. By default, only shows
    groups created by the current user. Use --all to show all groups.

    Examples:

        olmo-eval group list

        olmo-eval group list --all

        olmo-eval group list --workspace ai2/oe-data

        olmo-eval group list --search "benchmark" --limit 10
    """
    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    from olmo_eval.core.constants.infrastructure import BEAKER_DEFAULT_WORKSPACE

    launcher = BeakerLauncher()

    # Get current user ID for filtering
    current_user_id = None
    if mine:
        try:
            current_user_id = launcher.beaker.user.get(launcher.beaker.user_name).id
        except Exception:
            console.print(
                "[yellow]Warning: Could not get current user, showing all groups[/yellow]"
            )

    try:
        # Fetch more than limit if filtering by user, since we filter client-side
        fetch_limit = limit * 5 if mine and current_user_id else limit
        all_groups = list(
            launcher.beaker.group.list(
                workspace=workspace or BEAKER_DEFAULT_WORKSPACE,
                name_or_description=search,
                limit=fetch_limit,
            )
        )

        # Filter to current user's groups if requested
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

    # Cache workspace lookups
    workspace_names: dict[str, str] = {}

    # Status value mappings (from BeakerWorkloadStatus)
    RUNNING_STATUSES = {1, 2, 3, 4, 5, 6, 10}  # submitted, queued, initializing, running, etc.
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
            # Get experiment info from task metrics
            task_metrics = list(launcher.beaker.group.list_task_metrics(grp))

            # Count unique experiments and their statuses
            experiments: dict[str, int] = {}  # exp_id -> worst status
            for tm in task_metrics:
                exp_id = tm.experiment_id
                # Keep the worst status (failed > running > succeeded)
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

            # Format creation time from protobuf Timestamp
            created_str = "-"
            if grp.created and grp.created.seconds:
                from datetime import datetime

                created_dt = datetime.fromtimestamp(grp.created.seconds, tz=UTC)
                created_str = created_dt.strftime("%Y-%m-%d %H:%M")

            # Get workspace name (with caching)
            workspace_name = "-"
            if grp.workspace_id:
                if grp.workspace_id not in workspace_names:
                    try:
                        ws = launcher.beaker.workspace.get(grp.workspace_id)
                        workspace_names[grp.workspace_id] = ws.name
                    except Exception:
                        workspace_names[grp.workspace_id] = grp.workspace_id
                workspace_name = workspace_names[grp.workspace_id]

            table.add_row(
                grp.name,
                workspace_name,
                str(exp_count),
                status_str,
                created_str,
            )
        except Exception:
            table.add_row(grp.name, "-", "?", "[dim]error[/dim]", "-")

    console.print(table)


if __name__ == "__main__":
    main()

"""olmo-eval CLI entry point."""

from datetime import UTC

import click
from rich.console import Console
from rich.table import Table

import olmo_eval.evals  # noqa: F401 - triggers suite registration
import olmo_eval.evals.tasks  # noqa: F401 - triggers task registration
from olmo_eval.core import get_model_presets
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR, DEFAULT_MAX_GPUS_PER_NODE
from olmo_eval.evals.suites import get_suite, list_suites
from olmo_eval.evals.tasks import list_tasks
from olmo_eval.evals.tasks.registry import list_regimes

console = Console()


@click.group()
def main() -> None:
    """olmo-eval command line interface."""
    pass


@main.command()
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    required=True,
    help="Model name or preset. Can specify multiple times for multi-model runs.",
)
@click.option("--task", "-t", multiple=True, required=True, help="Task spec or suite")
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML config file")
@click.option("--output-dir", "-o", default=BEAKER_RESULT_DIR, help="Output directory")
@click.option("--num-shots", type=int, help="Override num_fewshot for all tasks")
@click.option("--limit", type=int, help="Override instance limit for all tasks")
@click.option("--backend", type=click.Choice(["hf", "vllm", "litellm"]), help="Override backend")
@click.option(
    "--storage-backend",
    "-s",
    "storage_backends",
    type=click.Choice(["s3", "postgres"]),
    multiple=True,
    help="Storage backend(s) for results. Can be specified multiple times.",
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
    "--async-stream",
    "use_async_stream",
    is_flag=True,
    help="Use streaming async runner with vLLM's AsyncLLMEngine for true continuous batching",
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
@click.option(
    "--parallelism",
    "-P",
    type=int,
    default=1,
    help="Number of model instances to run in parallel (passed from launch command)",
)
def run(
    models: tuple[str, ...],
    task: tuple[str, ...],
    config: str | None,
    output_dir: str,
    num_shots: int | None,
    limit: int | None,
    backend: str | None,
    storage_backends: tuple[str, ...],
    storage_config: str | None,
    dry_run: bool,
    use_async: bool,
    use_async_stream: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    parallelism: int,
) -> None:
    """Run evaluation on specified tasks.

    Supports multiple models: use -m multiple times for multi-model runs.
    With --async, runs all (model, task) pairs with per-model workers.
    With --async-stream, uses vLLM's AsyncLLMEngine for true continuous batching.
    Without --async or --async-stream, runs sequentially for each model.
    """
    import logging

    from olmo_eval.runners.sequential import EvalRunner, ValidationError

    # Configure logging for Beaker job visibility
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )

    # Warning for num-workers without async
    if num_workers is not None and not use_async and not use_async_stream:
        console.print(
            "[yellow]Warning:[/yellow] --num-workers has no effect without "
            "--async or --async-stream"
        )

    if gpus_per_worker != 1 and not use_async and not use_async_stream:
        console.print(
            "[yellow]Warning:[/yellow] --gpus-per-worker has no effect without "
            "--async or --async-stream"
        )

    # Warning for conflicting flags
    if use_async and use_async_stream:
        console.print(
            "[yellow]Warning:[/yellow] Both --async and --async-stream specified. "
            "Using --async-stream."
        )
        use_async = False

    # Warning for backend override with async-stream
    if use_async_stream and backend and backend != "vllm":
        console.print(
            f"[yellow]Warning:[/yellow] --async-stream only supports vLLM backend, "
            f"ignoring --backend={backend}"
        )

    # Set up storage backends if specified
    storages: list = []
    if storage_backends:
        from olmo_eval.storage import get_backend

        # Load storage config if provided
        storage_cfg = None
        if storage_config:
            from omegaconf import DictConfig, OmegaConf

            cfg = OmegaConf.load(storage_config)
            if isinstance(cfg, DictConfig):
                storage_cfg = cfg
            else:
                console.print("[red]Error:[/red] Storage config must be a YAML dict, not a list")
                raise SystemExit(1)

        for backend_name in storage_backends:
            # Get backend-specific config section
            storage_kwargs: dict = {}
            if storage_cfg:
                backend_cfg = storage_cfg.get(backend_name, {})
                storage_kwargs = OmegaConf.to_container(backend_cfg, resolve=True) or {}  # type: ignore

            try:
                storage = get_backend(backend_name, **storage_kwargs)
                storages.append(storage)
                console.print(f"[green]Initialized {backend_name} storage backend[/green]")
            except ImportError as e:
                console.print(f"[red]Storage backend error:[/red] {e}")
                raise SystemExit(1) from None
            except Exception as e:
                console.print(
                    f"[red]Failed to initialize {backend_name} storage backend:[/red] {e}"
                )
                raise SystemExit(1) from None

    # Choose runner based on --async or --async-stream flag
    if use_async_stream:
        from olmo_eval.runners.parallel import StreamingEvalRunner

        console.print(
            "[bold cyan]Using StreamingEvalRunner[/bold cyan] - "
            "true continuous batching with AsyncLLMEngine"
        )
        console.print(f"[bold]Models:[/bold] {len(models)}")

        runner = StreamingEvalRunner(
            model_names=list(models),
            task_specs=list(task),
            output_dir=output_dir,
            num_shots_override=num_shots,
            limit_override=limit,
            storages=storages,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
        )
    elif use_async:
        from olmo_eval.runners.parallel import AsyncEvalRunner

        console.print("[bold cyan]Using AsyncEvalRunner[/bold cyan] - parallel execution enabled")
        console.print(f"[bold]Models:[/bold] {len(models)}")

        runner = AsyncEvalRunner(
            model_names=list(models),
            task_specs=list(task),
            output_dir=output_dir,
            num_shots_override=num_shots,
            limit_override=limit,
            backend_override=backend,
            storages=storages,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
        )
    else:
        # Sequential runner - run each model in sequence
        if len(models) > 1:
            console.print(
                f"[bold cyan]Running {len(models)} models sequentially[/bold cyan] "
                "(use --async for parallel execution)"
            )

        # For sequential mode with multiple models, run each model separately
        for i, model in enumerate(models):
            if len(models) > 1:
                console.print(f"\n[bold]Model {i + 1}/{len(models)}:[/bold] {model}")

            runner = EvalRunner(
                model_name=model,
                task_specs=list(task),
                output_dir=output_dir,
                num_shots_override=num_shots,
                limit_override=limit,
                backend_override=backend,
                storages=storages,
            )

            try:
                runner.validate()
            except ValidationError as e:
                console.print(f"[red]Validation error:[/red]\n{e}")
                raise SystemExit(1) from None

            if dry_run:
                runner.print_config()
            else:
                runner.run()

        return  # Exit early since we handled everything in the loop

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
@click.option("--gpus", "-G", default=None, type=int, help="Number of GPUs per model instance")
@click.option(
    "--parallelism",
    "-P",
    default=None,
    type=int,
    help="Number of model instances to run in parallel",
)
@click.option(
    "--max-gpus-per-node",
    default=None,
    type=int,
    help="Maximum GPUs per node (default: 8). Tasks are split across experiments if exceeded.",
)
@click.option(
    "--priority",
    "-p",
    default=None,
    type=click.Choice(["low", "normal", "high", "urgent"]),
    help="Job priority",
)
@click.option("--preemptible/--no-preemptible", default=None, help="Allow preemption")
@click.option("--timeout", "-T", default=None, help="Job timeout (e.g., 24h, 30m)")
@click.option("--retries", "-r", type=int, help="Number of retries on failure")
@click.option("--workspace", "-w", help="Beaker workspace")
@click.option("--budget", "-B", help="Beaker budget")
@click.option(
    "--group",
    "-g",
    multiple=True,
    help="Add experiments to Beaker group(s) (can specify multiple, creates if needed)",
)
@click.option(
    "--backends",
    "-b",
    multiple=True,
    help="Backend optional groups to install at runtime (e.g., vllm, hf, litellm)",
)
@click.option("--async", "-a", "use_async", is_flag=True, help="Enable parallel task execution")
@click.option(
    "--async-stream",
    "use_async_stream",
    is_flag=True,
    help="Enable streaming async with vLLM's AsyncLLMEngine for true continuous batching",
)
@click.option("--num-workers", "-W", type=int, help="Number of workers for async mode")
@click.option("--gpus-per-worker", type=int, default=1, help="GPUs per worker for async mode")
@click.option(
    "--fa3",
    is_flag=True,
    help="Use Flash Attention 3 (for Hopper GPUs). FA2 is pre-installed by default.",
)
@click.option(
    "--no-flash-attn", is_flag=True, help="Disable Flash Attention (uninstalls FA2 at runtime)."
)
@click.option("--dry-run", "-d", is_flag=True, help="Print spec without launching")
@click.option(
    "--follow/--no-follow",
    default=True,
    help="Follow logs after launch (default). Use --no-follow to submit and exit immediately.",
)
def launch(
    config: str | None,
    name: str | None,
    model: tuple[str, ...],
    task: tuple[str, ...],
    cluster: str | None,
    gpus: int | None,
    parallelism: int | None,
    max_gpus_per_node: int | None,
    priority: str | None,
    preemptible: bool | None,
    timeout: str | None,
    retries: int | None,
    workspace: str | None,
    budget: str | None,
    group: tuple[str, ...],
    backends: tuple[str, ...],
    use_async: bool,
    use_async_stream: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    fa3: bool,
    no_flash_attn: bool,
    dry_run: bool,
    follow: bool,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.
    Use --group/-g to organize experiments into a Beaker group for result aggregation.
    Use --backends/-b to install inference backends at runtime (e.g., vllm, transformers).
    Use --fa3 to switch to Flash Attention 3 (for Hopper GPUs). FA2 is pre-installed.

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
    try:
        from olmo_eval.launch import (
            BeakerJobConfig,
            BeakerLauncher,
            LaunchConfig,
            ModelConfig,
            calculate_experiment_splits,
            parse_model_config,
            validate_priority_configuration,
        )
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    # Track which CLI args were explicitly set (vs using defaults)
    cli_cluster = cluster
    cli_gpus = gpus
    cli_parallelism = parallelism
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

        # Flash Attention 3: CLI flag overrides config
        if not fa3 and cfg.flash_attn == 3:
            fa3 = True

        # Get model configs from file (with per-model resource overrides)
        if not model:
            model_configs = cfg.get_model_configs()
        else:
            # CLI models override config file models
            model_configs = [parse_model_config(m) for m in model]

        # Set defaults from config (will be overridden by per-model or CLI)
        cluster = cluster if cluster is not None else cfg.cluster
        gpus = gpus if gpus is not None else cfg.gpus
        parallelism = parallelism if parallelism is not None else cfg.parallelism
        if max_gpus_per_node is None:
            max_gpus_per_node = cfg.max_gpus_per_node
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
    gpus = gpus if gpus is not None else 1
    parallelism = parallelism if parallelism is not None else 1
    if max_gpus_per_node is None:
        max_gpus_per_node = DEFAULT_MAX_GPUS_PER_NODE
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
    if not cluster:
        console.print("[red]Error:[/red] --cluster/-c is required (or set 'cluster' in config)")
        raise SystemExit(1)
    if not workspace:
        console.print("[red]Error:[/red] --workspace/-w is required (or set 'workspace' in config)")
        raise SystemExit(1)
    if not budget:
        console.print("[red]Error:[/red] --budget/-B is required (or set 'budget' in config)")
        raise SystemExit(1)

    # Validate and group tasks by priority
    # This will raise an error if --priority is used together with @priority suffixes
    try:
        tasks_by_priority = validate_priority_configuration(
            tasks=task,
            cli_priority=cli_priority,
            default_priority=priority,
        )
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
        console.print("\nUse 'olmo-eval tasks' to see available tasks.")
        console.print("Use 'olmo-eval suites' to see available suites.")
        raise SystemExit(1)

    launcher = BeakerLauncher(workspace=workspace)
    multiple_models = len(model_configs) > 1
    multiple_priorities = len(tasks_by_priority) > 1

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Build list of groups from CLI and config
    # Auto-generate one group name if none specified
    from datetime import datetime

    effective_groups: list[str] = list(group)  # CLI groups
    if cfg is not None and cfg.groups:
        # Add config groups (CLI groups take precedence, so they're first)
        for g in cfg.groups:
            if g not in effective_groups:
                effective_groups.append(g)

    # Auto-generate a group if none specified
    if not effective_groups:
        effective_groups = [f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"]

    console.print(f"[blue]Groups:[/blue] {', '.join(effective_groups)}")

    # Pre-create groups so they exist when experiments reference them
    if not dry_run:
        for grp in effective_groups:
            try:
                beaker_group = launcher.get_or_create_group(
                    name=grp,
                    workspace=workspace,
                )
                group_url = launcher.get_group_url(beaker_group)
                console.print(f"[blue]  {grp}:[/blue] {group_url}")
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Failed to create group '{grp}': {e}")

    # Track launched experiments
    launched_experiments: list[str] = []

    # Build experiment plan with parallelism and splitting
    experiment_plan: list[dict] = []
    split_models: list[str] = []  # Track models that require splitting

    for m_cfg in model_configs:
        m_name = m_cfg.name
        short_m = m_name.split("/")[-1].lower()
        if cfg is not None:
            m_resources = cfg.get_model_resources(m_cfg)
            m_gpus = cli_gpus if cli_gpus is not None else m_resources.get("gpus", 1)
            m_parallelism = (
                cli_parallelism if cli_parallelism is not None
                else m_resources.get("parallelism", 1)
            )
        else:
            m_gpus = cli_gpus if cli_gpus is not None else (m_cfg.gpus or gpus)
            m_parallelism = (
                cli_parallelism if cli_parallelism is not None
                else (m_cfg.parallelism or parallelism)
            )

        for t_priority, t_list in tasks_by_priority.items():
            base_name = name
            if multiple_models:
                base_name = f"{base_name}-{short_m}"
            if multiple_priorities:
                base_name = f"{base_name}-{t_priority}"

            # Calculate splits based on GPU constraints
            splits = calculate_experiment_splits(
                tasks=t_list,
                gpus_per_model=m_gpus,
                parallelism=m_parallelism,
                max_gpus_per_node=max_gpus_per_node,
            )

            if len(splits) > 1:
                split_models.append(m_name)

            total_splits = len(splits)
            for i, split in enumerate(splits):
                # Add zero-padded suffix for splits
                exp_name = f"{base_name}-{i + 1:03d}" if total_splits > 1 else base_name

                experiment_plan.append({
                    "name": exp_name,
                    "model_name": m_name,
                    "model_cfg": m_cfg,
                    "priority": t_priority,
                    "tasks": split["tasks"],
                    "gpus_per_model": m_gpus,
                    "num_gpus": split["num_gpus"],
                    "parallelism": split["parallelism"],
                    "split_index": i + 1 if total_splits > 1 else None,
                    "total_splits": total_splits if total_splits > 1 else None,
                })

    # Print experiment matrix summary
    console.print("\n[bold]Experiment Matrix:[/bold]")
    console.print(f"  Models: {len(model_configs)}")
    console.print(f"  Priority levels: {len(tasks_by_priority)}")
    total_experiments = len(experiment_plan)
    if split_models:
        unique_splits = list(set(split_models))
        split_msg = f"({len(unique_splits)} model(s) split due to GPU limits)"
        console.print(f"  Total experiments: {total_experiments} {split_msg}")
    else:
        console.print(f"  Total experiments: {total_experiments}")

    matrix_table = Table(title="Experiments to Launch", show_header=True)
    matrix_table.add_column("Experiment Name", style="cyan")
    matrix_table.add_column("Model", style="blue")
    matrix_table.add_column("Priority", style="yellow")
    matrix_table.add_column("Tasks", style="white")
    matrix_table.add_column("GPUs/Model", style="green", justify="right")
    matrix_table.add_column("Instances", style="magenta", justify="right")
    matrix_table.add_column("Total GPUs", style="green", justify="right")
    matrix_table.add_column("Split", style="dim", justify="center")

    for exp in experiment_plan:
        task_display = ", ".join(exp["tasks"])
        split_display = (
            f"{exp['split_index']}/{exp['total_splits']}"
            if exp["split_index"] is not None
            else "-"
        )
        matrix_table.add_row(
            exp["name"],
            exp["model_name"],
            exp["priority"],
            task_display,
            str(exp["gpus_per_model"]),
            str(exp["parallelism"]),
            str(exp["num_gpus"]),
            split_display,
        )

    console.print(matrix_table)
    console.print()

    # Launch experiments from the plan
    for exp in experiment_plan:
        model_cfg = exp["model_cfg"]
        model_name = exp["model_name"]
        exp_name = exp["name"]
        task_list = exp["tasks"]
        exp_num_gpus = exp["num_gpus"]
        exp_parallelism = exp["parallelism"]
        effective_priority = exp["priority"]

        # Get effective resources for this model (per-model overrides merged with defaults)
        if cfg is not None:
            model_resources = cfg.get_model_resources(model_cfg)
        else:
            # No config file - use ModelConfig values or defaults
            m_para = model_cfg.parallelism
            model_resources = {
                "gpus": model_cfg.gpus if model_cfg.gpus is not None else gpus,
                "parallelism": m_para if m_para is not None else parallelism,
                "cluster": model_cfg.cluster if model_cfg.cluster is not None else cluster,
                "preemptible": (
                    model_cfg.preemptible if model_cfg.preemptible is not None else preemptible
                ),
                "timeout": model_cfg.timeout if model_cfg.timeout is not None else timeout,
                "shared_memory": model_cfg.shared_memory,
                "backend": model_cfg.backend,
            }

        # CLI args always override per-model config
        # Cast values from model_resources dict to expected types
        effective_cluster: str = (
            cli_cluster if cli_cluster is not None else str(model_resources["cluster"])
        )
        effective_preemptible: bool = (
            cli_preemptible if cli_preemptible is not None else bool(model_resources["preemptible"])
        )
        effective_timeout: str = (
            cli_timeout if cli_timeout is not None else str(model_resources["timeout"])
        )
        res_shared_memory = model_resources.get("shared_memory")
        effective_shared_memory: str = str(res_shared_memory) if res_shared_memory else "10GiB"

        # Build command with this model and experiment's tasks
        command = ["olmo-eval", "run", "-m", model_name]
        for t in task_list:
            command.extend(["-t", t])

        # Add parallelism if > 1 (so the run command knows to run multiple instances)
        if exp_parallelism > 1:
            command.extend(["--parallelism", str(exp_parallelism)])

        # Add async flags if enabled (CLI flags override config)
        effective_use_async = use_async or model_resources.get("use_async", False)
        effective_use_async_stream = use_async_stream or model_resources.get(
            "use_async_stream", False
        )
        effective_num_workers = (
            num_workers if num_workers is not None else model_resources.get("num_workers")
        )
        effective_gpus_per_worker = (
            gpus_per_worker
            if gpus_per_worker != 1
            else model_resources.get("gpus_per_worker", 1)
        )

        # --async-stream takes precedence over --async
        if effective_use_async_stream:
            command.append("--async-stream")
            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])
        elif effective_use_async:
            command.append("--async")
            if effective_num_workers is not None:
                command.extend(["--num-workers", str(effective_num_workers)])
            if effective_gpus_per_worker and effective_gpus_per_worker != 1:
                command.extend(["--gpus-per-worker", str(effective_gpus_per_worker)])

        # Determine the backend this model will use at runtime
        # First check for explicit backend override in config, then get from model config
        from olmo_eval.core.configs import get_model_config as get_runtime_model_config
        from olmo_eval.core.constants.infrastructure import BACKEND_OPTIONAL_GROUPS

        config_backend = model_resources.get("backend")  # Explicit override from launch config
        if config_backend:
            runtime_backend: str = str(config_backend)
        else:
            # Get the backend from model config (preset or default)
            runtime_model_config = get_runtime_model_config(model_name)
            runtime_backend = runtime_model_config.backend

        # CLI backends override auto-detected backend optional group
        if backends:
            effective_backends = list(backends)
        else:
            # Get the optional group name for this backend
            backend_group = BACKEND_OPTIONAL_GROUPS.get(runtime_backend)
            effective_backends = [backend_group] if backend_group else []

        # Flash Attention: FA3 upgrade or disable entirely
        effective_flash_attn: int | None = 3 if fa3 else None

        job_config = BeakerJobConfig(
            name=exp_name,
            command=command,
            cluster=effective_cluster,
            num_gpus=exp_num_gpus,
            priority=effective_priority,
            preemptible=effective_preemptible,
            timeout=effective_timeout,
            shared_memory=effective_shared_memory,
            retries=retries,
            workspace=workspace,
            budget=budget,
            backends=effective_backends,
            flash_attn=effective_flash_attn,
            no_flash_attn=no_flash_attn,
            groups=effective_groups,
        )

        if dry_run:
            if len(experiment_plan) > 1:
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

        # Follow experiment(s) if requested
        if follow:
            if len(launched_experiments) == 1:
                # Single experiment: follow it
                import sys

                exit_code = launcher.follow_experiment(launched_experiments[0])
                sys.exit(exit_code)
            else:
                # Multiple experiments: don't follow, show URLs for watch command
                console.print(
                    "\n[bold]Multiple experiments launched. "
                    "Use 'olmo-eval watch -e <id>' to follow:[/bold]"
                )
                for exp_id in launched_experiments:
                    url = launcher.get_experiment_url(exp_id)
                    console.print(f"  - {url}")


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


@main.command(name="watch")
@click.option(
    "--experiment",
    "-e",
    required=True,
    help="Beaker experiment ID to watch",
)
@click.option(
    "--tail",
    "-t",
    is_flag=True,
    help="Only show recent logs (last 10 seconds). Useful for attaching to running experiments.",
)
def watch(experiment: str, tail: bool) -> None:
    """Watch an experiment's logs in real-time.

    Streams logs from a Beaker experiment until it completes. Shows startup
    events (pulling image, scheduling) followed by live log output.

    Use --tail/-t to show only recent logs when attaching to an already-running
    experiment.

    Examples:

        # Watch an experiment from the start
        olmo-eval watch -e 01abc123

        # Attach to a running experiment (show recent logs only)
        olmo-eval watch -e 01abc123 --tail
    """
    import sys

    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()

    try:
        exit_code = launcher.follow_experiment(experiment, tail=tail)
        sys.exit(exit_code)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None


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

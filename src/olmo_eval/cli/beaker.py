"""Beaker commands for olmo-eval CLI."""

from datetime import UTC

import click
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from olmo_eval.cli.utils import (
    EvalSummary,
    ModelSummary,
    RunnerConfig,
    TaskSummary,
    console,
    parse_model_spec,
)
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR, DEFAULT_MAX_GPUS_PER_NODE


@click.group()
def beaker() -> None:
    """Beaker job management commands.

    Commands for launching, monitoring, and managing evaluation jobs on Beaker.
    """
    pass


@beaker.command()
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
@click.option("--image", "-I", help="Beaker image (e.g., ai2-tylerm/olmo-eval-cu1261-trc280-amd64)")
@click.option(
    "--group",
    "-g",
    multiple=True,
    help="Add experiments to Beaker group(s) (can specify multiple, creates if needed)",
)
@click.option(
    "--extras",
    "-e",
    multiple=True,
    help="Optional dependency groups to install at runtime (e.g., vllm, postgres)",
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
@click.option("--dry-run", "-d", is_flag=True, help="Print spec without launching")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--follow/--no-follow",
    default=True,
    help="Follow logs after launch (default). Use --no-follow to submit and exit immediately.",
)
@click.option(
    "--aws-credentials/--no-aws-credentials",
    default=None,
    help="Inject AWS credentials for S3 model access. Auto-detected from s3:// model paths.",
)
@click.option(
    "--gcs-credentials/--no-gcs-credentials",
    default=None,
    help="Inject GCS credentials for gs:// model access. Auto-detected from gs:// model paths.",
)
@click.option(
    "--s3-bucket",
    help="S3 bucket for storing evaluation results (required for S3 uploads)",
)
@click.option(
    "--s3-prefix",
    help="S3 prefix/path within bucket for results (required for S3 uploads)",
)
@click.option(
    "--s3-endpoint-url",
    help="S3 endpoint URL (for S3-compatible storage like LocalStack)",
)
@click.option(
    "--s3-region",
    default="us-east-1",
    help="S3 region (default: us-east-1)",
)
@click.option(
    "--store/--no-store",
    default=False,
    help="Persist results to the configured database",
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
    image: str | None,
    group: tuple[str, ...],
    extras: tuple[str, ...],
    use_async: bool,
    use_async_stream: bool,
    num_workers: int | None,
    gpus_per_worker: int,
    dry_run: bool,
    yes: bool,
    follow: bool,
    aws_credentials: bool | None,
    gcs_credentials: bool | None,
    s3_bucket: str | None,
    s3_prefix: str | None,
    s3_endpoint_url: str | None,
    s3_region: str,
    store: bool,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.
    Use --group/-g to organize experiments into a Beaker group for result aggregation.
    Use --extras/-e to install optional dependencies at runtime (e.g., vllm, postgres).

    Examples:

        olmo-eval beaker launch -n "eval-llama3" -m llama3.1-8b -t mmlu

        olmo-eval beaker launch -n "eval-suite" -m llama3.1-8b -t mmlu -t gsm8k -t arc

        olmo-eval beaker launch -n "eval-70b" -m llama3.1-70b -t mmlu --cluster h100 --gpus 4

        # Multiple models (creates separate experiments per model)
        olmo-eval beaker launch -n "eval-compare" -m llama3.1-8b -m olmo-2-7b -t mmlu -t gsm8k

        # Per-task priorities (creates separate experiments per priority level)
        olmo-eval beaker launch -n "eval-mixed" -m llama3.1-8b -t "mmlu@high" -t "gsm8k@normal"

        # Install backends at runtime
        olmo-eval beaker launch -n "eval-vllm" -m llama3.1-8b -t mmlu -b vllm

        # From YAML config file
        olmo-eval beaker launch -f eval_config.yaml

        # Config file with CLI overrides
        olmo-eval beaker launch -f eval_config.yaml --gpus 4 --priority high

        # With grouping for result aggregation
        olmo-eval beaker launch -n "benchmark" --group "benchmark-2024" \\
            -m llama3.1-8b -t mmlu -t gsm8k
    """
    import json as json_module

    try:
        from olmo_eval.launch import (
            BeakerEnvSecret,
            BeakerJobConfig,
            BeakerLauncher,
            EvalConfig,
            ModelConfig,
            calculate_experiment_splits,
            get_model_short_name,
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
    cfg: EvalConfig | None = None
    model_configs: list[ModelConfig] = []

    if config:
        try:
            cfg = EvalConfig.from_yaml(config)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None
        except Exception as e:
            console.print(f"[red]Config error:[/red] {e}")
            raise SystemExit(1) from None

        # Use config values as defaults, CLI args override
        name = name or cfg.name
        task = task if task else tuple(cfg.tasks)
        extras = extras if extras else (tuple(cfg.extras) if cfg.extras else ())
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

    # Validate S3 options - both bucket and prefix required if either is set
    if s3_bucket and not s3_prefix:
        console.print("[red]Error:[/red] --s3-prefix is required when --s3-bucket is set")
        raise SystemExit(1)
    if s3_prefix and not s3_bucket:
        console.print("[red]Error:[/red] --s3-bucket is required when --s3-prefix is set")
        raise SystemExit(1)

    # --store requires S3 configuration for storing result artifacts
    if store and (not s3_bucket or not s3_prefix):
        console.print(
            "[red]Error:[/red] --s3-bucket and --s3-prefix are required when --store is enabled"
        )
        raise SystemExit(1)

    # Keep suites unexpanded - let the runner expand them so it knows suite names for aggregation
    from olmo_eval.core.configs import expand_tasks, validate_tasks

    original_task_specs = list(task)  # Preserve original suite/task names

    # Group by priority WITHOUT expanding first - this keeps suites as single units
    try:
        tasks_by_priority = validate_priority_configuration(
            tasks=original_task_specs,
            cli_priority=cli_priority,
            default_priority=priority,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

    # Get all specs (without @priority suffix, but with ::overrides)
    all_task_specs = [t for tasks in tasks_by_priority.values() for t in tasks]

    # Expand for validation only - ensure all tasks/suites exist
    expanded_for_validation = expand_tasks(all_task_specs)
    valid_tasks, invalid_tasks = validate_tasks(expanded_for_validation)

    # Track expanded task counts per priority for display purposes
    expanded_counts_by_priority: dict[str, int] = {}
    for priority_level, specs in tasks_by_priority.items():
        expanded_counts_by_priority[priority_level] = len(expand_tasks(specs))

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

    # Auto-detect when AWS credentials are needed
    from olmo_eval.launch.aws import get_local_aws_credentials, is_s3_path

    s3_models = [m.name_or_path for m in model_configs if is_s3_path(m.name_or_path)]
    inject_aws_credentials = aws_credentials
    if inject_aws_credentials is None:
        inject_aws_credentials = bool(s3_models) or store

    if inject_aws_credentials:
        local_creds = get_local_aws_credentials()
        beaker_user = launcher.beaker.user_name

        s3_table = Table(show_header=False, box=None, expand=True)
        s3_table.add_column("Key", style="blue")
        s3_table.add_column("Value")

        if local_creds:
            cred_type = "temporary" if local_creds.session_token else "long-term"
            s3_table.add_row("Credentials", f"[green]found[/green] ({cred_type})")
            s3_table.add_row("Beaker user", beaker_user)
            s3_table.add_row(
                "Beaker secrets",
                f"{beaker_user}_AWS_ACCESS_KEY_ID, {beaker_user}_AWS_SECRET_ACCESS_KEY",
            )
        else:
            s3_table.add_row(
                "Credentials",
                "[yellow]not found[/yellow] - job may fail if S3 access is required",
            )

        console.print()
        console.print(
            Panel(s3_table, title="[bold]S3 Access Configuration[/bold]", border_style="yellow")
        )
        console.print()

    # Auto-detect GCS model paths for GCS credential injection
    from olmo_eval.launch.gcs import get_local_gcs_credentials, is_gcs_path

    gcs_models = [m.name_or_path for m in model_configs if is_gcs_path(m.name_or_path)]
    inject_gcs_credentials = gcs_credentials
    if inject_gcs_credentials is None:
        inject_gcs_credentials = bool(gcs_models)

    if inject_gcs_credentials:
        local_gcs_creds = get_local_gcs_credentials()
        beaker_user = launcher.beaker.user_name

        gcs_table = Table(show_header=False, box=None, expand=True)
        gcs_table.add_column("Key", style="blue")
        gcs_table.add_column("Value")

        if local_gcs_creds:
            gcs_table.add_row("Credentials", "[green]found[/green] (service account)")
            if local_gcs_creds.client_email:
                gcs_table.add_row("Service account", local_gcs_creds.client_email)
            if local_gcs_creds.project_id:
                gcs_table.add_row("Project", local_gcs_creds.project_id)
            gcs_table.add_row("Beaker user", beaker_user)
            gcs_table.add_row("Beaker secret", f"{beaker_user}_GOOGLE_CREDENTIALS")
        else:
            gcs_table.add_row(
                "Credentials",
                "[yellow]not found[/yellow] - job may fail if GCS access is required",
            )

        console.print()
        console.print(
            Panel(gcs_table, title="[bold]GCS Access Configuration[/bold]", border_style="magenta")
        )
        console.print()

    # Get workspace object for beaker API calls that require it
    workspace_obj = launcher.beaker.workspace.get(workspace) if workspace else None

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Build list of groups from CLI and config
    from datetime import datetime

    effective_groups: list[str] = list(group)
    if cfg is not None and cfg.groups:
        for g in cfg.groups:
            if g not in effective_groups:
                effective_groups.append(g)

    # Auto-generate a group if none specified
    if not effective_groups:
        effective_groups = [f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"]

    # Determine the effective image
    from olmo_eval.core.constants.infrastructure import BEAKER_DEFAULT_IMAGE

    if image:
        effective_image = image
    elif cfg and cfg.beaker_image:
        effective_image = cfg.beaker_image
    else:
        effective_image = BEAKER_DEFAULT_IMAGE

    # Check which groups exist and which need to be created
    from beaker.exceptions import BeakerGroupNotFound

    existing_groups: list[str] = []
    missing_groups: list[str] = []

    for grp in effective_groups:
        qualified_name = f"{launcher.beaker.user_name}/{grp}" if "/" not in grp else grp
        try:
            launcher.beaker.group.get(qualified_name)
            existing_groups.append(grp)
        except BeakerGroupNotFound:
            missing_groups.append(grp)

    if dry_run:
        if missing_groups:
            console.print(
                f"[yellow]Note:[/yellow] The following groups would be created: "
                f"{', '.join(missing_groups)}"
            )
    else:
        if missing_groups:
            console.print(
                f"\n[yellow]The following groups do not exist:[/yellow] {', '.join(missing_groups)}"
            )
            if not click.confirm("Would you like to create these groups?", default=True):
                console.print("[red]Aborted.[/red] Cannot launch without required groups.")
                raise SystemExit(1)

            for grp in missing_groups:
                try:
                    beaker_group = launcher.beaker.group.create(
                        name=grp,
                        workspace=workspace_obj,
                    )
                    group_url = launcher.get_group_url(beaker_group)
                    console.print(f"[green]  Created {grp}:[/green] {group_url}")
                except Exception as e:
                    console.print(f"[red]Error:[/red] Failed to create group '{grp}': {e}")
                    raise SystemExit(1) from None

    # Track launched experiments
    launched_experiments: list[str] = []

    # Build experiment plan with parallelism and splitting
    experiment_plan: list[dict] = []
    split_models: list[str] = []

    for m_cfg in model_configs:
        m_name = m_cfg.name_or_path
        short_m = get_model_short_name(m_cfg)
        if cfg is not None:
            m_resources = cfg.get_model_resources(m_cfg)
            m_gpus = cli_gpus if cli_gpus is not None else m_resources.get("gpus", 1)
            m_parallelism = (
                cli_parallelism
                if cli_parallelism is not None
                else m_resources.get("parallelism", 1)
            )
        else:
            m_gpus = cli_gpus if cli_gpus is not None else (m_cfg.gpus or gpus)
            m_parallelism = (
                cli_parallelism
                if cli_parallelism is not None
                else (m_cfg.parallelism or parallelism)
            )

        for t_priority, t_list in tasks_by_priority.items():
            base_name = name
            if multiple_models:
                base_name = f"{base_name}-{short_m}"
            if multiple_priorities:
                base_name = f"{base_name}-{t_priority}"

            splits = calculate_experiment_splits(
                tasks=t_list,
                gpus_per_model=m_gpus,
                parallelism=m_parallelism,
                max_gpus_per_node=max_gpus_per_node,
            )

            if len(splits) > 1:
                split_models.append(m_name)

            total_splits = len(splits)
            total_expanded = expanded_counts_by_priority[t_priority]
            for i, split in enumerate(splits):
                exp_name = f"{base_name}-{i + 1:03d}" if total_splits > 1 else base_name

                experiment_plan.append(
                    {
                        "name": exp_name,
                        "model_name": m_name,
                        "model_cfg": m_cfg,
                        "priority": t_priority,
                        "tasks": split["tasks"],
                        "original_task_specs": original_task_specs,
                        "total_expanded_tasks": total_expanded,
                        "gpus_per_model": m_gpus,
                        "num_gpus": split["num_gpus"],
                        "parallelism": split["parallelism"],
                        "split_index": i + 1 if total_splits > 1 else None,
                        "total_splits": total_splits if total_splits > 1 else None,
                    }
                )

    # Calculate total expanded tasks
    total_experiments = len(experiment_plan)
    total_expanded_tasks = len(valid_tasks) * len(model_configs)

    # Fetch actual task configurations for display
    from olmo_eval.evals.tasks import get_task as get_task_instance
    from olmo_eval.evals.tasks.core.registry import parse_task_spec

    task_summaries: list[TaskSummary] = []
    for task_spec in valid_tasks:
        task_name, variants, inline_overrides = parse_task_spec(task_spec)

        task_instance = get_task_instance(task_spec)
        task_cfg = task_instance.config
        task_summaries.append(
            TaskSummary(
                name=task_cfg.name,
                spec=task_spec if task_spec != task_cfg.name else None,
                variants=variants if variants else None,
                formatter=task_cfg.formatter,
                scorers=task_cfg.scorers,
                metrics=task_cfg.metrics,
                num_fewshot=task_cfg.num_fewshot,
                split=task_cfg.split.value
                if hasattr(task_cfg.split, "value")
                else str(task_cfg.split),
                primary_metric=str(task_cfg.primary_metric) if task_cfg.primary_metric else None,
                sampling_params=task_cfg.sampling_params,
                overrides=inline_overrides if inline_overrides else None,
            )
        )

    # Build model summaries with resolved backends
    from olmo_eval.core.configs import get_model_config as get_runtime_model_config

    model_summaries: list[ModelSummary] = []
    for m in model_configs:
        model_base_name, model_inline_overrides = parse_model_spec(m.name_or_path)

        if m.backend:
            effective_backend = m.backend
        else:
            runtime_model_config = get_runtime_model_config(model_base_name)
            effective_backend = runtime_model_config.backend

        model_summaries.append(
            ModelSummary(
                name=model_base_name,
                gpus=m.gpus or gpus,
                parallelism=m.parallelism or parallelism,
                alias=m.alias,
                backend=effective_backend,
                overrides=model_inline_overrides if model_inline_overrides else None,
            )
        )

    # Build runner config
    from olmo_eval.runners import AsyncEvalRunner, StreamingEvalRunner, SyncEvalRunner

    effective_attention_backend = None
    if cfg is not None and model_configs:
        first_model = model_configs[0]
        model_resources = cfg.get_model_resources(first_model)
        effective_attention_backend = model_resources.get("attention_backend")

    if use_async_stream:
        effective_num_workers = num_workers
        if effective_num_workers is None and cfg is not None and model_configs:
            first_model = model_configs[0]
            model_resources = cfg.get_model_resources(first_model)
            effective_num_workers = model_resources.get("num_workers")

        runner_config = RunnerConfig(
            runner=StreamingEvalRunner,
            output_dir=BEAKER_RESULT_DIR,
            attention_backend=effective_attention_backend,
            num_workers=effective_num_workers if effective_num_workers is not None else "auto",
            gpus_per_worker=gpus_per_worker,
        )
    elif use_async:
        effective_num_workers = num_workers
        if effective_num_workers is None and cfg is not None and model_configs:
            first_model = model_configs[0]
            model_resources = cfg.get_model_resources(first_model)
            effective_num_workers = model_resources.get("num_workers")

        runner_config = RunnerConfig(
            runner=AsyncEvalRunner,
            output_dir=BEAKER_RESULT_DIR,
            attention_backend=effective_attention_backend,
            num_workers=effective_num_workers if effective_num_workers is not None else "auto",
            gpus_per_worker=gpus_per_worker,
        )
    else:
        runner_config = RunnerConfig(
            runner=SyncEvalRunner,
            output_dir=BEAKER_RESULT_DIR,
            attention_backend=effective_attention_backend,
        )

    # Build the eval summary for display
    eval_summary = EvalSummary(
        models=model_summaries,
        tasks=task_summaries,
        runner=runner_config,
    )

    # Print consolidated launch configuration using rich repr
    console.print()
    console.print(
        Panel(
            Pretty(eval_summary, expand_all=True, no_wrap=True, overflow="fold"),
            title="[bold]Launch Configuration[/bold]",
            border_style="blue",
        )
    )

    # Print experiment summary
    console.print(
        f"\n[bold]Experiments:[/bold] {total_experiments} experiment(s), "
        f"{total_expanded_tasks} task(s)"
    )
    if split_models:
        console.print(
            "[dim]  Tasks distributed across multiple experiments due to GPU constraints[/dim]"
        )

    # Simplified experiment table - only show if multiple experiments
    if total_experiments > 1:
        matrix_table = Table(show_header=True, title="Experiment Plan")
        matrix_table.add_column("Name", style="cyan")
        matrix_table.add_column("Model", style="blue")
        matrix_table.add_column("Priority", style="yellow")
        matrix_table.add_column("Tasks", justify="right")
        matrix_table.add_column("GPUs", style="green", justify="right")

        for exp in experiment_plan:
            task_count = len(exp["tasks"])
            total_tasks = exp["total_expanded_tasks"]
            task_display = (
                f"{task_count}/{total_tasks}" if exp["split_index"] is not None else str(task_count)
            )
            matrix_table.add_row(
                exp["name"],
                exp["model_name"],
                exp["priority"],
                task_display,
                str(exp["num_gpus"]),
            )

        console.print(matrix_table)

    console.print()

    # Ensure common secrets exist in Beaker
    from olmo_eval.launch.secrets import (
        ensure_common_secrets,
        get_local_hf_token,
        get_local_wandb_api_key,
    )

    beaker_username = launcher.beaker.user_name
    if dry_run:
        common_secrets: list[tuple[str, str]] = []
        if get_local_hf_token():
            common_secrets.append(("HF_TOKEN", f"{beaker_username}_HF_TOKEN"))
        if get_local_wandb_api_key():
            common_secrets.append(("WANDB_API_KEY", f"{beaker_username}_WANDB_API_KEY"))
    else:
        common_secrets = ensure_common_secrets(workspace=workspace)

    # Build store secrets list if --store is enabled
    # User must have manually created these secrets in Beaker (shared, not per-user)
    store_secrets: list[tuple[str, str]] = []
    if store:
        for env_var in ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]:
            store_secrets.append((env_var, f"olmo_eval_{env_var}"))

    # Build all BeakerJobConfig objects first
    job_configs: list[BeakerJobConfig] = []

    for exp in experiment_plan:
        model_cfg = exp["model_cfg"]
        model_name = exp["model_name"]
        exp_name = exp["name"]
        task_list = exp["tasks"]
        exp_num_gpus = exp["num_gpus"]
        exp_parallelism = exp["parallelism"]
        effective_priority = exp["priority"]

        # Get effective resources for this model
        if cfg is not None:
            model_resources = cfg.get_model_resources(model_cfg)
        else:
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

        # Build model spec with inline overrides for per-model vLLM loading options
        model_spec = model_name
        model_inline_overrides: list[str] = []

        effective_load_format = model_resources.get("load_format")
        if effective_load_format:
            model_inline_overrides.append(f"load_format={effective_load_format}")

        effective_extra_loader_config = model_resources.get("extra_loader_config")
        if effective_extra_loader_config:
            json_config = json_module.dumps(effective_extra_loader_config, separators=(",", ":"))
            model_inline_overrides.append(f"extra_loader_config={json_config}")

        if model_inline_overrides:
            model_spec = f"{model_name}::{','.join(model_inline_overrides)}"

        # Build command with this model and experiment's tasks
        command = ["olmo-eval", "run", "-m", model_spec]
        for t in task_list:
            command.extend(["-t", t])

        # Add alias if defined in model config
        if model_cfg.alias:
            command.extend(["--alias", model_cfg.alias])

        # Add parallelism if > 1
        if exp_parallelism > 1:
            command.extend(["--parallelism", str(exp_parallelism)])

        # Add async flags if enabled
        effective_use_async = use_async or model_resources.get("use_async", False)
        effective_use_async_stream = use_async_stream or model_resources.get(
            "use_async_stream", False
        )
        effective_num_workers = (
            num_workers if num_workers is not None else model_resources.get("num_workers")
        )
        effective_gpus_per_worker = (
            gpus_per_worker if gpus_per_worker != 1 else model_resources.get("gpus_per_worker", 1)
        )

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

        # Add S3 options if configured (group is inferred from beaker group)
        if s3_bucket and s3_prefix:
            command.extend(["--s3-bucket", s3_bucket])
            command.extend(["--s3-prefix", s3_prefix])
            # Use the first beaker group as the S3 group
            if effective_groups:
                command.extend(["--s3-group", effective_groups[0]])
            if s3_endpoint_url:
                command.extend(["--s3-endpoint-url", s3_endpoint_url])
            if s3_region != "us-east-1":
                command.extend(["--s3-region", s3_region])

        # Add experiment name for database storage
        command.extend(["--experiment-name", exp_name])

        # Enable PostgreSQL storage backend for persisting evaluation results
        # (model metrics, task scores, S3 locations) after each task completes.
        # Credentials are injected via Beaker secrets (PGHOST, PGPORT, etc.)
        if store:
            command.append("--store")

        # Determine the backend this model will use at runtime
        from olmo_eval.core.configs import get_model_config as get_runtime_model_config
        from olmo_eval.core.constants.infrastructure import BACKEND_OPTIONAL_GROUPS

        config_backend = model_resources.get("backend")
        if config_backend:
            runtime_backend: str = str(config_backend)
        else:
            runtime_model_config = get_runtime_model_config(model_name)
            runtime_backend = runtime_model_config.backend

        # CLI extras override auto-detected backend
        if extras:
            model_backends = list(extras)
        else:
            backend_group = BACKEND_OPTIONAL_GROUPS.get(runtime_backend)
            model_backends = [backend_group] if backend_group else []

        # Combine model backends and storage dependencies for installation
        install_extras = list(model_backends)
        if store:
            install_extras.append("postgres")

        # Convert secrets to BeakerEnvSecret objects
        env_secrets = [
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in common_secrets
        ]
        env_secrets.extend(
            BeakerEnvSecret(env_var, secret_name) for env_var, secret_name in store_secrets
        )

        # Build env vars: include defaults plus Beaker author for attribution
        job_env_vars = {
            "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
            "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
            "BEAKER_AUTHOR": beaker_username,
        }

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
            extras=install_extras,
            groups=effective_groups,
            beaker_image=effective_image,
            inject_aws_credentials=inject_aws_credentials,
            inject_gcs_credentials=inject_gcs_credentials,
            env_vars=job_env_vars,
            env_secrets=env_secrets,
        )
        job_configs.append(job_config)

    # Display storage configuration if enabled
    if store or (s3_bucket and s3_prefix):
        storage_lines = []
        if store:
            storage_lines.append("[bold]PostgreSQL:[/bold]")
            storage_lines.append(
                "  Credentials from Beaker secrets: olmo_eval_PGHOST, olmo_eval_PGPORT,"
            )
            storage_lines.append("    olmo_eval_PGDATABASE, olmo_eval_PGUSER, olmo_eval_PGPASSWORD")
        if s3_bucket and s3_prefix:
            storage_lines.append("[bold]S3:[/bold]")
            storage_lines.append(f"  Bucket: {s3_bucket}")
            storage_lines.append(f"  Prefix: {s3_prefix}")
            storage_lines.append(f"  Region: {s3_region}")
            if s3_endpoint_url:
                storage_lines.append(f"  Endpoint: {s3_endpoint_url}")
            if effective_groups:
                storage_lines.append(f"  Group: {effective_groups[0]}")
        console.print(
            Panel(
                "\n".join(storage_lines),
                title="[bold]Storage Configuration[/bold]",
                border_style="green",
            )
        )

    # Display all BeakerJobConfig objects
    for job_config in job_configs:
        console.print(
            Panel(
                Pretty(job_config, expand_all=True, no_wrap=True, overflow="fold"),
                title=f"[bold]BeakerJobConfig: {job_config.name}[/bold]",
                border_style="cyan",
            )
        )

    # Confirm before launching
    if not dry_run and not yes and not click.confirm("Proceed with launch?", default=True):
        console.print("[yellow]Launch cancelled[/yellow]")
        raise SystemExit(0)

    # Launch experiments
    for job_config in job_configs:
        if not dry_run:
            experiment = launcher.launch(job_config)
            if experiment:
                console.print(f"[green]Launched:[/green] {launcher.experiment_url(experiment)}")
                launched_experiments.append(experiment.id)

    # Summary and follow logic for launched experiments
    if launched_experiments and not dry_run:
        if len(launched_experiments) > 1:
            console.print(f"\n[bold]Launched {len(launched_experiments)} experiment(s)[/bold]")

        if follow:
            if len(launched_experiments) == 1:
                import sys

                exit_code = launcher.follow_experiment(launched_experiments[0])
                sys.exit(exit_code)
            else:
                console.print(
                    "\n[bold]Multiple experiments launched. "
                    "Use 'olmo-eval beaker watch -e <id>' to follow:[/bold]"
                )
                for exp_id in launched_experiments:
                    url = launcher.get_experiment_url(exp_id)
                    console.print(f"  - {url}")


@beaker.command(hidden=True)
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
@click.pass_context
def results(
    ctx: click.Context,
    group: str,
    output_format: str,
    wait: bool,
    poll_interval: int,
) -> None:
    """[DEPRECATED] Use 'olmo-eval beaker group info' instead."""
    console.print(
        "[yellow]Warning:[/yellow] 'olmo-eval beaker results' is deprecated.\n"
        f"Use: olmo-eval beaker group info {group}"
        + (" --wait" if wait else "")
        + (f" --format {output_format}" if output_format != "table" else "")
        + "\n"
    )
    ctx.invoke(
        group_info,
        group_name=group,
        output_format=output_format,
        verbose=False,
        wait=wait,
        poll_interval=poll_interval,
    )


@beaker.command(name="watch")
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
    """Watch an experiment's logs in real-time."""
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


@beaker.group()
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
        from beaker.exceptions import BeakerGroupNotFound

        beaker_group = launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

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
            f"  [green]✓ {status.get('succeeded', 0)} succeeded[/green]\n"
            f"  [yellow]● {status.get('running', 0)} running[/yellow]\n"
            f"  [dim]○ {status.get('pending', 0)} pending[/dim]\n"
            f"  [red]✗ {status.get('failed', 0)} failed[/red]\n"
            f"  [red]⊘ {status.get('canceled', 0)} canceled[/red]"
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
        from beaker.exceptions import BeakerGroupNotFound

        beaker_group = launcher.beaker.group.get(group_name)
    except BeakerGroupNotFound:
        console.print(f"[red]Error:[/red] Group '{group_name}' not found")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1) from None

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
        f"  [green]✓ {result.get('canceled', 0)} canceled[/green]\n"
        f"  [dim]○ {result.get('skipped', 0)} skipped (already completed)[/dim]"
    )
    if result.get("failed", 0) > 0:
        console.print(f"  [red]✗ {result.get('failed', 0)} failed to cancel[/red]")


@group.command(name="list")
@click.option("--workspace", "-w", required=True, help="Beaker workspace to list groups from")
@click.option("--limit", "-n", type=int, default=20, help="Number of groups to show")
@click.option("--search", "-s", help="Search by name or description")
@click.option("--mine/--all", default=True, help="Show only my groups (default) or all groups")
def group_list(workspace: str, limit: int, search: str | None, mine: bool) -> None:
    """List Beaker groups."""
    try:
        from olmo_eval.launch import BeakerLauncher
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    launcher = BeakerLauncher()
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

            table.add_row(grp.name, workspace_name, str(exp_count), status_str, created_str)
        except Exception:
            table.add_row(grp.name, "-", "?", "[dim]error[/dim]", "-")

    console.print(table)

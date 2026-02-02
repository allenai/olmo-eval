"""Launch command for Beaker jobs.

This module provides the main 'beaker launch' command for submitting evaluation jobs.
Configuration loading, validation, and job assembly are delegated to:
- config_loader.py: LaunchConfigLoader for loading/merging config
- task_validator.py: TaskValidator for task validation and priority grouping
- credentials.py: CredentialManager for AWS/GCS credential handling
- model_grouper.py: ModelGrouper for grouping models by runtime signature
- experiment_builder.py: ExperimentPlanBuilder for building experiment plans
- job_assembler.py: JobConfigAssembler for assembling BeakerJobConfig
"""

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from olmo_eval.cli.beaker.experiment_plan import ExperimentPlan
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from olmo_eval.cli.utils import (
    ExperimentSummary,
    OrderedMultiOption,
    RunnerConfig,
    console,
    extract_priority_from_overrides,
    process_ordered_args,
    reconstruct_ordered_args,
)
from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR, BEAKER_UV_CACHE_DIR
from olmo_eval.core.types import RunnerType


@click.command()
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
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Model name or preset (can specify multiple). Use -o after to add overrides.",
)
@click.option(
    "--task",
    "-t",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Task name with optional @priority suffix. Use -o after to add overrides.",
)
@click.option(
    "--override",
    "-o",
    multiple=True,
    cls=OrderedMultiOption,
    save_to="_ordered",
    help="Override for preceding -m or -t (e.g., -o provider.name=vllm -o limit=100)",
)
@click.option("--cluster", "-c", default=None, help="Cluster alias (h100, a100, aus) or full name")
@click.option(
    "--max-gpus-per-node",
    default=None,
    type=int,
    help="Maximum GPUs per node (default: 8). Models are split across experiments if exceeded.",
)
@click.option(
    "--pack/--no-pack",
    default=None,
    help="Pack multiple models into single experiments when they fit. "
    "Default is --no-pack: each model runs in its own experiment for easier resource acquisition.",
)
@click.option(
    "--priority",
    "-p",
    type=click.Choice(["low", "normal", "high", "urgent"]),
    default=None,
    help="Job priority level (low, normal, high, urgent). Can also use @priority suffix on tasks.",
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
    "--runner-type",
    "-R",
    type=click.Choice([e.value for e in RunnerType], case_sensitive=False),
    default=None,
    help="Runner type: sync (default), async, async-stream, or agent",
)
@click.option("--num-workers", "-W", type=int, help="Number of workers for async modes")
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
@click.option(
    "--debug-requests",
    is_flag=True,
    help="Log HTTP requests/responses to inference providers",
)
@click.option(
    "--debug-provider",
    is_flag=True,
    help="Enable verbose provider logging",
)
@click.option(
    "--save-predictions/--no-save-predictions",
    "save_predictions",
    default=True,
    help="Save per-instance predictions to JSONL (default: enabled)",
)
@click.option(
    "--save-requests/--no-save-requests",
    "save_requests",
    default=True,
    help="Save per-instance requests to JSONL (default: enabled)",
)
@click.option(
    "--inspect-instance",
    is_flag=True,
    help="Print the first instance of each task before running evaluation",
)
@click.option(
    "--inspect-formatted",
    is_flag=True,
    help="Show formatted prompt (after template applied) before evaluation",
)
@click.option(
    "--inspect-tokens",
    is_flag=True,
    help="Show token array before evaluation",
)
@click.option(
    "--inspect-response",
    is_flag=True,
    help="Print the first response of each task after model generation",
)
@click.option(
    "--inspect-request",
    is_flag=True,
    help="Print the first request of each task before model generation",
)
@click.option(
    "--uv-cache-dir",
    default=BEAKER_UV_CACHE_DIR,
    show_default=True,
    help="UV cache directory for package downloads (on Weka shared storage)",
)
def launch(
    config: str | None,
    name: str | None,
    model: tuple[str, ...],
    task: tuple[str, ...],
    override: tuple[str, ...],
    cluster: str | None,
    max_gpus_per_node: int | None,
    pack: bool | None,
    priority: str | None,
    preemptible: bool | None,
    timeout: str | None,
    retries: int | None,
    workspace: str | None,
    budget: str | None,
    image: str | None,
    group: tuple[str, ...],
    runner_type: str | None,
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
    debug_requests: bool,
    debug_provider: bool,
    save_predictions: bool,
    save_requests: bool,
    inspect_instance: bool,
    inspect_formatted: bool,
    inspect_tokens: bool,
    inspect_response: bool,
    inspect_request: bool,
    uv_cache_dir: str,
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-py to be installed: pip install 'olmo-eval-internal[beaker]'

    Multiple models and/or tasks with different priorities will create separate experiments.
    Models with compatible runtime configurations (GPUs, provider, etc.) are grouped together.
    Use --config/-f to load settings from a YAML file; CLI arguments override config values.
    Use --group/-g to organize experiments into a Beaker group for result aggregation.

    Use -o/--override after -m or -t to apply overrides to that model or task:

        olmo-eval beaker launch -n eval \\
            -m llama3.1-8b -o provider.name=vllm -o provider.package=vllm==0.14.0 \\
            -t mmlu -o limit=100
    """
    from datetime import datetime

    try:
        from olmo_eval.launch import BeakerLauncher, EvalConfig
    except ImportError:
        console.print(
            "[red]beaker-py is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    # Process ordered args to associate overrides with models/tasks
    import sys

    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader
    from olmo_eval.cli.beaker.credentials import CredentialManager
    from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
    from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
    from olmo_eval.cli.beaker.model_grouper import ModelGrouper
    from olmo_eval.cli.beaker.task_validator import TaskValidator
    from olmo_eval.core.constants.infrastructure import BEAKER_DEFAULT_IMAGE

    ordered_args = reconstruct_ordered_args(sys.argv[1:])
    model_overrides, raw_task_overrides = process_ordered_args(ordered_args)

    # Extract priority from task overrides (e.g., -o priority=urgent after -t)
    # This is done once here and the filtered overrides are used everywhere
    override_priority, task_overrides = extract_priority_from_overrides(raw_task_overrides)

    # Build CLI args dict
    cli_args = {
        "name": name,
        "model": model,
        "task": task,
        "model_overrides": model_overrides,
        "task_overrides": task_overrides,  # Already filtered (priority extracted)
        "cluster": cluster,
        "max_gpus_per_node": max_gpus_per_node,
        "pack_models": pack,
        "priority": priority,
        "preemptible": preemptible,
        "timeout": timeout,
        "retries": retries,
        "workspace": workspace,
        "budget": budget,
        "image": image,
        "group": group,
        "runner_type": runner_type,
        "num_workers": num_workers,
        "gpus_per_worker": gpus_per_worker,
        "s3_bucket": s3_bucket,
        "s3_prefix": s3_prefix,
        "s3_endpoint_url": s3_endpoint_url,
        "s3_region": s3_region,
        "store": store,
        "debug_requests": debug_requests,
        "debug_provider": debug_provider,
        "save_predictions": save_predictions,
        "save_requests": save_requests,
        "inspect_instance": inspect_instance,
        "inspect_formatted": inspect_formatted,
        "inspect_tokens": inspect_tokens,
        "inspect_response": inspect_response,
        "inspect_request": inspect_request,
        "uv_cache_dir": uv_cache_dir,
    }

    # Load configuration
    config_loader = LaunchConfigLoader(config, cli_args)
    launch_config = config_loader.load()

    # Load EvalConfig for resource lookups
    eval_config: EvalConfig | None = None
    if config:
        import contextlib

        with contextlib.suppress(Exception):
            eval_config = EvalConfig.from_yaml(config)

    # Validate tasks and group by priority
    # Use override_priority from -o priority=X if specified, else use config priority
    effective_priority = override_priority or launch_config.priority
    task_validator = TaskValidator(
        launch_config.task_specs,
        cli_priority=None,
        default_priority=effective_priority,
    )
    tasks_by_priority, valid_tasks, agent_task_specs = task_validator.validate_and_group()

    # Create launcher
    launcher = BeakerLauncher(workspace=launch_config.workspace)

    # Set up credentials
    cred_manager = CredentialManager(
        launch_config.model_configs,
        launch_config.store,
        aws_credentials,
        gcs_credentials,
    )
    inject_aws, inject_gcs = cred_manager.detect_and_setup(launcher)

    # Update config with credential settings
    launch_config.inject_aws_credentials = inject_aws
    launch_config.inject_gcs_credentials = inject_gcs

    if dry_run:
        console.print("[yellow]Dry run mode - not submitting[/yellow]")

    # Auto-generate group if needed
    effective_groups = list(launch_config.groups)
    if not effective_groups:
        effective_groups = [f"{launch_config.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"]

    # Display storage info
    cred_manager.display_storage_info(
        launcher,
        launch_config.s3_bucket,
        launch_config.s3_prefix,
        launch_config.s3_region,
        launch_config.s3_endpoint_url,
        effective_groups,
        inject_aws,
    )

    # Determine effective image
    if image:
        effective_image = image
    elif eval_config and eval_config.beaker_image:
        effective_image = eval_config.beaker_image
    else:
        effective_image = BEAKER_DEFAULT_IMAGE

    # Handle group creation
    _handle_group_creation(launcher, effective_groups, dry_run)

    # Group models and build experiment plan
    model_grouper = ModelGrouper(launch_config, eval_config)
    experiment_builder = ExperimentPlanBuilder(
        launch_config, model_grouper, tasks_by_priority, agent_task_specs, override_priority
    )
    experiment_plan, split_models = experiment_builder.build()

    # Get task configs with overrides applied
    # (task_overrides is already filtered - priority extracted above)
    task_configs_by_spec = _get_task_configs(valid_tasks, launch_config.task_overrides)

    # Collect required secrets
    all_required_secrets: set[str] = set()
    for task_cfg in task_configs_by_spec.values():
        if hasattr(task_cfg, "required_secrets") and task_cfg.required_secrets:
            all_required_secrets.update(task_cfg.required_secrets)

    # Ensure secrets
    common_secrets, store_secrets, task_secrets = _ensure_secrets(
        launcher, dry_run, launch_config, all_required_secrets
    )

    # Print summary header
    total_experiments = len(experiment_plan)
    total_expanded_tasks = len(valid_tasks) * len(launch_config.model_configs)
    console.print()
    console.print(
        f"[bold]Launching {total_experiments} experiment(s) "
        f"with {total_expanded_tasks} task(s)[/bold]"
    )
    if split_models:
        console.print(
            "[dim]  Tasks distributed across multiple experiments due to GPU constraints[/dim]"
        )

    # Show experiment matrix if multiple experiments
    if total_experiments > 1:
        _print_experiment_matrix(experiment_plan, launch_config.runner_type)

    console.print()

    # Build job configs and summaries
    job_assembler = JobConfigAssembler(
        launch_config,
        eval_config,
        effective_image,
        effective_groups,
        launcher.beaker.user_name,
        common_secrets,
        store_secrets,
        task_secrets,
        inject_aws,
        inject_gcs,
    )

    job_configs = []
    experiment_summaries = []

    for exp in experiment_plan:
        job_config = job_assembler.assemble(exp)
        job_configs.append(job_config)

        exp_summary = _build_experiment_summary(
            exp, job_config, task_configs_by_spec, launch_config.runner_type
        )
        experiment_summaries.append(exp_summary)

    # Print experiment summaries
    for exp_summary in experiment_summaries:
        console.print(
            Panel(
                Pretty(exp_summary, expand_all=True),
                title=f"[bold]{exp_summary.name}[/bold]",
                border_style="cyan",
            )
        )
        console.print()

    # Confirm and launch
    if not dry_run and not yes and not click.confirm("Proceed with launch?", default=True):
        console.print("[yellow]Launch cancelled[/yellow]")
        raise SystemExit(0)

    launched_experiments: list[str] = []
    for job_config in job_configs:
        if not dry_run:
            experiment = launcher.launch(job_config)
            if experiment:
                console.print(f"[green]Launched:[/green] {launcher.experiment_url(experiment)}")
                launched_experiments.append(experiment.id)

    # Follow launched experiments
    if launched_experiments and not dry_run:
        _handle_follow(launcher, launched_experiments, follow)


def _handle_group_creation(launcher, effective_groups: list[str], dry_run: bool) -> None:
    """Handle checking and creating Beaker groups."""
    from beaker.exceptions import BeakerGroupNotFound

    existing_groups = []
    missing_groups = []

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
                raise SystemExit(1) from None

            workspace_obj = launcher.beaker.workspace.get(launcher._workspace)
            for grp in missing_groups:
                try:
                    beaker_group = launcher.beaker.group.create(name=grp, workspace=workspace_obj)
                    group_url = launcher.get_group_url(beaker_group)
                    console.print(f"[green]  Created {grp}:[/green] {group_url}")
                except Exception as e:
                    console.print(f"[red]Error:[/red] Failed to create group '{grp}': {e}")
                    raise SystemExit(1) from None


def _get_task_configs(
    valid_tasks: list[str], task_overrides: dict[str, list[str]] | None = None
) -> dict:
    """Get task configs with overrides applied.

    Args:
        valid_tasks: List of task specifications.
        task_overrides: Optional dict of task_spec -> override strings from CLI
                       (already filtered - priority extracted).

    Returns:
        Dict mapping task_spec -> TaskConfig (with overrides applied).
    """
    from copy import deepcopy

    from olmo_eval.evals.tasks import get_task as get_task_instance
    from olmo_eval.evals.tasks.core.registry import parse_task_spec

    task_overrides = task_overrides or {}
    task_configs = {}

    for task_spec in valid_tasks:
        task_name, variants, _overrides = parse_task_spec(task_spec)
        try:
            task_instance = get_task_instance(task_spec)
        except ValueError as e:
            console.print(f"[red]Error loading task '{task_spec}':[/red] {e}")
            raise SystemExit(1) from None

        # Deep copy the config so we can apply overrides directly
        task_cfg = deepcopy(task_instance.config)

        # Apply CLI overrides directly to config
        cli_overrides = task_overrides.get(task_spec, [])
        for override_str in cli_overrides:
            if "=" in override_str:
                key, value = override_str.split("=", 1)
                # Try to parse value as int/float/bool if applicable
                parsed_value: str | int | float | bool = value
                try:
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    elif "." in value:
                        parsed_value = float(value)
                    else:
                        parsed_value = int(value)
                except ValueError:
                    parsed_value = value

                if hasattr(task_cfg, key):
                    setattr(task_cfg, key, parsed_value)

        task_configs[task_spec] = task_cfg

    return task_configs


def _ensure_secrets(
    launcher, dry_run: bool, launch_config, all_required_secrets: set[str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    """Ensure required secrets exist."""
    from olmo_eval.launch.beaker.secrets import (
        ensure_common_secrets,
        ensure_task_secrets,
        get_local_hf_token,
        get_local_wandb_api_key,
    )

    beaker_username = launcher.beaker.user_name

    if dry_run:
        common_secrets = []
        if get_local_hf_token():
            common_secrets.append(("HF_TOKEN", f"{beaker_username}_HF_TOKEN"))
        if get_local_wandb_api_key():
            common_secrets.append(("WANDB_API_KEY", f"{beaker_username}_WANDB_API_KEY"))
        task_secrets = [(s, f"{beaker_username}_{s}") for s in sorted(all_required_secrets)]
    else:
        common_secrets = ensure_common_secrets(workspace=launch_config.workspace)
        try:
            task_secrets = ensure_task_secrets(
                workspace=launch_config.workspace, required_secrets=all_required_secrets
            )
        except ValueError as e:
            console.print(f"[red]Error:[/red] {e}")
            raise SystemExit(1) from None

    store_secrets = []
    if launch_config.store:
        for env_var in ["PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD"]:
            store_secrets.append((env_var, f"olmo_eval_{env_var}"))

    return common_secrets, store_secrets, task_secrets


def _print_experiment_matrix(
    experiment_plan: list["ExperimentPlan"], runner_type: RunnerType
) -> None:
    """Print experiment matrix table."""
    matrix_table = Table(show_header=True, title="Experiment Plan")
    matrix_table.add_column("Name", style="cyan")
    matrix_table.add_column("Models", style="blue")
    matrix_table.add_column("Provider", style="white")
    matrix_table.add_column("Tasks", style="dim")
    matrix_table.add_column("Runner", style="magenta")
    matrix_table.add_column("Priority", style="yellow")
    matrix_table.add_column("GPUs", style="green", justify="right")

    for exp in experiment_plan:
        # Model display
        model_display = (
            exp.model_cfgs[0].name_or_path
            if len(exp.model_cfgs) == 1
            else f"{len(exp.model_cfgs)} models"
        )

        # Provider display
        if len(exp.model_cfgs) == 1:
            provider = exp.model_cfgs[0].provider
            provider_display = provider.name if provider else "default"
        else:
            providers = {m.provider.name if m.provider else "default" for m in exp.model_cfgs}
            provider_display = ", ".join(sorted(providers))

        # Task display - show actual task names
        if len(exp.tasks) <= 3:
            task_display = ", ".join(exp.tasks)
        else:
            task_display = f"{exp.tasks[0]}, ... ({len(exp.tasks)} total)"

        # Runner display
        runner_names = {
            RunnerType.SYNC: "SyncEvalRunner",
            RunnerType.ASYNC: "AsyncEvalRunner",
            RunnerType.ASYNC_STREAM: "StreamingEvalRunner",
            RunnerType.AGENT: "AgentEvalRunner",
        }
        runner_display = runner_names.get(exp.runner_type, "SyncEvalRunner")

        matrix_table.add_row(
            exp.name,
            model_display,
            provider_display,
            task_display,
            runner_display,
            exp.priority,
            str(exp.num_gpus),
        )

    console.print(matrix_table)


def _build_experiment_summary(
    exp: "ExperimentPlan",
    job_config,
    task_configs_by_spec: dict,
    runner_type: RunnerType,
) -> ExperimentSummary:
    """Build experiment summary for display."""
    from olmo_eval.runners import (
        AgentEvalRunner,
        AsyncEvalRunner,
        StreamingEvalRunner,
        SyncEvalRunner,
    )

    # Build task configs list (with overrides already applied)
    exp_task_configs = []
    for task_spec in exp.tasks:
        base_spec = task_spec.rsplit("@", 1)[0] if "@" in task_spec else task_spec
        if base_spec in task_configs_by_spec:
            exp_task_configs.append(task_configs_by_spec[base_spec])

    # Determine runner class
    runner_classes = {
        RunnerType.SYNC: SyncEvalRunner,
        RunnerType.ASYNC: AsyncEvalRunner,
        RunnerType.ASYNC_STREAM: StreamingEvalRunner,
        RunnerType.AGENT: AgentEvalRunner,
    }
    exp_runner_class = runner_classes.get(exp.runner_type, SyncEvalRunner)

    exp_runner_config = RunnerConfig(
        runner=exp_runner_class,
        output_dir=BEAKER_RESULT_DIR,
    )

    return ExperimentSummary(
        name=exp.name,
        models=list(exp.model_cfgs),
        tasks=exp_task_configs,
        runner=exp_runner_config,
        beaker=job_config,
    )


def _handle_follow(launcher, launched_experiments: list[str], follow: bool) -> None:
    """Handle following launched experiments."""
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

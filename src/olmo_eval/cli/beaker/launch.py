"""Launch command for Beaker jobs.

This module provides the main 'beaker launch' command for submitting evaluation jobs.
Configuration loading, validation, and job assembly are delegated to:
- config_loader.py: LaunchConfigLoader for loading/merging config
- task_validator.py: TaskValidator for task validation and priority grouping
- credentials.py: CredentialManager for AWS/GCS credential handling
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
    HarnessSummary,
    OrderedMultiOption,
    RunnerConfig,
    console,
    extract_priority_from_overrides,
    process_ordered_args,
    reconstruct_ordered_args,
)
from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR, BEAKER_UV_CACHE_DIR


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
    help="Override for preceding -m or -t (e.g., -o provider.kind=vllm -o limit=100)",
)
@click.option("--cluster", "-c", default=None, help="Cluster alias (h100, a100, aus) or full name")
@click.option(
    "--max-gpus-per-node",
    default=None,
    type=int,
    help="Maximum GPUs per node (default: 8). Models are split across experiments if exceeded.",
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
    "--inspect",
    is_flag=True,
    help="Enable all inspection flags (instance, formatted, tokens, request, response)",
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
    "-H",
    "--harness",
    type=str,
    default=None,
    help="Harness preset name",
)
@click.option(
    "--uv-cache-dir",
    default=BEAKER_UV_CACHE_DIR,
    show_default=True,
    help="UV cache directory for package downloads (on Weka shared storage)",
)
@click.option(
    "--secret-env",
    multiple=True,
    help="Map Beaker secret to env var: BEAKER_SECRET:ENV_VAR (e.g., my-openai-key:OPENAI_API_KEY)",
)
def launch(
    config: str | None,
    name: str | None,
    model: tuple[str, ...],
    task: tuple[str, ...],
    override: tuple[str, ...],
    cluster: str | None,
    max_gpus_per_node: int | None,
    priority: str | None,
    preemptible: bool | None,
    timeout: str | None,
    retries: int | None,
    workspace: str | None,
    budget: str | None,
    image: str | None,
    group: tuple[str, ...],
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
    inspect: bool,
    inspect_instance: bool,
    inspect_formatted: bool,
    inspect_tokens: bool,
    inspect_response: bool,
    inspect_request: bool,
    harness: str | None,
    uv_cache_dir: str,
    secret_env: tuple[str, ...],
) -> None:
    """Launch an evaluation job on Beaker.

    Requires beaker-gantry to be installed: pip install 'olmo-eval-internal[beaker]'
    """
    from datetime import datetime

    try:
        from olmo_eval.launch import BeakerLauncher, EvalConfig
    except ImportError:
        console.print(
            "[red]beaker-gantry is not installed.[/red]\n"
            "Install with: pip install 'olmo-eval-internal[beaker]'"
        )
        raise SystemExit(1) from None

    # Process ordered args to associate overrides with models/tasks
    import sys

    from olmo_eval.cli.beaker.config_loader import LaunchConfigLoader
    from olmo_eval.cli.beaker.credentials import CredentialManager
    from olmo_eval.cli.beaker.experiment_builder import ExperimentPlanBuilder
    from olmo_eval.cli.beaker.job_assembler import JobConfigAssembler
    from olmo_eval.cli.beaker.task_validator import TaskValidator
    from olmo_eval.common.constants.infrastructure import (
        BEAKER_DEFAULT_IMAGE,
        BEAKER_SANDBOX_IMAGE,
    )

    ordered_args = reconstruct_ordered_args(sys.argv[1:])
    raw_task_overrides, harness_overrides = process_ordered_args(ordered_args)

    # Extract priority from task overrides (e.g., -o priority=urgent after -t)
    # This is done once here and the filtered overrides are used everywhere
    override_priority, task_overrides = extract_priority_from_overrides(raw_task_overrides)

    # Expand --inspect to enable all individual inspect flags
    if inspect:
        inspect_instance = True
        inspect_formatted = True
        inspect_tokens = True
        inspect_response = True
        inspect_request = True

    # Parse secret_env mappings: BEAKER_SECRET:ENV_VAR -> {beaker_secret: env_var}
    secret_env_overrides: dict[str, str] = {}
    for mapping in secret_env:
        if ":" not in mapping:
            console.print(
                f"[red]Error:[/red] Invalid --secret-env format '{mapping}'. "
                "Expected BEAKER_SECRET:ENV_VAR"
            )
            raise SystemExit(1)
        beaker_secret, env_var = mapping.split(":", 1)
        secret_env_overrides[beaker_secret] = env_var

    # Build CLI args dict
    cli_args = {
        "name": name,
        "model": model,
        "task": task,
        "task_overrides": task_overrides,  # Already filtered (priority extracted)
        "cluster": cluster,
        "max_gpus_per_node": max_gpus_per_node,
        "priority": priority,
        "preemptible": preemptible,
        "timeout": timeout,
        "retries": retries,
        "workspace": workspace,
        "budget": budget,
        "image": image,
        "group": group,
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
        "harness": harness,
        "harness_overrides": harness_overrides,
        "uv_cache_dir": uv_cache_dir,
        "secret_env_overrides": secret_env_overrides,
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
        default_priority=effective_priority,
    )
    tasks_by_priority, valid_tasks = task_validator.validate_and_group()

    # Create launcher
    launcher = BeakerLauncher(workspace=launch_config.workspace)

    # Set up credentials
    cred_manager = CredentialManager(
        launch_config.model_specs,
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
    # Check if harness has a sandbox configured with local deployment - if so, use sandbox image
    # Apply harness overrides first so -o sandbox.mode=docker works correctly
    harness_needs_sandbox = False
    if launch_config.harness:
        from olmo_eval.harness import get_harness_preset

        harness_preset = get_harness_preset(launch_config.harness)
        if launch_config.harness_overrides:
            harness_preset = _apply_harness_overrides(
                harness_preset, launch_config.harness_overrides
            )
        if harness_preset.sandboxes:
            harness_needs_sandbox = any(s.is_local for s in harness_preset.sandboxes)

    if image:
        effective_image = image
    elif eval_config and eval_config.beaker_image:
        effective_image = eval_config.beaker_image
    elif harness_needs_sandbox:
        effective_image = BEAKER_SANDBOX_IMAGE
    else:
        effective_image = BEAKER_DEFAULT_IMAGE

    # Handle group creation
    _handle_group_creation(launcher, effective_groups, dry_run, yes)

    # Build experiment plan
    experiment_builder = ExperimentPlanBuilder(launch_config, tasks_by_priority, override_priority)
    experiment_plan, split_models = experiment_builder.build()

    # Get task configs with overrides applied
    # (task_overrides is already filtered - priority extracted above)
    task_configs_by_spec = _get_task_configs(valid_tasks, launch_config.task_overrides)

    # Collect required secrets from tasks
    all_required_secrets: set[str] = set()
    for task_cfg in task_configs_by_spec.values():
        if hasattr(task_cfg, "required_secrets") and task_cfg.required_secrets:
            all_required_secrets.update(task_cfg.required_secrets)

    # Collect required secrets from model presets
    from olmo_eval.common.configs import get_provider_config

    for model_spec in launch_config.model_specs:
        provider_config = get_provider_config(model_spec)
        if provider_config.required_secrets:
            all_required_secrets.update(provider_config.required_secrets)

    # Collect harness-required secrets
    if launch_config.harness:
        from olmo_eval.harness import get_harness_preset

        harness_config = get_harness_preset(launch_config.harness)
        if harness_config.required_secrets:
            all_required_secrets.update(harness_config.required_secrets)
        # Collect sandbox-required secrets
        for sandbox in harness_config.sandboxes:
            if sandbox.required_secrets:
                all_required_secrets.update(sandbox.required_secrets)

    # Ensure secrets
    common_secrets, store_secrets, task_secrets = _ensure_secrets(
        launcher, dry_run, launch_config, all_required_secrets
    )

    # Print summary header
    total_experiments = len(experiment_plan)
    total_expanded_tasks = len(valid_tasks) * len(launch_config.model_specs)
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
        _print_experiment_matrix(experiment_plan)

    console.print()

    # Build job configs and summaries
    job_assembler = JobConfigAssembler(
        launch_config,
        effective_image,
        effective_groups,
        launcher.beaker.user_name,
        common_secrets,
        store_secrets,
        task_secrets,
        inject_aws,
        inject_gcs,
        enable_sandbox=harness_needs_sandbox,
        secret_env_overrides=launch_config.secret_env_overrides,
    )

    job_configs = []
    experiment_summaries = []

    for exp in experiment_plan:
        job_config = job_assembler.assemble(exp)
        job_configs.append(job_config)

        exp_summary = _build_experiment_summary(
            exp,
            job_config,
            task_configs_by_spec,
            launch_config.harness,
            launch_config.harness_overrides,
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


def _handle_group_creation(
    launcher, effective_groups: list[str], dry_run: bool, yes: bool = False
) -> None:
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
            confirmed = yes or click.confirm("Would you like to create these groups?", default=True)
            if not confirmed:
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

    from olmo_eval.evals.tasks.common import get_task as get_task_instance
    from olmo_eval.evals.tasks.common import parse_task_spec

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


def _print_experiment_matrix(experiment_plan: list["ExperimentPlan"]) -> None:
    """Print experiment matrix table."""
    matrix_table = Table(show_header=True, title="Experiment Plan")
    matrix_table.add_column("Name", style="cyan")
    matrix_table.add_column("Models", style="blue")
    matrix_table.add_column("Tasks", style="dim")
    matrix_table.add_column("Priority", style="yellow")
    matrix_table.add_column("GPUs", style="green", justify="right")

    for exp in experiment_plan:
        model_display = exp.model_spec

        if len(exp.tasks) <= 3:
            task_display = ", ".join(exp.tasks)
        else:
            task_display = f"{exp.tasks[0]}, ... ({len(exp.tasks)} total)"

        matrix_table.add_row(
            exp.name,
            model_display,
            task_display,
            exp.priority,
            str(exp.num_gpus),
        )

    console.print(matrix_table)


def _build_experiment_summary(
    exp: "ExperimentPlan",
    job_config,
    task_configs_by_spec: dict,
    harness: str | None = None,
    harness_overrides: list[str] | None = None,
) -> ExperimentSummary:
    """Build experiment summary for display."""
    from olmo_eval.runners import AsyncEvalRunner

    exp_task_configs = []
    for task_spec in exp.tasks:
        base_spec = task_spec.rsplit("@", 1)[0] if "@" in task_spec else task_spec
        if base_spec in task_configs_by_spec:
            exp_task_configs.append(task_configs_by_spec[base_spec])

    exp_runner_class = AsyncEvalRunner

    exp_runner_config = RunnerConfig(
        runner=exp_runner_class,
        output_dir=BEAKER_RESULT_DIR,
    )

    from olmo_eval.common.configs import get_provider_config
    from olmo_eval.harness import get_harness_preset

    harness_config = get_harness_preset(harness or "default")
    if harness_overrides:
        harness_config = _apply_harness_overrides(harness_config, harness_overrides)
    provider_config = get_provider_config(exp.model_spec)
    harness_config = harness_config.merge_provider(provider_config)

    harness_summary = HarnessSummary(config=harness_config)

    return ExperimentSummary(
        name=exp.name,
        tasks=exp_task_configs,
        harness=harness_summary,
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


def _apply_harness_overrides(harness_config, overrides: list[str]):
    """Apply CLI overrides to harness config.

    Args:
        harness_config: Base HarnessConfig to modify.
        overrides: List of dotlist override strings (e.g., ["sandbox.mode=docker"]).

    Returns:
        New HarnessConfig with overrides applied.
    """
    from omegaconf import OmegaConf

    from olmo_eval.harness import HarnessConfig

    harness_dict = harness_config.to_dict()
    override_config = OmegaConf.from_dotlist(overrides)
    base = OmegaConf.create(harness_dict)
    merged = OmegaConf.merge(base, override_config)
    harness_dict = OmegaConf.to_container(merged, resolve=True)
    return HarnessConfig.from_dict(harness_dict)  # type: ignore[arg-type]

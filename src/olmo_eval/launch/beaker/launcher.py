"""Unified Beaker launcher for olmo-eval jobs using beaker-gantry.

Dataclass-based API for submitting evaluation jobs
to Beaker using beaker-gantry's Python API for reproducible experiments.

Example:
    config = BeakerJobConfig(
        name="eval-llama3-mmlu",
        command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
        cluster="ai2/ceres",
        workspace="ai2/oe-data",
        budget="ai2/oe-base",
        num_gpus=1,
    )
    launcher = BeakerLauncher(workspace="ai2/oe-data")
    experiment = launcher.launch(config)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from olmo_eval.core.constants.infrastructure import (
    BEAKER_DEFAULT_IMAGE,
    BEAKER_DEFAULT_WORKSPACE,
    BEAKER_KNOWN_CLUSTERS,
    NEW_CLUSTER_ALIASES,
)

if TYPE_CHECKING:
    from beaker import Beaker, BeakerExperiment, BeakerGroup

log = logging.getLogger(__name__)

__all__ = [
    "BeakerEnvSecret",
    "BeakerWekaBucket",
    "BeakerJobConfig",
    "BeakerLauncher",
    "calculate_experiment_splits",
    "parse_task_with_priority",
    "validate_priority_configuration",
    "print_experiment_config",
    "resolve_clusters",
]

# Rich console for pretty printing
_console = Console()


def print_experiment_config(
    spec_dict: dict[str, Any],
    name: str | None = None,
    show_header: bool = True,
) -> None:
    """Pretty print a Beaker experiment spec with colorization.

    Uses Rich library for syntax-highlighted JSON output with optional
    header panel showing experiment metadata.

    Args:
        spec_dict: The experiment spec as a dictionary.
        name: Optional experiment name to display in header.
        show_header: Whether to show the header panel.

    Example:
        spec = launcher.build_spec(config)
        print_experiment_config(spec.to_json(), name=config.name)
    """
    # Extract key info for header
    tasks = spec_dict.get("tasks", [])
    task_spec = tasks[0] if tasks else {}
    context = task_spec.get("context", {})
    resources = task_spec.get("resources", {})
    constraints = task_spec.get("constraints", {})
    command = task_spec.get("command", [])

    # Build header with key metadata
    if show_header:
        header_lines = []
        if name:
            header_lines.append(f"[bold cyan]Experiment:[/] {name}")

        # Extract model and tasks from command
        model = None
        task_names = []
        for i, arg in enumerate(command):
            if arg == "-m" and i + 1 < len(command):
                model = command[i + 1]
            elif arg == "-t" and i + 1 < len(command):
                task_names.append(command[i + 1])

        if model:
            header_lines.append(f"[bold blue]Model:[/] {model}")
        if task_names:
            header_lines.append(f"[bold blue]Tasks:[/] {', '.join(task_names)}")

        # Resource info
        priority = context.get("priority", "normal")
        priority_color = {
            "low": "dim",
            "normal": "white",
            "high": "yellow",
            "urgent": "red bold",
        }.get(priority, "white")
        header_lines.append(f"[bold blue]Priority:[/] [{priority_color}]{priority}[/]")

        gpu_count = resources.get("gpuCount", 1)
        header_lines.append(f"[bold blue]GPUs:[/] {gpu_count}")

        clusters = constraints.get("cluster", [])
        if clusters:
            header_lines.append(f"[bold blue]Clusters:[/] {', '.join(clusters)}")

        preemptible = context.get("preemptible", True)
        preempt_str = "[green]yes[/]" if preemptible else "[red]no[/]"
        header_lines.append(f"[bold blue]Preemptible:[/] {preempt_str}")

        header_text = Text.from_markup("\n".join(header_lines))
        _console.print(Panel(header_text, title="[bold]Beaker Experiment[/]", border_style="blue"))

    # Print JSON with syntax highlighting
    json_str = json.dumps(spec_dict, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
    _console.print(syntax)


# Valid Beaker priority levels
VALID_PRIORITIES = ("low", "normal", "high", "urgent")


def parse_task_with_priority(task_spec: str, default_priority: str = "normal") -> tuple[str, str]:
    """Parse task spec with optional @priority suffix.

    Format: task_name[@priority] or task_name::regime[@priority]

    Examples:
        - "mmlu" -> ("mmlu", "normal")
        - "mmlu@high" -> ("mmlu", "high")
        - "mmlu::olmes@high" -> ("mmlu::olmes", "high")

    Args:
        task_spec: Task specification, optionally with @priority suffix.
        default_priority: Priority to use if not specified in task_spec.

    Returns:
        Tuple of (task_name, priority).

    Raises:
        ValueError: If priority is not valid.
    """
    if "@" in task_spec:
        task_name, priority = task_spec.rsplit("@", 1)
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. Must be one of: {', '.join(VALID_PRIORITIES)}"
            )
        return task_name, priority
    return task_spec, default_priority


def validate_priority_configuration(
    tasks: tuple[str, ...] | list[str],
    cli_priority: str | None,
    default_priority: str = "normal",
) -> dict[str, list[str]]:
    """Validate priority configuration and group tasks by priority.

    Detects conflicting priority specifications where --priority CLI flag
    is used together with @priority suffixes on tasks.

    Args:
        tasks: Task specifications (may include @priority suffixes).
        cli_priority: Priority from --priority CLI flag, or None.
        default_priority: Default priority for tasks without @priority suffix.

    Returns:
        Dictionary mapping priority levels to lists of task names.

    Raises:
        ValueError: If conflicting priority specifications detected
            (--priority used with @priority suffixes).

    Examples:
        # Tasks without @priority suffix, no CLI flag
        >>> validate_priority_configuration(["mmlu", "gsm8k"], None)
        {"normal": ["mmlu", "gsm8k"]}

        # Tasks with @priority suffixes
        >>> validate_priority_configuration(["mmlu@high", "gsm8k@normal"], None)
        {"high": ["mmlu"], "normal": ["gsm8k"]}

        # CLI flag with no @priority suffixes
        >>> validate_priority_configuration(["mmlu", "gsm8k"], "high")
        {"high": ["mmlu", "gsm8k"]}

        # CLI flag WITH @priority suffixes - raises error
        >>> validate_priority_configuration(["mmlu@high"], "normal")
        ValueError: Conflicting priority specification...
    """
    from collections import defaultdict

    tasks_by_priority: dict[str, list[str]] = defaultdict(list)
    tasks_with_priority_suffix: list[str] = []

    for task_spec in tasks:
        if "@" in task_spec:
            tasks_with_priority_suffix.append(task_spec)
            task_name, task_priority = parse_task_with_priority(task_spec)
            tasks_by_priority[task_priority].append(task_name)
        else:
            # Use CLI priority if provided, otherwise use default
            effective_priority = cli_priority if cli_priority is not None else default_priority
            tasks_by_priority[effective_priority].append(task_spec)

    # Conflict check: --priority used with @priority suffixes
    if cli_priority is not None and tasks_with_priority_suffix:
        raise ValueError(
            f"Conflicting priority specification: --priority={cli_priority} "
            f"cannot be used together with @priority suffixes on tasks.\n\n"
            f"Either:\n"
            f"  1. Use --priority to set priority for ALL tasks, or\n"
            f"  2. Use @priority suffixes to set priority per-task\n\n"
            f"Tasks with @priority suffixes:\n"
            + "\n".join(f"  - {t}" for t in tasks_with_priority_suffix)
        )

    return dict(tasks_by_priority)


def calculate_experiment_splits(
    tasks: list[str],
    gpus_per_model: int,
    parallelism: int,
    max_gpus_per_node: int,
) -> list[dict[str, Any]]:
    """Calculate how to split tasks across experiments based on GPU constraints.

    When the total GPU requirement (gpus_per_model × parallelism) exceeds the
    maximum GPUs available per node, tasks are distributed across multiple experiments.
    Each experiment runs a subset of tasks with reduced parallelism.

    Args:
        tasks: List of expanded task specs to run.
        gpus_per_model: GPUs required per model instance.
        parallelism: Number of model instances desired.
        max_gpus_per_node: Maximum GPUs available per node.

    Returns:
        List of dicts, each containing:
        - tasks: subset of tasks for this experiment
        - num_gpus: GPUs to request for this experiment
        - parallelism: effective parallelism for this split
    """
    import math

    total_gpus_needed = gpus_per_model * parallelism

    if total_gpus_needed <= max_gpus_per_node:
        # Fits on single node - no splitting needed
        return [
            {
                "tasks": tasks,
                "num_gpus": total_gpus_needed,
                "parallelism": parallelism,
            }
        ]

    # Need to split across multiple experiments
    # Calculate how many instances can fit per experiment
    instances_per_experiment = max_gpus_per_node // gpus_per_model
    if instances_per_experiment < 1:
        # Edge case: single model instance needs more GPUs than max
        # Fall back to 1 instance per experiment
        instances_per_experiment = 1

    gpus_per_experiment = instances_per_experiment * gpus_per_model
    num_experiments = math.ceil(parallelism / instances_per_experiment)

    # Distribute tasks across experiments
    tasks_per_experiment = math.ceil(len(tasks) / num_experiments)
    splits = []

    for i in range(num_experiments):
        start_idx = i * tasks_per_experiment
        end_idx = min(start_idx + tasks_per_experiment, len(tasks))
        split_tasks = tasks[start_idx:end_idx]

        if split_tasks:  # Only add if there are tasks
            splits.append(
                {
                    "tasks": split_tasks,
                    "num_gpus": gpus_per_experiment,
                    "parallelism": instances_per_experiment,
                }
            )

    return splits


@dataclass
class BeakerEnvSecret:
    """Environment variable sourced from a Beaker secret.

    Attributes:
        name: Environment variable name to set in the container.
        secret: Name of the secret in Beaker's secret store.
    """

    name: str
    secret: str


@dataclass
class BeakerWekaBucket:
    """Weka bucket mount configuration.

    Attributes:
        bucket: Weka bucket name (e.g., "oe-eval-default").
        mount: Mount path in the container. Defaults to /weka/{bucket}.
    """

    bucket: str
    mount: str | None = None

    def __post_init__(self) -> None:
        if self.mount is None:
            self.mount = f"/weka/{self.bucket}"


@dataclass
class BeakerJobConfig:
    """Configuration for a Beaker evaluation job.

    This dataclass provides sensible defaults while allowing full customization
    of job parameters. Use `BeakerLauncher.launch()` to submit jobs.

    Example:
        config = BeakerJobConfig(
            name="eval-llama3-mmlu",
            command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
            cluster="ai2/ceres",
            workspace="ai2/oe-data",
            budget="ai2/oe-base",
            num_gpus=1,
        )

    Attributes:
        name: Experiment name (required).
        command: Command to run in the container (required).
        cluster: Cluster alias ("h100", "a100", "aus") or full name(s) (required).
        workspace: Beaker workspace (required).
        budget: Beaker budget (required).
        num_gpus: Number of GPUs to request.
        shared_memory: Shared memory size (e.g., "10GiB").
        priority: Job priority level.
        preemptible: Whether the job can be preempted.
        timeout: Job timeout (e.g., "24h", "30m").
        retries: Number of retries on failure.
        beaker_image: Container image to use.
        description: Optional job description.
        weka_buckets: Weka storage mounts.
        nfs: Whether to mount NFS.
        env_vars: Additional environment variables.
        env_secrets: Environment variables from Beaker secrets.
        result_path: Path for job results.
    """

    # Required
    name: str
    command: list[str]
    cluster: str | list[str]  # Cluster alias ("h100", "a100", "aus") or full name(s)
    workspace: str  # Beaker workspace
    budget: str  # Beaker budget

    # Resources
    num_gpus: int = 1
    shared_memory: str = "10GiB"

    # Job settings
    priority: str = "normal"
    preemptible: bool = True
    timeout: str | None = "24h"
    retries: int | None = None

    # Beaker settings
    beaker_image: str = BEAKER_DEFAULT_IMAGE
    description: str | None = None

    # Storage - defaults include common eval buckets
    weka_buckets: list[BeakerWekaBucket] = field(
        default_factory=lambda: [
            BeakerWekaBucket("oe-training-default"),
            BeakerWekaBucket("oe-eval-default"),
        ]
    )
    nfs: bool = False

    # Environment - defaults include HuggingFace cache on Weka
    env_vars: dict[str, str] = field(
        default_factory=lambda: {
            "HF_HOME": "/weka/oe-eval-default/oyvindt/hf-cache",
            "HF_HUB_CACHE": "/weka/oe-eval-default/oyvindt/hf-cache",
        }
    )
    env_secrets: list[BeakerEnvSecret] = field(default_factory=list)

    # Result path
    result_path: str = "/results"

    # Optional dependency groups to install at runtime (e.g., ["vllm", "postgres"])
    extras: list[str] = field(default_factory=list)

    # Group assignment - experiment will be added to these groups at creation time
    groups: list[str] | None = None

    # AWS S3 access - when True, injects user's AWS credentials as env secrets
    inject_aws_credentials: bool = False

    # GCS access - when True, injects user's GCS credentials as env secret
    inject_gcs_credentials: bool = False


def resolve_clusters(cluster: str | list[str]) -> list[str]:
    """Resolve cluster aliases to full cluster names.

    Supports:
    - Aliases: "h100", "a100", "aus", "goog", "80g", etc.
    - Full names: "ai2/jupiter", "ai2/saturn", etc.
    - Legacy names: "ai2/jupiter-cirrascale-2" -> "ai2/jupiter"

    Args:
        cluster: Single cluster or list of clusters/aliases.

    Returns:
        List of resolved cluster names.
    """
    clusters = [cluster] if isinstance(cluster, str) else list(cluster)
    resolved: list[str] = []

    for c in clusters:
        # Check if it's a legacy name that needs aliasing
        if c in NEW_CLUSTER_ALIASES:
            c = NEW_CLUSTER_ALIASES[c]

        # Check if it's a known alias group
        if c in BEAKER_KNOWN_CLUSTERS:
            resolved.extend(BEAKER_KNOWN_CLUSTERS[c])
        else:
            resolved.append(c)

    return list(set(resolved))  # Deduplicate


def _parse_timeout(timeout: str) -> int:
    """Parse timeout string to nanoseconds.

    Supports formats: "24h", "30m", "1h30m", "90s".

    Args:
        timeout: Timeout string.

    Returns:
        Timeout in nanoseconds.
    """
    total_ns = 0
    patterns = [
        (r"(\d+)h", 3600_000_000_000),
        (r"(\d+)m", 60_000_000_000),
        (r"(\d+)s", 1_000_000_000),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, timeout)
        if match:
            total_ns += int(match.group(1)) * multiplier

    return total_ns if total_ns else 86400_000_000_000  # Default 24h


class BeakerLauncher:
    """Launches evaluation jobs on Beaker using beaker-gantry.

    This class provides a clean API for submitting Beaker experiments
    using gantry's Python API for reproducible, git-tracked experiments.

    Example:
        launcher = BeakerLauncher()

        # Launch the job
        experiment = launcher.launch(config)
        print(f"Experiment: {launcher.experiment_url(experiment)}")

        # Or do a dry run
        launcher.launch(config, dry_run=True)

    Attributes:
        beaker: Lazy-initialized Beaker client (for group operations).
    """

    def __init__(self, workspace: str | None = None) -> None:
        """Initialize the launcher.

        Args:
            workspace: Beaker workspace for the launcher. Defaults to BEAKER_DEFAULT_WORKSPACE
                for read-only operations (results, watch, group info). When launching jobs,
                callers should always provide an explicit workspace.
        """
        self._workspace = workspace or BEAKER_DEFAULT_WORKSPACE
        self._beaker: Beaker | None = None

    @property
    def beaker(self) -> Beaker:
        """Lazy-initialized Beaker client."""
        if self._beaker is None:
            from beaker import Beaker

            self._beaker = Beaker.from_env(default_workspace=self._workspace)
        return self._beaker

    def _build_command_with_extras(
        self,
        command: list[str],
        extras: list[str],
        env_exports: dict[str, str] | None = None,
    ) -> list[str]:
        """Build command with source installation and extras installation prepended.

        Gantry clones the source code to /gantry-runtime, so we:
        1. Install olmo-eval from the cloned source with optional extras
        2. Run the actual command

        Args:
            command: The command to run after setup.
            extras: Optional dependency group names from pyproject.toml.
            env_exports: Optional dict of environment variables to export before running.
        """
        # Build the full command
        # Export UV_PROJECT_ENVIRONMENT so all uv commands use Docker's /opt/venv
        steps = ["export UV_PROJECT_ENVIRONMENT=/opt/venv"]

        # Export additional environment variables (e.g., UV_CACHE_DIR)
        if env_exports:
            for key, value in env_exports.items():
                steps.append(f"export {key}={value}")

        # Install olmo-eval from gantry-cloned source with optional extras
        # Generate constraints from pre-installed CUDA packages to prevent uv from changing them
        constraints = "/tmp/cuda-constraints.txt"
        steps.append(f"uv pip freeze -q | grep -E '^(torch|nvidia-)' > {constraints}")
        if extras:
            extras_str = ",".join(extras)
            install_cmd = f"uv pip install -e '.[{extras_str}]' -c {constraints}"
            steps.append(f"cd /gantry-runtime && {install_cmd}")
        else:
            steps.append(f"cd /gantry-runtime && uv pip install -e . -c {constraints}")

        # Run the actual command (use shlex.join to properly quote special characters)
        import shlex

        steps.append(shlex.join(command))

        wrapped_command = ["bash", "-c", " && ".join(steps)]
        return wrapped_command

    def launch(self, config: BeakerJobConfig, dry_run: bool = False) -> BeakerExperiment | None:
        """Launch an experiment on Beaker using gantry.

        Args:
            config: Job configuration.
            dry_run: If True, print spec and exit without launching.

        Returns:
            Experiment object if launched, None if dry_run.
        """
        from gantry.api import launch_experiment

        clusters = resolve_clusters(config.cluster)

        # Build env vars that need to be exported in the shell command (before uv runs)
        env_exports: dict[str, str] = {}
        if "UV_CACHE_DIR" in config.env_vars:
            env_exports["UV_CACHE_DIR"] = config.env_vars["UV_CACHE_DIR"]

        final_command = self._build_command_with_extras(config.command, config.extras, env_exports)

        # Build weka mounts as tuples: (bucket, mount_path)
        weka_mounts: list[tuple[str, str]] = []
        for bucket in config.weka_buckets:
            assert bucket.mount is not None  # Set by __post_init__
            weka_mounts.append((bucket.bucket, bucket.mount))

        # Build env secrets as tuples: (env_var_name, secret_name)
        env_secrets: list[tuple[str, str]] = [
            (secret.name, secret.secret) for secret in config.env_secrets
        ]

        # Inject AWS credentials if requested
        if config.inject_aws_credentials:
            from olmo_eval.launch.beaker.aws import ensure_aws_secrets

            aws_secrets = ensure_aws_secrets(config.workspace)
            env_secrets.extend(aws_secrets)
            log.info("Injecting AWS credentials for S3 access")

        # Inject GCS credentials if requested
        google_credentials_secret: str | None = None
        if config.inject_gcs_credentials:
            from olmo_eval.launch.beaker.gcs import ensure_gcs_secrets

            google_credentials_secret = ensure_gcs_secrets(config.workspace)
            log.info("Injecting GCS credentials for GCS access")

        # Build env vars as tuples: (name, value)
        env_vars: list[tuple[str, str]] = list(config.env_vars.items())

        # Build mounts for NFS if requested
        mounts: list[tuple[str, str]] | None = None
        if config.nfs:
            mounts = [("/net/nfs.cirrascale", "/net/nfs.cirrascale")]

        # Build group names list if groups are specified
        group_names: list[str] | None = config.groups if config.groups else None

        # If doing a dry run, print config summary and return
        if dry_run:
            self._print_dry_run_config(config, clusters)
            # Still call launch_experiment with dry_run=True to show full spec
            # Pass groups so they appear in the spec output (the CLI handles
            # checking which groups exist and informing the user)
            launch_experiment(
                args=final_command,
                name=config.name,
                description=config.description,
                workspace=config.workspace,
                group_names=group_names,
                clusters=clusters,
                gpus=config.num_gpus,
                shared_memory=config.shared_memory,
                priority=config.priority,
                preemptible=config.preemptible,
                task_timeout=config.timeout,
                retries=config.retries,
                budget=config.budget,
                beaker_image=config.beaker_image,
                weka=weka_mounts if weka_mounts else None,
                env_vars=env_vars if env_vars else None,
                env_secrets=env_secrets if env_secrets else None,
                google_credentials_secret=google_credentials_secret,
                mounts=mounts,
                results=config.result_path,
                no_python=True,  # Use pre-built image, skip Python setup
                dry_run=True,
                yes=True,  # Skip confirmation prompts
            )
            return None

        # Launch the experiment
        workload = launch_experiment(
            args=final_command,
            name=config.name,
            description=config.description,
            workspace=config.workspace,
            group_names=group_names,
            clusters=clusters,
            gpus=config.num_gpus,
            shared_memory=config.shared_memory,
            priority=config.priority,
            preemptible=config.preemptible,
            task_timeout=config.timeout,
            retries=config.retries,
            budget=config.budget,
            beaker_image=config.beaker_image,
            weka=weka_mounts if weka_mounts else None,
            env_vars=env_vars if env_vars else None,
            env_secrets=env_secrets if env_secrets else None,
            google_credentials_secret=google_credentials_secret,
            mounts=mounts,
            results=config.result_path,
            no_python=True,  # Use pre-built image, skip Python setup
            timeout=0,  # Don't wait for completion
            yes=True,  # Skip confirmation prompts
        )

        if workload is None:
            log.warning("Gantry returned None workload - experiment may not have been created")
            return None

        # Get the experiment from the workload
        # The workload contains an embedded experiment object from gantry
        experiment = workload.experiment
        log.info(f"Experiment submitted: {self.experiment_url(experiment)}")
        return experiment

    def _print_dry_run_config(self, config: BeakerJobConfig, clusters: list[str]) -> None:
        """Print a summary of the config for dry run mode."""
        # Build header with key metadata
        header_lines = []
        header_lines.append(f"[bold cyan]Experiment:[/] {config.name}")

        # Extract model and tasks from command
        model = None
        task_names = []
        for i, arg in enumerate(config.command):
            if arg == "-m" and i + 1 < len(config.command):
                model = config.command[i + 1]
            elif arg == "-t" and i + 1 < len(config.command):
                task_names.append(config.command[i + 1])

        if model:
            header_lines.append(f"[bold blue]Model:[/] {model}")
        if task_names:
            header_lines.append(f"[bold blue]Tasks:[/] {', '.join(task_names)}")

        # Resource info
        priority = config.priority
        priority_color = {
            "low": "dim",
            "normal": "white",
            "high": "yellow",
            "urgent": "red bold",
        }.get(priority, "white")
        header_lines.append(f"[bold blue]Priority:[/] [{priority_color}]{priority}[/]")
        header_lines.append(f"[bold blue]GPUs:[/] {config.num_gpus}")
        header_lines.append(f"[bold blue]Clusters:[/] {', '.join(clusters)}")
        header_lines.append(f"[bold blue]Image:[/] {config.beaker_image}")

        preempt_str = "[green]yes[/]" if config.preemptible else "[red]no[/]"
        header_lines.append(f"[bold blue]Preemptible:[/] {preempt_str}")

        header_text = Text.from_markup("\n".join(header_lines))
        _console.print(Panel(header_text, title="[bold]Beaker Experiment[/]", border_style="blue"))

    def experiment_url(self, experiment: BeakerExperiment) -> str:
        """Get the Beaker URL for an experiment.

        Args:
            experiment: The experiment object.

        Returns:
            URL to view the experiment in Beaker.
        """
        return self.beaker.experiment.url(experiment)

    # -------------------------------------------------------------------------
    # Group Management
    # -------------------------------------------------------------------------

    def get_or_create_group(
        self,
        name: str,
        workspace: str | None = None,
        description: str | None = None,
    ) -> BeakerGroup:
        """Get existing group or create a new one.

        Args:
            name: Group name (can be short name or user-qualified).
            workspace: Workspace for the group. Uses default if None.
            description: Optional description for new groups.

        Returns:
            The existing or newly created group.
        """
        from beaker.exceptions import BeakerGroupConflict, BeakerGroupNotFound

        # Try to get existing group - needs user-qualified name for lookup
        # If name doesn't contain "/", try with current user prefix
        qualified_name = f"{self.beaker.user_name}/{name}" if "/" not in name else name

        try:
            return self.beaker.group.get(qualified_name)
        except BeakerGroupNotFound:
            try:
                # Get workspace object for the API call
                ws_name = workspace or self._workspace
                ws_obj = self.beaker.workspace.get(ws_name) if ws_name else None
                return self.beaker.group.create(
                    name=name,  # Create uses short name
                    workspace=ws_obj,
                    description=description,
                )
            except BeakerGroupConflict:
                # Group was created between our get and create calls, fetch it
                return self.beaker.group.get(qualified_name)

    def add_experiments_to_group(
        self,
        group: str | BeakerGroup,
        experiment_ids: list[str],
    ) -> BeakerGroup:
        """Add experiments to a group.

        Args:
            group: Group name or object.
            experiment_ids: List of experiment IDs to add.

        Returns:
            Updated group object.
        """
        # Convert string to Group object if needed
        group_obj = self.beaker.group.get(group) if isinstance(group, str) else group
        return self.beaker.group.update(
            group_obj,
            add_experiment_ids=experiment_ids,
        )

    def _get_experiment_ids_from_group(self, group: str | BeakerGroup) -> set[str]:
        """Get unique experiment IDs from a group using task metrics.

        Args:
            group: Group name or object.

        Returns:
            Set of experiment IDs.
        """
        if isinstance(group, str):
            group = self.beaker.group.get(group)
        task_metrics = self.beaker.group.list_task_metrics(group)
        return {tm.experiment_id for tm in task_metrics}

    def get_group_status(self, group: str | BeakerGroup) -> dict[str, int]:
        """Get status summary for all experiments in a group.

        Args:
            group: Group name or object.

        Returns:
            Dictionary mapping status to count.
        """
        from beaker import BeakerWorkloadStatus

        experiment_ids = self._get_experiment_ids_from_group(group)
        status_counts = {
            "succeeded": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "canceled": 0,
        }

        for exp_id in experiment_ids:
            # Get the workload status
            workload = self.beaker.workload.get(exp_id)
            status = workload.status

            if status == BeakerWorkloadStatus.succeeded:
                status_counts["succeeded"] += 1
            elif status == BeakerWorkloadStatus.failed:
                status_counts["failed"] += 1
            elif status == BeakerWorkloadStatus.canceled:
                status_counts["canceled"] += 1
            elif status in (
                BeakerWorkloadStatus.running,
                BeakerWorkloadStatus.uploading_results,
            ):
                status_counts["running"] += 1
            else:
                # queued, initializing, submitted
                status_counts["pending"] += 1

        return status_counts

    def get_group_experiments(
        self,
        group: str | BeakerGroup,
    ) -> list[BeakerExperiment]:
        """Get all experiments in a group.

        Args:
            group: Group name or object.

        Returns:
            List of experiment objects.
        """
        experiment_ids = self._get_experiment_ids_from_group(group)
        experiments = []
        for exp_id in experiment_ids:
            workload = self.beaker.workload.get(exp_id)
            experiments.append(workload.experiment)
        return experiments

    def export_group_metrics(self, group: str | BeakerGroup) -> str:
        """Export group metrics as CSV.

        Args:
            group: Group name or object.

        Returns:
            CSV string with experiment metrics.
        """
        # Convert string to Group object if needed
        group_obj = self.beaker.group.get(group) if isinstance(group, str) else group
        return self.beaker.group.export_metrics(group_obj)

    def cancel_group(self, group: str | BeakerGroup) -> dict[str, int]:
        """Cancel all active experiments in a group.

        Stops all running and pending experiments in the group. Experiments that
        have already completed (succeeded, failed, or canceled) are skipped.

        Args:
            group: Group name or object.

        Returns:
            Dictionary with counts: {"canceled": N, "skipped": M, "failed": K}
        """
        from beaker import BeakerWorkloadStatus
        from beaker.exceptions import BeakerExperimentConflict, BeakerExperimentNotFound

        experiment_ids = self._get_experiment_ids_from_group(group)
        results = {"canceled": 0, "skipped": 0, "failed": 0}

        for exp_id in experiment_ids:
            # Check current status before attempting to cancel
            workload = self.beaker.workload.get(exp_id)
            status = workload.status

            # Skip already-completed experiments
            if status in (
                BeakerWorkloadStatus.succeeded,
                BeakerWorkloadStatus.failed,
                BeakerWorkloadStatus.canceled,
            ):
                results["skipped"] += 1
                continue

            # Attempt to cancel the workload
            try:
                list(self.beaker.workload.cancel(workload))
                results["canceled"] += 1
            except (BeakerExperimentNotFound, BeakerExperimentConflict):
                # Already completed or canceled while we were iterating
                results["skipped"] += 1
            except Exception:
                results["failed"] += 1

        return results

    def get_group_url(self, group: BeakerGroup) -> str:
        """Get the Beaker URL for a group.

        Args:
            group: The group object.

        Returns:
            URL to view the group in Beaker.
        """
        return f"https://beaker.org/gr/{group.id}"

    # -------------------------------------------------------------------------
    # Experiment Following / Watching
    # -------------------------------------------------------------------------

    def follow_experiment(
        self,
        experiment_id: str,
        tail: bool = False,
    ) -> int:
        """Follow an experiment's logs until completion.

        Streams logs in real-time, showing startup events and job output.

        Args:
            experiment_id: Beaker experiment ID.
            tail: If True, only show last 10 seconds of logs (useful for
                  attaching to already-running experiments).

        Returns:
            Exit code: 0 for success, non-zero for failure.
        """
        import time
        from datetime import datetime, timedelta

        # Get the workload (experiment)
        workload = self.beaker.workload.get(experiment_id)
        workload_url = self.beaker.workload.url(workload)

        # Phase 1: Wait for job creation
        job = self.beaker.workload.get_latest_job(workload)
        if job is None:
            _console.print("[dim]Waiting for job to be created...[/dim]")
            while job is None:
                time.sleep(1.0)
                workload = self.beaker.workload.get(experiment_id)
                job = self.beaker.workload.get_latest_job(workload)

        # Phase 2: Wait for job to start, showing events
        events_seen: set[str] = set()
        assert job is not None  # We waited for job creation above
        job = self.beaker.job.get(job.id)
        while not (job.status.finalized or job.status.started):
            for event in self.beaker.job.list_summarized_events(job):
                # Use latest_message as key since events don't have stable IDs
                event_key = f"{event.status}:{event.latest_message}"
                if event_key not in events_seen:
                    events_seen.add(event_key)
                    _console.print(f"[cyan]>[/cyan] {event.latest_message}")
            time.sleep(1.0)
            job = self.beaker.job.get(job.id)

        # Phase 3: Stream logs
        _console.print("\n[bold]Logs:[/bold]")
        try:
            since = datetime.now(UTC) - timedelta(seconds=10) if tail else None
            for log_entry in self.beaker.job.logs(job, follow=True, since=since):
                # log_entry.message is bytes, decode it
                if log_entry.message:
                    line = log_entry.message.decode(errors="ignore").rstrip("\n")
                    if line:
                        print(line)
        except KeyboardInterrupt:
            _console.print("\n[yellow]Interrupted. Experiment continues running.[/yellow]")
            _console.print(f"[dim]View at: {workload_url}[/dim]")
            return 130  # Standard exit code for SIGINT

        # Phase 4: Check exit status
        _console.print()
        job = self.beaker.job.get(job.id)
        exit_code = job.status.exit_code

        if exit_code is None:
            _console.print(f"[red]Experiment failed[/red]: {workload_url}")
            return 1
        elif exit_code > 0:
            _console.print(f"[red]Experiment exited with code {exit_code}[/red]")
            return exit_code
        else:
            _console.print("[green]Experiment completed successfully[/green]")
            return 0

    def get_experiment_url(self, experiment_id: str) -> str:
        """Get the Beaker URL for an experiment by ID.

        Args:
            experiment_id: The experiment ID.

        Returns:
            URL to view the experiment in Beaker.
        """
        return f"https://beaker.org/ex/{experiment_id}"

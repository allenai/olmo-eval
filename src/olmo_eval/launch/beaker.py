"""Unified Beaker launcher for olmo-eval jobs using beaker-gantry.

Dataclass-based API for submitting evaluation jobs
to Beaker using beaker-gantry's Python API for reproducible experiments.

Example:
    config = BeakerJobConfig(
        name="eval-llama3-mmlu",
        command=["olmo-eval", "run", "-m", "llama3.1-8b", "-t", "mmlu"],
        cluster="h100",
        num_gpus=1,
    )
    launcher = BeakerLauncher()
    experiment = launcher.launch(config)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from olmo_eval.core.constants.infrastructure import (
    BEAKER_DEFAULT_BUDGET,
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
    "parse_task_with_priority",
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
            cluster="h100",
            num_gpus=1,
        )

    Attributes:
        name: Experiment name (required).
        command: Command to run in the container (required).
        num_gpus: Number of GPUs to request.
        shared_memory: Shared memory size (e.g., "10GiB").
        cluster: Cluster alias ("h100", "a100", "aus") or full name(s).
        priority: Job priority level.
        preemptible: Whether the job can be preempted.
        timeout: Job timeout (e.g., "24h", "30m").
        retries: Number of retries on failure.
        workspace: Beaker workspace.
        budget: Beaker budget.
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

    # Resources
    num_gpus: int = 1
    shared_memory: str = "10GiB"

    # Cluster - supports aliases like "h100", "aus", or full names
    cluster: str | list[str] = "h100"

    # Job settings
    priority: str = "normal"
    preemptible: bool = True
    timeout: str | None = "24h"
    retries: int | None = None

    # Beaker settings
    workspace: str = BEAKER_DEFAULT_WORKSPACE
    budget: str = BEAKER_DEFAULT_BUDGET
    beaker_image: str = BEAKER_DEFAULT_IMAGE
    description: str | None = None

    # Storage - defaults include common eval buckets
    weka_buckets: list[BeakerWekaBucket] = field(
        default_factory=lambda: [
            BeakerWekaBucket("oe-eval-default"),
            BeakerWekaBucket("oe-data-default"),
        ]
    )
    nfs: bool = False

    # Environment - defaults include HF and WandB tokens
    env_vars: dict[str, str] = field(default_factory=dict)
    env_secrets: list[BeakerEnvSecret] = field(
        default_factory=lambda: [
            BeakerEnvSecret("HF_TOKEN", "HF_TOKEN"),
            BeakerEnvSecret("WANDB_API_KEY", "WANDB_API_KEY"),
        ]
    )

    # Result path
    result_path: str = "/results"


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
            workspace: Override default workspace. Uses BEAKER_DEFAULT_WORKSPACE if None.
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

        # Build weka mounts as tuples: (bucket, mount_path)
        weka_mounts: list[tuple[str, str]] = []
        for bucket in config.weka_buckets:
            assert bucket.mount is not None  # Set by __post_init__
            weka_mounts.append((bucket.bucket, bucket.mount))

        # Build env secrets as tuples: (env_var_name, secret_name)
        env_secrets: list[tuple[str, str]] = [
            (secret.name, secret.secret) for secret in config.env_secrets
        ]

        # Build env vars as tuples: (name, value)
        env_vars: list[tuple[str, str]] = list(config.env_vars.items())

        # Build mounts for NFS if requested
        mounts: list[tuple[str, str]] | None = None
        if config.nfs:
            mounts = [("/net/nfs.cirrascale", "/net/nfs.cirrascale")]

        # If doing a dry run, print config summary and return
        if dry_run:
            self._print_dry_run_config(config, clusters)
            # Still call launch_experiment with dry_run=True to show full spec
            launch_experiment(
                args=config.command,
                name=config.name,
                description=config.description,
                workspace=config.workspace,
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
                mounts=mounts,
                results=config.result_path,
                no_python=True,  # Use pre-built image, skip Python setup
                dry_run=True,
                yes=True,  # Skip confirmation prompts
            )
            return None

        # Launch the experiment
        workload = launch_experiment(
            args=config.command,
            name=config.name,
            description=config.description,
            workspace=config.workspace,
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
        # The workload contains an embedded experiment object, but we need the full one
        # from the Beaker client for URL generation
        experiment = self.beaker.experiment.get(workload.experiment.id)
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
            name: Group name.
            workspace: Workspace for the group. Uses default if None.
            description: Optional description for new groups.

        Returns:
            The existing or newly created group.
        """
        from beaker.exceptions import BeakerGroupNotFound

        try:
            return self.beaker.group.get(name)
        except BeakerGroupNotFound:
            return self.beaker.group.create(
                name=name,
                workspace=workspace or self._workspace,
                description=description,
            )

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
        return self.beaker.group.update(
            group,
            add_experiment_ids=experiment_ids,
        )

    def get_group_status(self, group: str | BeakerGroup) -> dict[str, int]:
        """Get status summary for all experiments in a group.

        Args:
            group: Group name or object.

        Returns:
            Dictionary mapping status to count.
        """
        from beaker import BeakerWorkloadStatus

        experiments = list(self.beaker.group.experiments(group))
        status_counts = {
            "succeeded": 0,
            "failed": 0,
            "running": 0,
            "pending": 0,
            "canceled": 0,
        }

        for exp in experiments:
            # Get the workload status
            workload = self.beaker.workload.get(exp.id)
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
        return list(self.beaker.group.experiments(group))

    def export_group_metrics(self, group: str | BeakerGroup) -> str:
        """Export group metrics as CSV.

        Args:
            group: Group name or object.

        Returns:
            CSV string with experiment metrics.
        """
        return self.beaker.group.export_metrics(group)

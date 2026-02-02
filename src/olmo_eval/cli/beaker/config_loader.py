"""Configuration loading for Beaker launch command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console

from olmo_eval.core.types import RunnerType

if TYPE_CHECKING:
    from olmo_eval.launch import BeakerModelSpec, EvalConfig

console = Console()


@dataclass
class LaunchConfig:
    """Parsed and validated configuration for Beaker launch."""

    # Required fields
    name: str
    model_configs: list[BeakerModelSpec]
    model_specs: list[str]  # Original specs (without -o overrides applied via CLI)
    task_specs: list[str]
    cluster: str
    workspace: str
    budget: str

    # Per-model and per-task overrides from -o flag
    # model_overrides is positional (list index corresponds to model index)
    model_overrides: list[list[str]] = field(default_factory=list)
    task_overrides: dict[str, list[str]] = field(default_factory=dict)

    # Optional fields with defaults
    max_gpus_per_node: int = 8
    pack_models: bool = False  # Default: each model in own experiment for easier scheduling
    priority: str = "normal"
    preemptible: bool = True
    timeout: str = "24h"
    retries: int | None = None
    image: str | None = None
    groups: list[str] = field(default_factory=list)

    # Runner type and worker options
    runner_type: RunnerType = RunnerType.SYNC
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # S3 options
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"

    # Other flags
    store: bool = False
    debug_requests: bool = False
    debug_provider: bool = False
    save_predictions: bool = True
    save_requests: bool = True
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    # Credential injection
    inject_aws_credentials: bool = False
    inject_gcs_credentials: bool = False

    # UV cache directory
    uv_cache_dir: str | None = None


class LaunchConfigLoader:
    """Loads and merges configuration from YAML file and CLI arguments."""

    def __init__(
        self,
        config_path: str | None,
        cli_args: dict[str, Any],
    ):
        """Initialize the loader.

        Args:
            config_path: Path to YAML config file, or None.
            cli_args: Dictionary of CLI arguments.
        """
        self.config_path = config_path
        self.cli_args = cli_args

    def load(self) -> LaunchConfig:
        """Load and merge configuration.

        Returns:
            Merged LaunchConfig.

        Raises:
            SystemExit: If required fields are missing or config file is invalid.
        """
        from olmo_eval.core.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE
        from olmo_eval.launch import EvalConfig, parse_model_config

        cfg: EvalConfig | None = None
        model_configs: list = []
        model_specs: list[str] = []

        # Load config from file if provided
        if self.config_path:
            try:
                cfg = EvalConfig.from_yaml(self.config_path)
            except FileNotFoundError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1) from None
            except Exception as e:
                console.print(f"[red]Config error:[/red] {e}")
                raise SystemExit(1) from None

        # Extract CLI values
        cli_name = self.cli_args.get("name")
        cli_model = self.cli_args.get("model", ())
        cli_task = self.cli_args.get("task", ())
        cli_model_overrides: list[list[str]] = self.cli_args.get("model_overrides", [])
        cli_task_overrides: dict[str, list[str]] = self.cli_args.get("task_overrides", {})
        cli_cluster = self.cli_args.get("cluster")
        cli_max_gpus_per_node = self.cli_args.get("max_gpus_per_node")
        cli_pack_models = self.cli_args.get("pack_models")
        cli_priority = self.cli_args.get("priority")
        cli_preemptible = self.cli_args.get("preemptible")
        cli_timeout = self.cli_args.get("timeout")
        cli_retries = self.cli_args.get("retries")
        cli_workspace = self.cli_args.get("workspace")
        cli_budget = self.cli_args.get("budget")
        cli_image = self.cli_args.get("image")
        cli_groups = self.cli_args.get("group", ())

        # Merge config file with CLI args (CLI takes precedence)
        if cfg is not None:
            name = cli_name or cfg.name
            task_specs = list(cli_task) if cli_task else list(cfg.tasks)
            retries = cli_retries if cli_retries is not None else cfg.retries
            workspace = cli_workspace or cfg.workspace
            budget = cli_budget or cfg.budget

            if not cli_model:
                model_configs = cfg.get_model_configs()
                model_specs = [m.name_or_path for m in model_configs]
            else:
                model_specs = list(cli_model)
                model_configs = [
                    parse_model_config(
                        m,
                        overrides=cli_model_overrides[i] if i < len(cli_model_overrides) else [],
                    )
                    for i, m in enumerate(cli_model)
                ]

            cluster = cli_cluster if cli_cluster is not None else cfg.cluster
            max_gpus_per_node = (
                cli_max_gpus_per_node
                if cli_max_gpus_per_node is not None
                else cfg.max_gpus_per_node
            )
            pack_models = cli_pack_models if cli_pack_models is not None else cfg.pack_models
            priority = cli_priority if cli_priority is not None else cfg.priority
            preemptible = cli_preemptible if cli_preemptible is not None else cfg.preemptible
            timeout = cli_timeout if cli_timeout is not None else cfg.timeout
            # CLI runner_type takes precedence over config
            cli_runner_type = self.cli_args.get("runner_type")
            runner_type = (
                RunnerType(cli_runner_type) if cli_runner_type else RunnerType(cfg.runner_type)
            )
            num_workers = (
                self.cli_args.get("num_workers")
                if self.cli_args.get("num_workers") is not None
                else cfg.num_workers
            )
            gpus_per_worker = (
                self.cli_args.get("gpus_per_worker", 1)
                if self.cli_args.get("gpus_per_worker", 1) != 1
                else cfg.gpus_per_worker
            )
            image = cli_image or cfg.beaker_image
        else:
            name = cli_name
            task_specs = list(cli_task)
            retries = cli_retries
            workspace = cli_workspace
            budget = cli_budget
            model_specs = list(cli_model) if cli_model else []
            model_configs = (
                [
                    parse_model_config(
                        m,
                        overrides=cli_model_overrides[i] if i < len(cli_model_overrides) else [],
                    )
                    for i, m in enumerate(cli_model)
                ]
                if cli_model
                else []
            )
            cluster = cli_cluster
            max_gpus_per_node = cli_max_gpus_per_node
            pack_models = cli_pack_models
            priority = cli_priority  # Will default to "normal" if None
            preemptible = cli_preemptible
            timeout = cli_timeout
            cli_runner_type = self.cli_args.get("runner_type")
            runner_type = RunnerType(cli_runner_type) if cli_runner_type else RunnerType.SYNC
            num_workers = self.cli_args.get("num_workers")
            gpus_per_worker = self.cli_args.get("gpus_per_worker", 1)
            image = cli_image

        # Apply defaults
        max_gpus_per_node = (
            max_gpus_per_node if max_gpus_per_node is not None else DEFAULT_MAX_GPUS_PER_NODE
        )
        pack_models = pack_models if pack_models is not None else False
        priority = priority or "normal"
        preemptible = preemptible if preemptible is not None else True
        timeout = timeout or "24h"

        # Validate required fields (raises SystemExit if any are None)
        self._validate_required(name, model_configs, task_specs, cluster, workspace, budget)

        # After validation, these are guaranteed to be non-None
        assert name is not None
        assert cluster is not None
        assert workspace is not None
        assert budget is not None

        # Validate S3 options
        s3_bucket = self.cli_args.get("s3_bucket")
        s3_prefix = self.cli_args.get("s3_prefix")
        self._validate_s3(s3_bucket, s3_prefix, self.cli_args.get("store", False))

        # Build groups list
        effective_groups = list(cli_groups)
        if cfg is not None and cfg.groups:
            for g in cfg.groups:
                if g not in effective_groups:
                    effective_groups.append(g)

        return LaunchConfig(
            name=name,
            model_configs=model_configs,
            model_specs=model_specs,
            task_specs=task_specs,
            cluster=cluster,
            workspace=workspace,
            budget=budget,
            model_overrides=cli_model_overrides,
            task_overrides=cli_task_overrides,
            max_gpus_per_node=max_gpus_per_node,
            pack_models=pack_models,
            priority=priority,
            preemptible=preemptible,
            timeout=timeout,
            retries=retries,
            image=image,
            groups=effective_groups,
            runner_type=runner_type,
            num_workers=num_workers,
            gpus_per_worker=gpus_per_worker,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            s3_endpoint_url=self.cli_args.get("s3_endpoint_url"),
            s3_region=self.cli_args.get("s3_region", "us-east-1"),
            store=self.cli_args.get("store", False),
            debug_requests=self.cli_args.get("debug_requests", False),
            debug_provider=self.cli_args.get("debug_provider", False),
            save_predictions=self.cli_args.get("save_predictions", True),
            save_requests=self.cli_args.get("save_requests", True),
            inspect_instance=self.cli_args.get("inspect_instance", False),
            inspect_formatted=self.cli_args.get("inspect_formatted", False),
            inspect_tokens=self.cli_args.get("inspect_tokens", False),
            inspect_response=self.cli_args.get("inspect_response", False),
            inspect_request=self.cli_args.get("inspect_request", False),
            uv_cache_dir=self.cli_args.get("uv_cache_dir"),
        )

    def _validate_required(
        self,
        name: str | None,
        model_configs: list,
        task_specs: list[str],
        cluster: str | None,
        workspace: str | None,
        budget: str | None,
    ) -> None:
        """Validate required fields."""
        if not name:
            console.print("[red]Error:[/red] --name/-n is required (or set 'name' in config)")
            raise SystemExit(1) from None
        if not model_configs:
            console.print("[red]Error:[/red] --model/-m is required (or set 'models' in config)")
            raise SystemExit(1) from None
        if not task_specs:
            console.print("[red]Error:[/red] --task/-t is required (or set 'tasks' in config)")
            raise SystemExit(1) from None
        if not cluster:
            console.print("[red]Error:[/red] --cluster/-c is required (or set 'cluster' in config)")
            raise SystemExit(1) from None
        if not workspace:
            console.print(
                "[red]Error:[/red] --workspace/-w is required (or set 'workspace' in config)"
            )
            raise SystemExit(1) from None
        if not budget:
            console.print("[red]Error:[/red] --budget/-B is required (or set 'budget' in config)")
            raise SystemExit(1) from None

    def _validate_s3(self, s3_bucket: str | None, s3_prefix: str | None, store: bool) -> None:
        """Validate S3 configuration."""
        if s3_bucket and not s3_prefix:
            console.print("[red]Error:[/red] --s3-prefix is required when --s3-bucket is set")
            raise SystemExit(1) from None
        if s3_prefix and not s3_bucket:
            console.print("[red]Error:[/red] --s3-bucket is required when --s3-prefix is set")
            raise SystemExit(1) from None
        if store and (not s3_bucket or not s3_prefix):
            console.print(
                "[red]Error:[/red] --s3-bucket and --s3-prefix are required when --store is enabled"
            )
            raise SystemExit(1) from None

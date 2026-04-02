"""Configuration loading for Beaker launch command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from olmo_eval.launch import EvalConfig

console = Console()


@dataclass
class LaunchConfig:
    """Parsed and validated configuration for Beaker launch."""

    name: str
    model_specs: list[str]
    task_specs: list[str]
    cluster: str
    workspace: str
    budget: str

    task_overrides: dict[str, list[str]] = field(default_factory=dict)

    max_gpus_per_node: int = 8
    priority: str = "normal"
    preemptible: bool = True
    timeout: str = "6h"
    retries: int | None = None
    image: str | None = None
    groups: list[str] = field(default_factory=list)

    gpus: int = 0

    s3_bucket: str = "ai2-llm"
    s3_prefix: str = "olmo-eval"
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"

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

    harness: str | None = None
    harness_overrides: list[str] = field(default_factory=list)

    uv_cache_dir: str | None = None

    # Secret env overrides: maps beaker_secret_name -> env_var_name
    # Allows overriding default {username}_{ENV_VAR} pattern
    secret_env_overrides: dict[str, str] = field(default_factory=dict)


class LaunchConfigLoader:
    """Loads and merges configuration from YAML file and CLI arguments."""

    def __init__(self, config_path: str | None, cli_args: dict[str, Any]):
        self.config_path = config_path
        self.cli_args = cli_args

    def load(self) -> LaunchConfig:
        """Load and merge configuration."""
        from olmo_eval.common.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE
        from olmo_eval.launch import EvalConfig

        cfg: EvalConfig | None = None

        if self.config_path:
            try:
                cfg = EvalConfig.from_yaml(self.config_path)
            except FileNotFoundError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1) from None
            except Exception as e:
                console.print(f"[red]Config error:[/red] {e}")
                raise SystemExit(1) from None

        cli_name = self.cli_args.get("name")
        cli_model = self.cli_args.get("model", ())
        cli_task = self.cli_args.get("task", ())
        cli_task_overrides: dict[str, list[str]] = self.cli_args.get("task_overrides", {})
        cli_cluster = self.cli_args.get("cluster")
        cli_max_gpus_per_node = self.cli_args.get("max_gpus_per_node")
        cli_priority = self.cli_args.get("priority")
        cli_preemptible = self.cli_args.get("preemptible")
        cli_timeout = self.cli_args.get("timeout")
        cli_retries = self.cli_args.get("retries")
        cli_workspace = self.cli_args.get("workspace")
        cli_budget = self.cli_args.get("budget")
        cli_image = self.cli_args.get("image")
        cli_groups = self.cli_args.get("group", ())
        cli_gpus = self.cli_args.get("gpus")

        if cfg is not None:
            name = cli_name or cfg.name
            task_specs = list(cli_task) if cli_task else list(cfg.tasks)
            retries = cli_retries if cli_retries is not None else cfg.retries
            workspace = cli_workspace or cfg.workspace
            budget = cli_budget or cfg.budget
            model_specs = list(cli_model) if cli_model else list(cfg.models)
            cluster = cli_cluster if cli_cluster is not None else cfg.cluster
            max_gpus_per_node = (
                cli_max_gpus_per_node
                if cli_max_gpus_per_node is not None
                else cfg.max_gpus_per_node
            )
            priority = cli_priority if cli_priority is not None else cfg.priority
            preemptible = cli_preemptible if cli_preemptible is not None else cfg.preemptible
            timeout = cli_timeout if cli_timeout is not None else cfg.timeout
            gpus = cli_gpus if cli_gpus is not None else cfg.gpus
            image = cli_image or cfg.beaker_image
        else:
            name = cli_name
            task_specs = list(cli_task)
            retries = cli_retries
            workspace = cli_workspace
            budget = cli_budget
            model_specs = list(cli_model) if cli_model else []
            cluster = cli_cluster
            max_gpus_per_node = cli_max_gpus_per_node
            priority = cli_priority
            preemptible = cli_preemptible
            timeout = cli_timeout
            gpus = cli_gpus  # None means auto-detect from provider
            image = cli_image

        # Auto-detect GPU requirements from provider if not explicitly set
        if gpus is None and model_specs:
            gpus = self._detect_gpu_requirement(model_specs[0])
        elif gpus is None:
            gpus = 0  # Default to 0 if no model specified

        max_gpus_per_node = (
            max_gpus_per_node if max_gpus_per_node is not None else DEFAULT_MAX_GPUS_PER_NODE
        )
        priority = priority or "normal"
        preemptible = preemptible if preemptible is not None else True
        timeout = timeout or "24h"

        self._validate_required(model_specs, task_specs, cluster, workspace, budget)

        # Auto-generate name if not provided
        if not name:
            name = self._generate_experiment_name(model_specs, task_specs)

        assert name is not None
        assert cluster is not None
        assert workspace is not None
        assert budget is not None

        from olmo_eval.launch.beaker.constants import DEFAULT_S3_BUCKET, DEFAULT_S3_PREFIX

        s3_bucket = self.cli_args.get("s3_bucket") or DEFAULT_S3_BUCKET
        s3_prefix = self.cli_args.get("s3_prefix") or DEFAULT_S3_PREFIX

        effective_groups = list(cli_groups)
        if cfg is not None and cfg.groups:
            for g in cfg.groups:
                if g not in effective_groups:
                    effective_groups.append(g)

        return LaunchConfig(
            name=name,
            model_specs=model_specs,
            task_specs=task_specs,
            cluster=cluster,
            workspace=workspace,
            budget=budget,
            task_overrides=cli_task_overrides,
            max_gpus_per_node=max_gpus_per_node,
            priority=priority,
            preemptible=preemptible,
            timeout=timeout,
            retries=retries,
            image=image,
            groups=effective_groups,
            gpus=gpus,
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
            harness=self.cli_args.get("harness"),
            harness_overrides=self.cli_args.get("harness_overrides", []),
            uv_cache_dir=self.cli_args.get("uv_cache_dir"),
            secret_env_overrides=self.cli_args.get("secret_env_overrides", {}),
        )

    def _validate_required(
        self,
        model_specs: list[str],
        task_specs: list[str],
        cluster: str | None,
        workspace: str | None,
        budget: str | None,
    ) -> None:
        if not model_specs:
            console.print("[red]Error:[/red] --model/-m is required")
            raise SystemExit(1) from None
        if not task_specs:
            console.print("[red]Error:[/red] --task/-t is required")
            raise SystemExit(1) from None
        if not cluster:
            console.print("[red]Error:[/red] --cluster/-c is required")
            raise SystemExit(1) from None
        if not workspace:
            console.print("[red]Error:[/red] --workspace/-w is required")
            raise SystemExit(1) from None
        if not budget:
            console.print("[red]Error:[/red] --budget/-B is required")
            raise SystemExit(1) from None

    def _generate_experiment_name(self, model_specs: list[str], task_specs: list[str]) -> str:
        """Generate experiment name from model and task specs.

        Format:
        - 1-2 tasks: {model}-{task1}-{task2}
        - 3+ tasks: {model}-{task1}-and-{N}-more
        """
        from olmo_eval.launch import get_model_short_name, sanitize_beaker_name

        # Use first model for the name (multi-model runs append model name later)
        model_name = get_model_short_name(model_specs[0]) if model_specs else "eval"

        if len(task_specs) <= 2:
            tasks_part = "-".join(task_specs)
        else:
            tasks_part = f"{task_specs[0]}-and-{len(task_specs) - 1}-more"

        return sanitize_beaker_name(f"{model_name}-{tasks_part}")

    def _detect_gpu_requirement(self, model_spec: str) -> int:
        """Detect GPU requirement based on provider type.

        Args:
            model_spec: Model specification (name or preset).

        Returns:
            1 if provider requires GPU (vllm, hf), 0 otherwise (litellm, mock).
        """
        from olmo_eval.common.configs import get_provider_config

        try:
            provider_config = get_provider_config(model_spec)
            return 1 if provider_config.requires_gpu else 0
        except Exception:
            # If we can't determine, default to 1 GPU for safety
            return 1

"""Configuration building for the run command."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from olmo_eval.harness.config import HarnessConfig, ProviderConfig

console = Console()


@dataclass
class RunConfig:
    """Parsed and validated configuration for an evaluation run.

    This is the fully-formed configuration after CLI parsing and override
    application.
    """

    harness_config: HarnessConfig
    task_specs: list[str] = field(default_factory=list, repr=False)
    # Per-task overrides (task_spec -> overrides dict)
    # These are applied when preparing tasks since tasks are loaded by spec
    task_overrides: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    output_dir: str = "/tmp/results/"

    # Storage configuration
    store: bool = False
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    s3_group: str | None = None
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "olmo_eval"
    db_user: str = "postgres"
    db_password: str = field(default="postgres", repr=False)

    # Experiment metadata
    experiment_name: str | None = None
    experiment_group: str | None = None

    # Output options
    save_predictions: bool = True
    save_requests: bool = True

    # Debug/inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    @property
    def model_name(self) -> str:
        """Get the model name from the harness config."""
        return self.harness_config.provider.model

    @property
    def provider_config(self) -> ProviderConfig:
        """Get the provider config from the harness config."""
        return self.harness_config.provider

    @property
    def tasks(self) -> list:
        """Resolve task_specs to TaskConfig objects for display."""
        from olmo_eval.common.configs import expand_tasks
        from olmo_eval.evals.tasks.common import get_task

        return [get_task(spec).config for spec in expand_tasks(self.task_specs)]

    def __rich_repr__(self):
        """Rich repr that shows resolved tasks instead of task_specs."""
        from dataclasses import fields

        for f in fields(self):
            if f.name == "task_specs":
                yield "tasks", self.tasks
            elif not f.repr:
                continue
            else:
                yield f.name, getattr(self, f.name)


class RunConfigBuilder:
    """Builds and validates run configuration from CLI arguments."""

    def __init__(
        self,
        model: str,
        task: tuple[str, ...],
        output_dir: str,
        num_gpus: int = 1,
        parallelism: int = 1,
        store: bool = False,
        s3_bucket: str | None = None,
        s3_prefix: str | None = None,
        s3_group: str | None = None,
        s3_endpoint_url: str | None = None,
        s3_region: str = "us-east-1",
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "olmo_eval",
        db_user: str = "postgres",
        db_password: str = "postgres",
        experiment_name: str | None = None,
        experiment_group: str | None = None,
        save_predictions: bool = True,
        save_requests: bool = True,
        inspect_instance: bool = False,
        inspect_formatted: bool = False,
        inspect_tokens: bool = False,
        inspect_response: bool = False,
        inspect_request: bool = False,
        cli_task_overrides: dict[str, list[str]] | None = None,
        harness_preset: str | None = None,
        harness_config_path: str | None = None,
        cli_harness_overrides: list[str] | None = None,
    ):
        """Initialize the builder with raw CLI arguments.

        Args:
            model: Model name/path from -m flag.
            task: Tuple of task specs from -t flags.
            output_dir: Output directory for results.
            cli_task_overrides: Per-task overrides from -o flags (task_spec -> [overrides]).
            harness_preset: Name of a harness preset (e.g., "search").
            harness_config_path: Path to a harness config YAML/JSON file.
            cli_harness_overrides: Harness overrides from -o flags after --harness.
            ... (other standard args)
        """
        self.model = model
        self.task = task
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.parallelism = parallelism
        self.store = store
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_group = s3_group
        self.s3_endpoint_url = s3_endpoint_url
        self.s3_region = s3_region
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.experiment_name = experiment_name
        self.experiment_group = experiment_group
        self.save_predictions = save_predictions
        self.save_requests = save_requests
        self.inspect_instance = inspect_instance
        self.inspect_formatted = inspect_formatted
        self.inspect_tokens = inspect_tokens
        self.inspect_response = inspect_response
        self.inspect_request = inspect_request
        self.cli_task_overrides = cli_task_overrides or {}
        self.harness_preset = harness_preset
        self.harness_config_path = harness_config_path
        self.cli_harness_overrides = cli_harness_overrides or []

    def build(self) -> RunConfig:
        """Parse inputs and build configuration.

        Returns:
            RunConfig with parsed and validated settings.
        """
        from omegaconf import OmegaConf

        # Build task specs and overrides from CLI -o flags
        task_specs: list[str] = list(self.task)
        task_overrides: dict[str, dict[str, Any]] = {}
        for task_spec, cli_overrides in self.cli_task_overrides.items():
            if cli_overrides:
                override_config = OmegaConf.from_dotlist(cli_overrides)
                override_dict = OmegaConf.to_container(override_config)
                task_overrides[task_spec] = override_dict  # type: ignore[assignment]

        # Resolve harness configuration with provider config built in
        harness_config = self._resolve_harness_config(self.model)

        return RunConfig(
            harness_config=harness_config,
            task_specs=task_specs,
            task_overrides=task_overrides,
            output_dir=self.output_dir,
            store=self.store,
            s3_bucket=self.s3_bucket,
            s3_prefix=self.s3_prefix,
            s3_group=self.s3_group,
            s3_endpoint_url=self.s3_endpoint_url,
            s3_region=self.s3_region,
            db_host=self.db_host,
            db_port=self.db_port,
            db_name=self.db_name,
            db_user=self.db_user,
            db_password=self.db_password,
            experiment_name=self.experiment_name,
            experiment_group=self.experiment_group,
            save_predictions=self.save_predictions,
            save_requests=self.save_requests,
            inspect_instance=self.inspect_instance,
            inspect_formatted=self.inspect_formatted,
            inspect_tokens=self.inspect_tokens,
            inspect_response=self.inspect_response,
            inspect_request=self.inspect_request,
        )

    def _resolve_harness_config(self, model_name: str) -> HarnessConfig:
        """Resolve harness configuration with provider fully configured.

        Args:
            model_name: Model name/path.

        Returns:
            HarnessConfig with provider configured.

        Raises:
            SystemExit: If harness preset or config file is invalid.
        """
        from omegaconf import OmegaConf

        if self.harness_preset and self.harness_config_path:
            console.print("[red]Error:[/red] Cannot specify both --harness and --harness-config")
            raise SystemExit(1)

        harness_config: HarnessConfig

        if self.harness_preset:
            try:
                from olmo_eval.harness import get_harness_preset

                harness_config = get_harness_preset(self.harness_preset)
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1) from None

        elif self.harness_config_path:
            import json

            import yaml

            try:
                with open(self.harness_config_path) as f:
                    if self.harness_config_path.endswith(".json"):
                        config_dict = json.load(f)
                    else:
                        config_dict = yaml.safe_load(f)

                harness_config = HarnessConfig.from_dict(config_dict)
                console.print(f"[dim]Using harness config: {self.harness_config_path}[/dim]")
            except FileNotFoundError:
                console.print(
                    f"[red]Error:[/red] Harness config file not found: {self.harness_config_path}"
                )
                raise SystemExit(1) from None
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                console.print(f"[red]Error:[/red] Invalid harness config file: {e}")
                raise SystemExit(1) from None
            except (KeyError, TypeError) as e:
                console.print(f"[red]Error:[/red] Invalid harness config format: {e}")
                raise SystemExit(1) from None

        else:
            harness_config = HarnessConfig(name="default")

        # Apply CLI overrides to harness config
        if self.cli_harness_overrides:
            harness_dict = harness_config.to_dict()
            override_config = OmegaConf.from_dotlist(self.cli_harness_overrides)
            base = OmegaConf.create(harness_dict)
            merged = OmegaConf.merge(base, override_config)
            harness_dict = OmegaConf.to_container(merged, resolve=True)
            harness_config = HarnessConfig.from_dict(harness_dict)  # type: ignore[arg-type]

        from olmo_eval.common.configs import get_provider_config

        provider_config = get_provider_config(model_name)
        harness_config = harness_config.merge_provider(provider_config)

        return harness_config

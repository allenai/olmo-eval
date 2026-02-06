"""Configuration building for the run command."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console

console = Console()


@dataclass
class RunConfig:
    """Parsed and validated configuration for an evaluation run."""

    # Model configuration
    model_names: list[str]
    per_model_overrides: dict[str, dict[str, Any]]

    # Task configuration
    task_specs: list[str]
    task_overrides: dict[str, dict[str, Any]]

    # Runner configuration
    output_dir: str
    provider: str | None = None
    attention_backend: str | None = None
    num_workers: int | None = None
    gpus_per_worker: int = 1
    num_gpus: int = 1
    parallelism: int = 1

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
    db_password: str = "postgres"

    # Experiment identification
    experiment_name: str | None = None
    experiment_group: str | None = None
    alias: str | None = None

    # Output options
    save_predictions: bool = True
    save_requests: bool = True

    # Debug/inspection options
    inspect_instance: bool = False
    inspect_formatted: bool = False
    inspect_tokens: bool = False
    inspect_response: bool = False
    inspect_request: bool = False

    # Harness configuration for tool/prompt injection
    harness_config: dict[str, Any] | None = None


class RunConfigBuilder:
    """Builds and validates run configuration from CLI arguments."""

    def __init__(
        self,
        models: tuple[str, ...],
        task: tuple[str, ...],
        output_dir: str,
        provider: str | None = None,
        attention_backend: str | None = None,
        num_workers: int | None = None,
        gpus_per_worker: int = 1,
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
        alias: str | None = None,
        save_predictions: bool = True,
        save_requests: bool = True,
        inspect_instance: bool = False,
        inspect_formatted: bool = False,
        inspect_tokens: bool = False,
        inspect_response: bool = False,
        inspect_request: bool = False,
        cli_model_overrides: list[list[str]] | None = None,
        cli_task_overrides: dict[str, list[str]] | None = None,
        harness_preset: str | None = None,
        harness_config_path: str | None = None,
    ):
        """Initialize the builder with raw CLI arguments.

        Args:
            models: Tuple of model names/paths from -m flags.
            task: Tuple of task specs from -t flags.
            output_dir: Output directory for results.
            cli_model_overrides: Per-model overrides from -o flags (positional list).
            cli_task_overrides: Per-task overrides from -o flags (task_spec -> [overrides]).
            harness_preset: Name of a harness preset (e.g., "search").
            harness_config_path: Path to a harness config YAML/JSON file.
            ... (other standard args)
        """
        self.models = models
        self.task = task
        self.output_dir = output_dir
        self.provider = provider
        self.attention_backend = attention_backend
        self.num_workers = num_workers
        self.gpus_per_worker = gpus_per_worker
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
        self.alias = alias
        self.save_predictions = save_predictions
        self.save_requests = save_requests
        self.inspect_instance = inspect_instance
        self.inspect_formatted = inspect_formatted
        self.inspect_tokens = inspect_tokens
        self.inspect_response = inspect_response
        self.inspect_request = inspect_request
        self.cli_model_overrides = cli_model_overrides or []
        self.cli_task_overrides = cli_task_overrides or {}
        self.harness_preset = harness_preset
        self.harness_config_path = harness_config_path

    def build(self) -> RunConfig:
        """Parse inputs and build configuration.

        Returns:
            RunConfig with parsed and validated settings.
        """
        from omegaconf import OmegaConf

        from olmo_eval.cli.utils import parse_model_spec, parse_task_spec_with_overrides

        # Parse model specs to extract overrides
        parsed_models: list[tuple[str, dict[str, Any]]] = [parse_model_spec(m) for m in self.models]

        # Parse task specs to extract overrides
        task_overrides: dict[str, dict[str, Any]] = {}
        task_specs: list[str] = []
        for t in self.task:
            spec_without_overrides, overrides = parse_task_spec_with_overrides(t)
            task_specs.append(spec_without_overrides)
            if overrides:
                task_overrides[spec_without_overrides] = overrides

        # Extract model names
        model_names = [name for name, _overrides in parsed_models]

        # Build per-model overrides from CLI -o flags (positional)
        per_model_overrides: dict[str, dict[str, Any]] = {}
        for i, cli_overrides in enumerate(self.cli_model_overrides):
            if cli_overrides and i < len(model_names):
                model_name = model_names[i]
                override_config = OmegaConf.from_dotlist(cli_overrides)
                per_model_overrides[model_name] = OmegaConf.to_container(override_config)  # type: ignore[assignment]

        # Build task overrides from CLI -o flags
        for task_spec, cli_overrides in self.cli_task_overrides.items():
            if cli_overrides:
                override_config = OmegaConf.from_dotlist(cli_overrides)
                override_dict = OmegaConf.to_container(override_config)
                if task_spec in task_overrides:
                    task_overrides[task_spec].update(override_dict)  # type: ignore[arg-type]
                else:
                    task_overrides[task_spec] = override_dict  # type: ignore[assignment]

        # Apply first model's provider/attention_backend as defaults if not specified globally
        provider = self.provider
        attention_backend = self.attention_backend
        if model_names:
            first_overrides = per_model_overrides.get(model_names[0], {})
            if not provider and "provider" in first_overrides:
                provider = first_overrides["provider"]
                if isinstance(provider, dict):
                    provider = provider.get("kind")
            if not attention_backend and "attention_backend" in first_overrides:
                attention_backend = first_overrides["attention_backend"]

        # Resolve harness configuration from preset or file
        harness_config = self._resolve_harness_config()

        return RunConfig(
            model_names=model_names,
            per_model_overrides=per_model_overrides,
            task_specs=task_specs,
            task_overrides=task_overrides,
            output_dir=self.output_dir,
            provider=provider,
            attention_backend=attention_backend,
            num_workers=self.num_workers,
            gpus_per_worker=self.gpus_per_worker,
            num_gpus=self.num_gpus,
            parallelism=self.parallelism,
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
            alias=self.alias,
            save_predictions=self.save_predictions,
            save_requests=self.save_requests,
            inspect_instance=self.inspect_instance,
            inspect_formatted=self.inspect_formatted,
            inspect_tokens=self.inspect_tokens,
            inspect_response=self.inspect_response,
            inspect_request=self.inspect_request,
            harness_config=harness_config,
        )

    def _resolve_harness_config(self) -> dict[str, Any] | None:
        """Resolve harness configuration from preset name or config file.

        Returns:
            Serialized harness config dict, or None if no harness specified.

        Raises:
            SystemExit: If harness preset or config file is invalid.
        """
        if self.harness_preset and self.harness_config_path:
            console.print("[red]Error:[/red] Cannot specify both --harness and --harness-config")
            raise SystemExit(1)

        if self.harness_preset:
            try:
                from olmo_eval.core.harness import get_harness_preset

                config = get_harness_preset(self.harness_preset)
                console.print(f"[dim]Using harness preset: {self.harness_preset}[/dim]")
                return config.to_dict()
            except ValueError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise SystemExit(1) from None

        if self.harness_config_path:
            import json

            import yaml

            try:
                with open(self.harness_config_path) as f:
                    if self.harness_config_path.endswith(".json"):
                        config_dict = json.load(f)
                    else:
                        config_dict = yaml.safe_load(f)

                # Validate the config can be loaded
                from olmo_eval.core.harness import HarnessConfig

                config = HarnessConfig.from_dict(config_dict)
                console.print(f"[dim]Using harness config: {self.harness_config_path}[/dim]")
                return config.to_dict()
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

        return None

    def validate_flags(self) -> bool:
        """Validate CLI flag combinations and print warnings.

        Returns:
            True if validation passes.
        """
        return True

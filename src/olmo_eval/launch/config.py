"""YAML-based configuration for Beaker launch jobs.

Provides OmegaConf-based configuration loading for composing complex
evaluation experiments from YAML files.

Example config (eval_config.yaml):
    name: eval-llama-suite
    models:
      - name_or_path: llama3.1-8b
        gpus: 1
      - name_or_path: olmo-2-7b
        gpus: 1
    tasks:
      - mmlu
      - gsm8k
      - hellaswag
    cluster: h100
    priority: normal

Example with per-model resources:
    name: eval-mixed-sizes
    models:
      - name_or_path: llama3.1-8b
        gpus: 1
      - name_or_path: llama3.1-70b
        gpus: 4
        timeout: 48h
    tasks:
      - mmlu@high
      - gsm8k@normal
    cluster: h100

Example usage:
    config = EvalConfig.from_yaml("eval_config.yaml")
    # Or with CLI overrides:
    config = EvalConfig.from_yaml("eval_config.yaml", overrides=["priority=high"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from olmo_eval.core.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE
from olmo_eval.core.types import RunnerType


@dataclass
class ProviderConfig:
    """Configuration for the inference provider to install at runtime.

    Attributes:
        name: Provider name ("vllm", "hf", "litellm"). Maps to pyproject.toml extras.
        package: Optional custom package specifier (URL, path, or PyPI version).
                 If set, installed after base extras to override default version.
    """

    name: str = "vllm"
    package: str | None = None


@dataclass
class BeakerModelSpec:
    """Specification for a single model with resource settings for Beaker launch.

    This allows specifying per-model resources for mixed-size evaluations.

    Attributes:
        name_or_path: Model name, HuggingFace path, or local checkpoint path (required).
        alias: Optional short name for experiment naming. If not set, a short name
            is derived from name_or_path (last path component, or last 16 chars for long paths).
        gpus: Number of GPUs per model instance. Defaults to 1.
        parallelism: Number of model instances to run in parallel. Total GPUs
            requested will be gpus × parallelism. Defaults to 1.
        cluster: Cluster for this model (overrides default).
        preemptible: Whether this model's jobs can be preempted.
        timeout: Timeout for this model's jobs.
        shared_memory: Shared memory size (e.g., "10GiB").
        runner_type: Runner type (sync, async, async-stream, agent). Overrides default.
        num_workers: Number of workers for async modes (overrides default).
        gpus_per_worker: GPUs per worker for async modes (overrides default).
        provider: Inference provider configuration (ProviderConfig with name and optional package).
        load_format: vLLM model loading format (e.g., "runai_streamer" for distributed loading).
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true}).

    Example:
        models:
          - name_or_path: llama3.1-8b
            gpus: 1
            parallelism: 4  # 4 instances × 1 GPU = 4 GPUs total
            provider:
              name: vllm
              package: vllm==0.14.0  # Custom version
          - name_or_path: /weka/checkpoints/my-model/step1000-hf
            alias: my-model-1k  # Short name for experiment naming
            gpus: 4
            parallelism: 2  # 2 instances × 4 GPUs = 8 GPUs total
            provider:
              name: vllm
              package: https://github.com/davidheineman/vllm@my-branch
            runner_type: async
            num_workers: 2
            gpus_per_worker: 4
            timeout: 48h
          - name_or_path: llama3.1-70b
            gpus: 4
            provider:
              name: vllm
            load_format: runai_streamer  # Use distributed streaming loader
            extra_loader_config:
              distributed: true
              concurrency: 16
    """

    name_or_path: str = MISSING
    alias: str | None = None
    gpus: int = 1
    parallelism: int = 1
    cluster: str | None = None
    preemptible: bool | None = None
    timeout: str | None = None
    shared_memory: str | None = None

    # Runner type (sync, async, async-stream, agent)
    runner_type: str | None = None  # String for OmegaConf compatibility
    num_workers: int | None = None
    gpus_per_worker: int | None = None

    # Runtime inference provider installation
    provider: ProviderConfig | None = None

    # vLLM model loading configuration
    load_format: str | None = None
    extra_loader_config: dict[str, Any] | None = None  # e.g., {"distributed": true}


def apply_overrides_to_model(name_or_path: str, overrides: list[str]) -> BeakerModelSpec:
    """Create BeakerModelSpec with OmegaConf overrides.

    This is the preferred way to apply overrides from the -o CLI flag.

    Args:
        name_or_path: Model name or path.
        overrides: List of override strings in dotlist format (e.g., ["provider.name=vllm"]).

    Returns:
        BeakerModelSpec instance with overrides applied.

    Examples:
        >>> apply_overrides_to_model("llama3.1-8b", ["provider.name=vllm", "gpus=4"])
        BeakerModelSpec(name_or_path='llama3.1-8b', provider=ProviderConfig(name='vllm'), gpus=4)
    """
    config_dict: dict[str, Any] = {"name_or_path": name_or_path}

    if overrides:
        # Direct to OmegaConf - no custom parsing!
        override_config = OmegaConf.from_dotlist(overrides)
        override_dict = OmegaConf.to_container(override_config)
        config_dict.update(override_dict)  # type: ignore[arg-type]

    schema = OmegaConf.structured(BeakerModelSpec)
    merged = OmegaConf.merge(schema, OmegaConf.create(config_dict))
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def parse_model_config(
    model: str | dict[str, Any] | BeakerModelSpec,
    overrides: list[str] | None = None,
) -> BeakerModelSpec:
    """Parse a model specification into BeakerModelSpec.

    Handles simple string format, dict format, or existing BeakerModelSpec.
    Use the `overrides` parameter to apply CLI overrides from the -o flag.

    Args:
        model: Model name/path string, dict with model config, or BeakerModelSpec.
        overrides: Optional list of override strings in dotlist format
            (e.g., ["provider.name=vllm"]).

    Returns:
        BeakerModelSpec instance with any overrides applied.

    Examples:
        parse_model_config("llama3.1-8b")
        parse_model_config("llama3.1-8b", overrides=["provider.name=vllm", "gpus=4"])
        parse_model_config({"name_or_path": "llama3.1-70b", "gpus": 4})
    """
    if isinstance(model, BeakerModelSpec):
        if overrides:
            config_dict = OmegaConf.to_container(OmegaConf.structured(model))
            override_config = OmegaConf.from_dotlist(overrides)
            override_dict = OmegaConf.to_container(override_config)
            config_dict.update(override_dict)  # type: ignore[union-attr]
            schema = OmegaConf.structured(BeakerModelSpec)
            merged = OmegaConf.merge(schema, OmegaConf.create(config_dict))
            return OmegaConf.to_object(merged)  # type: ignore[return-value]
        return model

    if isinstance(model, str):
        config_dict: dict[str, Any] = {"name_or_path": model}
        if overrides:
            override_config = OmegaConf.from_dotlist(overrides)
            override_dict = OmegaConf.to_container(override_config)
            config_dict.update(override_dict)  # type: ignore[arg-type]
        schema = OmegaConf.structured(BeakerModelSpec)
        merged = OmegaConf.merge(schema, OmegaConf.create(config_dict))
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    if isinstance(model, dict):
        config_dict = dict(model)
        if overrides:
            override_config = OmegaConf.from_dotlist(overrides)
            override_dict = OmegaConf.to_container(override_config)
            config_dict.update(override_dict)  # type: ignore[arg-type]
        schema = OmegaConf.structured(BeakerModelSpec)
        merged = OmegaConf.merge(schema, OmegaConf.create(config_dict))
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    raise TypeError(f"Invalid model specification: {type(model)}")


def get_model_short_name(model: BeakerModelSpec) -> str:
    """Get a short display name for a model suitable for experiment naming.

    If alias is set, returns the alias. Otherwise derives a short name from
    name_or_path by taking the last path component. If the result is empty
    or longer than 32 characters, takes the last 16 characters of the path.

    Args:
        model: BeakerModelSpec instance.

    Returns:
        Short name suitable for use in experiment names.

    Examples:
        >>> get_model_short_name(BeakerModelSpec(name_or_path="llama3.1-8b"))
        'llama3.1-8b'
        >>> get_model_short_name(BeakerModelSpec(name_or_path="meta-llama/Llama-3.1-8B"))
        'llama-3.1-8b'
        >>> get_model_short_name(
        ...     BeakerModelSpec(name_or_path="/weka/checkpoints/model/step1000-hf/")
        ... )
        'step1000-hf'
        >>> get_model_short_name(
        ...     BeakerModelSpec(name_or_path="/weka/checkpoints/model/", alias="my-model")
        ... )
        'my-model'
    """
    # Use alias if provided
    if model.alias:
        return model.alias.lower()

    name = model.name_or_path

    # Strip trailing slashes and get the last path component
    name = name.rstrip("/")
    short_name = name.split("/")[-1]

    # If empty or too long, use last 16 chars of the full path
    if not short_name or len(short_name) > 32:
        # Clean up the name for use in experiment names
        short_name = name[-16:].lstrip("/").lstrip("-").lstrip("_")

    return short_name.lower()


def _shorten_task_name(task: str, max_len: int = 8) -> str:
    """Shorten a single task name for use in experiment naming.

    Removes common suffixes and prefixes, strips variants (after ':'),
    and truncates to max_len characters.
    """
    # Strip @priority suffix if present
    if "@" in task:
        task = task.split("@")[0]

    # Strip variant suffix (e.g., "arc:mc" -> "arc")
    if ":" in task:
        task = task.split(":")[0]

    # Remove common suffixes/prefixes
    task = task.replace("_challenge", "").replace("_easy", "")

    # Truncate if needed
    if len(task) > max_len:
        task = task[:max_len]

    return task.lower().rstrip("_").rstrip("-")


def get_tasks_short_name(tasks: list[str], max_total_len: int = 24) -> str:
    """Generate a short identifier from a list of task names.

    Creates a concise name suitable for experiment naming:
    - Single task: uses the task name (shortened)
    - 2-3 tasks: joins abbreviated names with '_'
    - 4+ tasks: uses first task name + count (e.g., "mmlu_3more")

    Args:
        tasks: List of task names (may include @priority or :variant suffixes).
        max_total_len: Maximum length of the returned string.

    Returns:
        Short identifier string for the task list.

    Examples:
        >>> get_tasks_short_name(["mmlu"])
        'mmlu'
        >>> get_tasks_short_name(["gsm8k", "arc_challenge"])
        'gsm8k_arc'
        >>> get_tasks_short_name(["mmlu", "gsm8k", "hellaswag", "arc_challenge"])
        'mmlu_3more'
    """
    if not tasks:
        return "notasks"

    if len(tasks) == 1:
        return _shorten_task_name(tasks[0], max_len=max_total_len)

    if len(tasks) <= 3:
        # Join abbreviated names
        shortened = [_shorten_task_name(t, max_len=8) for t in tasks]
        result = "_".join(shortened)
        if len(result) > max_total_len:
            # Fall back to first task + count
            first = _shorten_task_name(tasks[0], max_len=12)
            return f"{first}_{len(tasks) - 1}more"
        return result

    # 4+ tasks: first task + count
    first = _shorten_task_name(tasks[0], max_len=12)
    return f"{first}_{len(tasks) - 1}more"


@dataclass
class EvalConfig:
    """Configuration for launching Beaker evaluation jobs.

    This dataclass can be loaded from YAML files using OmegaConf,
    allowing for complex configuration composition and overrides.

    Models can be specified as simple strings or with per-model resource overrides:

        # Simple format (uses default gpus=1, parallelism=1)
        models:
          - llama3.1-8b
          - olmo-2-7b

        # Per-model resources
        models:
          - name_or_path: llama3.1-8b
            gpus: 1
          - name_or_path: llama3.1-70b
            gpus: 4
            timeout: 48h

    Attributes:
        name: Experiment name (required).
        models: List of model names/paths or BeakerModelSpec dicts (required).
            Each model can specify its own inference provider via the 'provider' field.
            GPU and parallelism settings are per-model (default 1 each).
        tasks: List of task specs, optionally with @priority suffix (required).
        cluster: Default cluster alias or full name.
        max_gpus_per_node: Maximum GPUs available per node. When total GPUs
            exceeds this, models are split across experiments.
        priority: Default job priority for tasks without @priority suffix.
        preemptible: Default preemption setting.
        timeout: Default job timeout (e.g., "24h", "48h").
        retries: Number of retries on failure.
        workspace: Beaker workspace.
        budget: Beaker budget.
        beaker_image: Container image to use.
        description: Optional experiment description.
        groups: List of Beaker groups to add experiments to.
        runner_type: Default runner type (sync, async, async-stream, agent).
        num_workers: Number of workers for async modes.
        gpus_per_worker: GPUs per worker for async modes.
        pack_models: Pack multiple models into single experiments when they fit.
            Default False: each model runs in its own experiment for easier scheduling.
    """

    # Required fields
    name: str = MISSING
    models: list[Any] = MISSING  # list[str] or list[dict] for BeakerModelSpec
    tasks: list[str] = MISSING

    # Default cluster and resources
    cluster: str | None = None
    max_gpus_per_node: int = DEFAULT_MAX_GPUS_PER_NODE
    pack_models: bool = False

    # Default job settings (can be overridden per-model)
    priority: str = "normal"
    preemptible: bool = True
    timeout: str = "24h"
    retries: int | None = None

    # Runner type and worker settings
    runner_type: str = RunnerType.SYNC.value  # String for OmegaConf compatibility
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # Beaker settings
    workspace: str | None = None
    budget: str | None = None
    beaker_image: str | None = None
    description: str | None = None
    groups: list[str] | None = None  # Groups to add experiments to

    def get_model_configs(self) -> list[BeakerModelSpec]:
        """Get parsed BeakerModelSpec objects for all models.

        Returns a list of BeakerModelSpec objects, parsing simple strings
        into BeakerModelSpec with just the name set.

        Returns:
            List of BeakerModelSpec objects.
        """
        return [parse_model_config(m) for m in self.models]

    def get_model_resources(self, model: BeakerModelSpec) -> dict[str, Any]:
        """Get effective resources for a model, merging defaults with overrides.

        Args:
            model: BeakerModelSpec with resource settings.

        Returns:
            Dict with effective resource values including:
            - gpus: GPUs per model instance (not multiplied by parallelism)
            - parallelism: Number of model instances to run
            - Other resource settings (cluster, timeout, etc.)
        """
        # Determine runner type (model overrides config default)
        runner_type_str = model.runner_type if model.runner_type is not None else self.runner_type
        runner_type = RunnerType(runner_type_str)

        num_workers = model.num_workers if model.num_workers is not None else self.num_workers
        gpus_per_worker = (
            model.gpus_per_worker if model.gpus_per_worker is not None else self.gpus_per_worker
        )

        # Calculate GPUs per model instance
        # For async modes, GPUs are calculated from workers
        if runner_type in (RunnerType.ASYNC, RunnerType.ASYNC_STREAM) and num_workers is not None:
            gpus_per_model = num_workers * gpus_per_worker
        else:
            gpus_per_model = model.gpus

        # Parallelism is always per-model (default 1)
        parallelism = model.parallelism

        # Extract provider name and package from ProviderConfig
        provider_name = model.provider.name if model.provider else None
        provider_package = model.provider.package if model.provider else None

        return {
            "gpus": gpus_per_model,
            "parallelism": parallelism,
            "cluster": model.cluster if model.cluster is not None else self.cluster,
            "preemptible": model.preemptible if model.preemptible is not None else self.preemptible,
            "timeout": model.timeout if model.timeout is not None else self.timeout,
            "shared_memory": model.shared_memory,  # None uses BeakerJobConfig default
            "runner_type": runner_type,
            "num_workers": num_workers,
            "gpus_per_worker": gpus_per_worker,
            "provider": provider_name,
            "provider_package": provider_package,
            "load_format": model.load_format,
            "extra_loader_config": model.extra_loader_config,
        }

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        overrides: list[str] | None = None,
    ) -> EvalConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.
            overrides: Optional list of dotlist overrides (e.g., ["priority=high"]).

        Returns:
            EvalConfig instance with merged configuration.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If deprecated top-level gpus/parallelism fields are used.
            omegaconf.errors.MissingMandatoryValue: If required fields are missing.

        Example:
            # Load basic config
            config = EvalConfig.from_yaml("eval_config.yaml")

            # Load with overrides
            config = EvalConfig.from_yaml(
                "eval_config.yaml",
                overrides=["priority=high", "cluster=a100"]
            )
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Load YAML file
        file_config = OmegaConf.load(path)

        # Check for deprecated top-level gpus/parallelism fields
        file_dict = OmegaConf.to_container(file_config) if file_config else {}
        if isinstance(file_dict, dict):
            if "gpus" in file_dict:
                raise ValueError(
                    "Top-level 'gpus' is no longer supported. "
                    "Specify gpus per-model instead:\n\n"
                    "  models:\n"
                    "    - name_or_path: your-model\n"
                    "      gpus: 4\n"
                )
            if "parallelism" in file_dict:
                raise ValueError(
                    "Top-level 'parallelism' is no longer supported. "
                    "Specify parallelism per-model instead:\n\n"
                    "  models:\n"
                    "    - name_or_path: your-model\n"
                    "      parallelism: 4\n"
                )

        # Create structured config with defaults
        schema = OmegaConf.structured(cls)

        # Merge: schema defaults <- file config
        merged = OmegaConf.merge(schema, file_config)

        # Apply CLI overrides if provided
        if overrides:
            override_config = OmegaConf.from_dotlist(overrides)
            merged = OmegaConf.merge(merged, override_config)

        # Convert to dataclass instance
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            EvalConfig instance.
        """
        schema = OmegaConf.structured(cls)
        merged = OmegaConf.merge(schema, OmegaConf.create(data))
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Export configuration to YAML.

        Args:
            path: Optional path to write YAML file.

        Returns:
            YAML string representation.
        """
        config = OmegaConf.structured(self)
        yaml_str = OmegaConf.to_yaml(config)

        if path:
            Path(path).write_text(yaml_str)

        return yaml_str

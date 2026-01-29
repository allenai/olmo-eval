"""YAML-based configuration for Beaker launch jobs.

Provides OmegaConf-based configuration loading for composing complex
evaluation experiments from YAML files.

Example config (eval_config.yaml):
    name: eval-llama-suite
    models:
      - llama3.1-8b
      - olmo-2-7b
    tasks:
      - mmlu
      - gsm8k
      - hellaswag
    cluster: h100
    gpus: 1
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
    config = EvalConfig.from_yaml("eval_config.yaml", overrides=["gpus=4", "priority=high"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from olmo_eval.core.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE


@dataclass
class ModelConfig:
    """Configuration for a single model with optional resource overrides for Beaker launch.

    This allows specifying per-model resources for mixed-size evaluations.

    Note: There are multiple ModelConfig classes with different purposes:
    - core/configs.py:ModelConfig - Core model config for inference
    - launch/config.py:ModelConfig (this one) - Beaker launch config with resource settings
    - runners/mixins.py:ModelConfig - Metrics output format for JSON serialization

    Attributes:
        name_or_path: Model name, HuggingFace path, or local checkpoint path (required).
        alias: Optional short name for experiment naming. If not set, a short name
            is derived from name_or_path (last path component, or last 16 chars for long paths).
        gpus: Number of GPUs per model instance (overrides default).
        parallelism: Number of model instances to run in parallel. Total GPUs
            requested will be gpus × parallelism.
        cluster: Cluster for this model (overrides default).
        preemptible: Whether this model's jobs can be preempted.
        timeout: Timeout for this model's jobs.
        shared_memory: Shared memory size (e.g., "10GiB").
        use_async: Enable parallel task execution (overrides default).
        use_async_stream: Enable streaming async with vLLM (overrides default).
        num_workers: Number of workers for async modes (overrides default).
        gpus_per_worker: GPUs per worker for async modes (overrides default).
        provider: Inference provider to install for this model (e.g., "vllm").
        load_format: vLLM model loading format (e.g., "runai_streamer" for distributed loading).
        extra_loader_config: Extra config for model loader (e.g., {"distributed": true}).

    Example:
        models:
          - name_or_path: llama3.1-8b
            gpus: 1
            parallelism: 4  # 4 instances × 1 GPU = 4 GPUs total
            provider: vllm==0.13.0
          - name_or_path: /weka/checkpoints/my-model/step1000-hf
            alias: my-model-1k  # Short name for experiment naming
            gpus: 4
            parallelism: 2  # 2 instances × 4 GPUs = 8 GPUs total
            provider: transformers
            use_async: true
            num_workers: 2
            gpus_per_worker: 4
            timeout: 48h
          - name_or_path: llama3.1-70b
            gpus: 4
            load_format: runai_streamer  # Use distributed streaming loader
            extra_loader_config:
              distributed: true
              concurrency: 16
    """

    name_or_path: str = MISSING
    alias: str | None = None
    gpus: int | None = None
    parallelism: int | None = None
    cluster: str | None = None
    preemptible: bool | None = None
    timeout: str | None = None
    shared_memory: str | None = None

    # Async execution settings
    use_async: bool | None = None
    use_async_stream: bool | None = None
    num_workers: int | None = None
    gpus_per_worker: int | None = None

    # Runtime inference provider installation
    provider: str | None = None

    # vLLM model loading configuration
    load_format: str | None = None
    extra_loader_config: dict[str, Any] | None = None  # e.g., {"distributed": true}


def parse_model_config(model: str | dict[str, Any] | ModelConfig) -> ModelConfig:
    """Parse a model specification into ModelConfig.

    Handles both simple string format and detailed dict/ModelConfig format.
    Inline overrides (::key=value) are parsed and applied to matching ModelConfig fields.

    Args:
        model: Model name/path string, dict with model config, or ModelConfig.

    Returns:
        ModelConfig instance with any inline overrides applied.

    Examples:
        parse_model_config("llama3.1-8b")
        parse_model_config({"name_or_path": "llama3.1-70b", "gpus": 4})
        parse_model_config("llama3.1-8b::provider=vllm,load_format=auto")
    """
    if isinstance(model, ModelConfig):
        return model
    if isinstance(model, str):
        from olmo_eval.evals.tasks.core.registry import parse_overrides

        # Parse inline overrides (::key=value) from model spec
        name_or_path, _, override_str = model.partition("::")
        overrides = parse_overrides(override_str) if override_str else {}

        # Map inline overrides to ModelConfig fields
        config_kwargs: dict[str, Any] = {"name_or_path": name_or_path}

        # Fields that can be set via inline overrides
        if "provider" in overrides:
            config_kwargs["provider"] = overrides["provider"]
        if "load_format" in overrides:
            config_kwargs["load_format"] = overrides["load_format"]
        if "gpus_per_worker" in overrides:
            config_kwargs["gpus_per_worker"] = int(overrides["gpus_per_worker"])
        if "extra_loader_config" in overrides:
            config_kwargs["extra_loader_config"] = overrides["extra_loader_config"]

        return ModelConfig(**config_kwargs)
    if isinstance(model, dict):
        schema = OmegaConf.structured(ModelConfig)
        merged = OmegaConf.merge(schema, OmegaConf.create(model))
        return OmegaConf.to_object(merged)  # type: ignore[return-value]
    raise TypeError(f"Invalid model specification: {type(model)}")


def get_model_short_name(model: ModelConfig) -> str:
    """Get a short display name for a model suitable for experiment naming.

    If alias is set, returns the alias. Otherwise derives a short name from
    name_or_path by taking the last path component. If the result is empty
    or longer than 32 characters, takes the last 16 characters of the path.

    Args:
        model: ModelConfig instance.

    Returns:
        Short name suitable for use in experiment names.

    Examples:
        >>> get_model_short_name(ModelConfig(name_or_path="llama3.1-8b"))
        'llama3.1-8b'
        >>> get_model_short_name(ModelConfig(name_or_path="meta-llama/Llama-3.1-8B"))
        'llama-3.1-8b'
        >>> get_model_short_name(ModelConfig(name_or_path="/weka/checkpoints/model/step1000-hf/"))
        'step1000-hf'
        >>> get_model_short_name(
        ...     ModelConfig(name_or_path="/weka/checkpoints/model/", alias="my-model")
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

    Removes common suffixes and prefixes, strips variants (after ':' or '::'),
    and truncates to max_len characters.
    """
    # Strip @priority suffix if present
    if "@" in task:
        task = task.split("@")[0]

    # Strip variant/regime suffix (e.g., "mmlu::olmes" -> "mmlu", "arc:mc" -> "arc")
    if "::" in task:
        task = task.split("::")[0]
    elif ":" in task:
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
        tasks: List of task names (may include @priority or ::variant suffixes).
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

        # Simple format
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
        models: List of model names/paths or ModelConfig dicts (required).
            Each model can specify its own inference provider via the 'provider' field.
        tasks: List of task specs, optionally with @priority suffix (required).
        cluster: Default cluster alias or full name.
        gpus: Default number of GPUs per model instance.
        parallelism: Default number of model instances to run in parallel.
        max_gpus_per_node: Maximum GPUs available per node. When total GPUs
            (gpus × parallelism) exceeds this, tasks are split across experiments.
        priority: Default job priority for tasks without @priority suffix.
        preemptible: Default preemption setting.
        timeout: Default job timeout (e.g., "24h", "48h").
        retries: Number of retries on failure.
        workspace: Beaker workspace.
        budget: Beaker budget.
        beaker_image: Container image to use.
        description: Optional experiment description.
        groups: List of Beaker groups to add experiments to.
        use_async: Enable parallel task execution with multiple workers.
        use_async_stream: Enable streaming async with vLLM's AsyncLLMEngine (vLLM only).
        num_workers: Number of workers for async modes.
        gpus_per_worker: GPUs per worker for async modes.
    """

    # Required fields
    name: str = MISSING
    models: list[Any] = MISSING  # list[str] or list[dict] for ModelConfig
    tasks: list[str] = MISSING

    # Default cluster and resources (can be overridden per-model)
    cluster: str | None = None
    gpus: int = 1
    parallelism: int = 1
    max_gpus_per_node: int = DEFAULT_MAX_GPUS_PER_NODE

    # Default job settings (can be overridden per-model)
    priority: str = "normal"
    preemptible: bool = True
    timeout: str = "24h"
    retries: int | None = None

    # Async execution defaults
    use_async: bool = False
    use_async_stream: bool = False
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # Beaker settings
    workspace: str | None = None
    budget: str | None = None
    beaker_image: str | None = None
    description: str | None = None
    groups: list[str] | None = None  # Groups to add experiments to

    def get_model_configs(self) -> list[ModelConfig]:
        """Get parsed ModelConfig objects for all models.

        Returns a list of ModelConfig objects, parsing simple strings
        into ModelConfig with just the name set.

        Returns:
            List of ModelConfig objects.
        """
        return [parse_model_config(m) for m in self.models]

    def get_model_resources(self, model: ModelConfig) -> dict[str, Any]:
        """Get effective resources for a model, merging defaults with overrides.

        Args:
            model: ModelConfig with optional resource overrides.

        Returns:
            Dict with effective resource values including:
            - gpus: GPUs per model instance (not multiplied by parallelism)
            - parallelism: Number of model instances to run
            - Other resource settings (cluster, timeout, etc.)
        """
        # Determine async settings
        use_async = model.use_async if model.use_async is not None else self.use_async
        use_async_stream = (
            model.use_async_stream if model.use_async_stream is not None else self.use_async_stream
        )
        num_workers = model.num_workers if model.num_workers is not None else self.num_workers
        gpus_per_worker = (
            model.gpus_per_worker if model.gpus_per_worker is not None else self.gpus_per_worker
        )

        # Calculate GPUs per model instance
        if (use_async or use_async_stream) and num_workers is not None:
            gpus_per_model = num_workers * gpus_per_worker
        else:
            gpus_per_model = model.gpus if model.gpus is not None else self.gpus

        # Determine parallelism
        parallelism = model.parallelism if model.parallelism is not None else self.parallelism

        return {
            "gpus": gpus_per_model,
            "parallelism": parallelism,
            "cluster": model.cluster if model.cluster is not None else self.cluster,
            "preemptible": model.preemptible if model.preemptible is not None else self.preemptible,
            "timeout": model.timeout if model.timeout is not None else self.timeout,
            "shared_memory": model.shared_memory,  # None uses BeakerJobConfig default
            "use_async": use_async,
            "use_async_stream": use_async_stream,
            "num_workers": num_workers,
            "gpus_per_worker": gpus_per_worker,
            "provider": model.provider,
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
            overrides: Optional list of dotlist overrides (e.g., ["gpus=4", "priority=high"]).

        Returns:
            EvalConfig instance with merged configuration.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            omegaconf.errors.MissingMandatoryValue: If required fields are missing.

        Example:
            # Load basic config
            config = EvalConfig.from_yaml("eval_config.yaml")

            # Load with overrides
            config = EvalConfig.from_yaml(
                "eval_config.yaml",
                overrides=["gpus=4", "cluster=a100"]
            )
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Load YAML file
        file_config = OmegaConf.load(path)

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


# Pre-defined configuration templates
TEMPLATES: dict[str, dict[str, Any]] = {
    "quick": {
        "cluster": "h100",
        "gpus": 1,
        "priority": "normal",
        "timeout": "4h",
        "preemptible": True,
    },
    "standard": {
        "cluster": "h100",
        "gpus": 1,
        "priority": "normal",
        "timeout": "24h",
        "preemptible": True,
    },
    "large-model": {
        "cluster": "h100",
        "gpus": 4,
        "priority": "high",
        "timeout": "48h",
        "preemptible": False,
    },
    "urgent": {
        "cluster": "h100",
        "gpus": 1,
        "priority": "urgent",
        "timeout": "24h",
        "preemptible": False,
    },
}


def get_template(name: str) -> dict[str, Any]:
    """Get a pre-defined configuration template.

    Available templates:
        - quick: Fast jobs with 4h timeout
        - standard: Normal priority, 24h timeout
        - large-model: 4 GPUs, high priority, 48h timeout
        - urgent: Urgent priority, non-preemptible

    Args:
        name: Template name.

    Returns:
        Dictionary with template configuration.

    Raises:
        ValueError: If template name is not found.
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    return TEMPLATES[name].copy()

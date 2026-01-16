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
    config = LaunchConfig.from_yaml("eval_config.yaml")
    # Or with CLI overrides:
    config = LaunchConfig.from_yaml("eval_config.yaml", overrides=["gpus=4", "priority=high"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from olmo_eval.core.constants.infrastructure import DEFAULT_MAX_GPUS_PER_NODE


@dataclass
class ModelConfig:
    """Configuration for a single model with optional resource overrides.

    This allows specifying per-model resources for mixed-size evaluations.

    Attributes:
        name_or_path: Model name, HuggingFace path, or local checkpoint path (required).
        gpus: Number of GPUs per model instance (overrides default).
        parallelism: Number of model instances to run in parallel. Total GPUs
            requested will be gpus × parallelism.
        cluster: Cluster for this model (overrides default).
        preemptible: Whether this model's jobs can be preempted.
        timeout: Timeout for this model's jobs.
        shared_memory: Shared memory size (e.g., "10GiB").
        use_async: Enable parallel task execution (overrides default).
        num_workers: Number of workers for async mode (overrides default).
        gpus_per_worker: GPUs per worker for async mode (overrides default).
        backend: Backend to install for this model (e.g., "vllm==0.13.0").

    Example:
        models:
          - name_or_path: llama3.1-8b
            gpus: 1
            parallelism: 4  # 4 instances × 1 GPU = 4 GPUs total
            backend: vllm==0.13.0
          - name_or_path: llama3.1-70b
            gpus: 4
            parallelism: 2  # 2 instances × 4 GPUs = 8 GPUs total
            backend: transformers
            use_async: true
            num_workers: 2
            gpus_per_worker: 4
            timeout: 48h
    """

    name_or_path: str = MISSING
    gpus: int | None = None
    parallelism: int | None = None
    cluster: str | None = None
    preemptible: bool | None = None
    timeout: str | None = None
    shared_memory: str | None = None

    # Async execution settings
    use_async: bool | None = None
    num_workers: int | None = None
    gpus_per_worker: int | None = None

    # Runtime backend installation (overrides default backend)
    backend: str | None = None


def parse_model_config(model: str | dict[str, Any] | ModelConfig) -> ModelConfig:
    """Parse a model specification into ModelConfig.

    Handles both simple string format and detailed dict/ModelConfig format.

    Args:
        model: Model name/path string, dict with model config, or ModelConfig.

    Returns:
        ModelConfig instance.

    Examples:
        parse_model_config("llama3.1-8b")
        parse_model_config({"name_or_path": "llama3.1-70b", "gpus": 4})
    """
    if isinstance(model, ModelConfig):
        return model
    if isinstance(model, str):
        return ModelConfig(name_or_path=model)
    if isinstance(model, dict):
        schema = OmegaConf.structured(ModelConfig)
        merged = OmegaConf.merge(schema, OmegaConf.create(model))
        return OmegaConf.to_object(merged)  # type: ignore[return-value]
    raise TypeError(f"Invalid model specification: {type(model)}")


@dataclass
class LaunchConfig:
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
        backends: Optional dependency groups to install at runtime (e.g., ["vllm", "hf"]).
        flash_attn: Set to 3 to use Flash Attention 3 (for Hopper GPUs). FA2 is pre-installed.
        no_flash_attn: If True, uninstall Flash Attention at runtime (disables FA2).
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
    num_workers: int | None = None
    gpus_per_worker: int = 1

    # Beaker settings
    workspace: str | None = None
    budget: str | None = None
    beaker_image: str | None = None
    description: str | None = None
    groups: list[str] | None = None  # Groups to add experiments to

    # Runtime backend installation
    backends: list[str] | None = None

    # Flash Attention options
    flash_attn: int | None = None  # Set to 3 to switch to FA3
    no_flash_attn: bool = False  # Uninstall FA2 at runtime

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
        num_workers = model.num_workers if model.num_workers is not None else self.num_workers
        gpus_per_worker = (
            model.gpus_per_worker if model.gpus_per_worker is not None else self.gpus_per_worker
        )

        # Calculate GPUs per model instance
        if use_async and num_workers is not None:
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
            "num_workers": num_workers,
            "gpus_per_worker": gpus_per_worker,
            "backend": model.backend,
        }

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        overrides: list[str] | None = None,
    ) -> LaunchConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.
            overrides: Optional list of dotlist overrides (e.g., ["gpus=4", "priority=high"]).

        Returns:
            LaunchConfig instance with merged configuration.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            omegaconf.errors.MissingMandatoryValue: If required fields are missing.

        Example:
            # Load basic config
            config = LaunchConfig.from_yaml("eval_config.yaml")

            # Load with overrides
            config = LaunchConfig.from_yaml(
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
    def from_dict(cls, data: dict[str, Any]) -> LaunchConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            LaunchConfig instance.
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

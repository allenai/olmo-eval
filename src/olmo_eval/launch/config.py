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


@dataclass
class LaunchConfig:
    """Configuration for launching Beaker evaluation jobs.

    This dataclass can be loaded from YAML files using OmegaConf,
    allowing for complex configuration composition and overrides.

    Attributes:
        name: Experiment name (required).
        models: List of model names or HuggingFace paths (required).
        tasks: List of task specs, optionally with @priority suffix (required).
        cluster: Cluster alias or full name.
        gpus: Number of GPUs per job.
        priority: Default job priority (can be overridden per-task with @priority).
        preemptible: Whether jobs can be preempted.
        timeout: Job timeout (e.g., "24h", "48h").
        retries: Number of retries on failure.
        workspace: Beaker workspace.
        budget: Beaker budget.
        beaker_image: Container image to use.
        description: Optional experiment description.
    """

    # Required fields
    name: str = MISSING
    models: list[str] = MISSING
    tasks: list[str] = MISSING

    # Cluster and resources
    cluster: str = "h100"
    gpus: int = 1

    # Job settings
    priority: str = "normal"
    preemptible: bool = True
    timeout: str = "24h"
    retries: int | None = None

    # Beaker settings
    workspace: str | None = None
    budget: str | None = None
    beaker_image: str | None = None
    description: str | None = None

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

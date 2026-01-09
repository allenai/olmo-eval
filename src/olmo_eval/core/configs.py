"""Configuration types, presets, and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


@dataclass
class ModelConfig:
    """Model/backend configuration."""

    model: str
    backend: str = "hf"  # BackendType value as string to avoid circular import
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level configuration for an evaluation run."""

    model: ModelConfig
    tasks: list[str] = field(default_factory=list)
    output_dir: str = "./results"
    batch_size: int | str = "auto"


def load_config(path: str) -> DictConfig | ListConfig:
    """Load a YAML configuration file."""
    return OmegaConf.load(path)


def expand_tasks(tasks: list[str]) -> list[str]:
    """Expand suites and specs to individual task names.

    Supports both Suite names from the named_tasks registry
    and individual task specs.

    Args:
        tasks: List of task specs or suite names.

    Returns:
        Flattened list with suites expanded to their constituent tasks.
    """
    from olmo_eval.evals.suites import get_suite, suite_exists

    result = []
    for t in tasks:
        if suite_exists(t):
            suite = get_suite(t)
            result.extend(suite.expand())
        else:
            result.append(t)
    return result


def get_model_config(name: str, **overrides: Any) -> ModelConfig:
    """Get a model config by preset name with optional overrides.

    Args:
        name: Preset name (e.g., "llama3.1-8b") or HuggingFace model path.
        **overrides: Override specific config fields.

    Returns:
        ModelConfig instance.
    """
    from olmo_eval.core.constants.models import get_model_presets

    models = get_model_presets()
    if name in models:
        base = models[name]
        if overrides:
            # Create new config with overrides
            return ModelConfig(
                model=overrides.get("model", base.model),
                backend=overrides.get("backend", base.backend),
                revision=overrides.get("revision", base.revision),
                trust_remote_code=overrides.get("trust_remote_code", base.trust_remote_code),
                dtype=overrides.get("dtype", base.dtype),
                extra_args={**base.extra_args, **overrides.get("extra_args", {})},
            )
        return base
    # Treat as HuggingFace model path
    return ModelConfig(model=name, **overrides)

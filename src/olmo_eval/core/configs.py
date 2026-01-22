"""Configuration types, presets, and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR


@dataclass
class ModelConfig:
    """Model/backend configuration."""

    model: str
    tokenizer: str | None = None  # Tokenizer path/identifier, defaults to model if None
    backend: str = "vllm"  # BackendType value as string to avoid circular import
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: int | None = None  # Override model's default context length (vLLM)
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level configuration for an evaluation run."""

    model: ModelConfig
    tasks: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    batch_size: int | str = "auto"


def load_config(path: str) -> DictConfig | ListConfig:
    """Load a YAML configuration file."""
    return OmegaConf.load(path)


def expand_tasks(tasks: list[str]) -> list[str]:
    """Expand suites and specs to individual task names.

    Supports both Suite names from the named_tasks registry
    and individual task specs. Preserves inline overrides (::key=value)
    and priority suffixes (@priority) when expanding suites.

    Args:
        tasks: List of task specs or suite names, optionally with
               overrides (::key=value) and/or priority (@priority).

    Returns:
        Flattened list with suites expanded to their constituent tasks,
        with overrides and priorities propagated to each expanded task.
    """
    from olmo_eval.evals.suites import get_suite, suite_exists

    result = []
    for t in tasks:
        # Parse out priority suffix first (e.g., "suite::temp=0@high" -> "suite::temp=0", "high")
        priority_suffix = ""
        spec_without_priority = t
        if "@" in t:
            spec_without_priority, priority = t.rsplit("@", 1)
            priority_suffix = f"@{priority}"

        # Parse out overrides (e.g., "suite:variant::temp=0" -> "suite:variant", "temp=0")
        override_suffix = ""
        base_spec = spec_without_priority
        if "::" in spec_without_priority:
            base_spec, overrides = spec_without_priority.split("::", 1)
            override_suffix = f"::{overrides}"

        # Check if the base spec (without overrides/priority) is a suite
        if suite_exists(base_spec):
            suite = get_suite(base_spec)
            # Propagate overrides and priority to each expanded task
            for expanded_task in suite.expand():
                result.append(f"{expanded_task}{override_suffix}{priority_suffix}")
        else:
            result.append(t)
    return result


def validate_tasks(tasks: list[str]) -> tuple[list[str], list[str]]:
    """Validate that all tasks/suites exist and return expanded task list.

    Args:
        tasks: List of task specs or suite names, optionally with
               overrides (::key=value) and/or priority (@priority).

    Returns:
        Tuple of (valid_tasks, invalid_tasks). valid_tasks is the expanded list
        of all task specs. invalid_tasks contains any specs that don't exist.
    """
    from olmo_eval.evals.suites import suite_exists
    from olmo_eval.evals.tasks import task_exists

    valid_tasks = []
    invalid_tasks = []

    expanded = expand_tasks(tasks)

    for spec in expanded:
        # Strip priority suffix first (e.g., "task::temp=0@high" -> "task::temp=0")
        task_spec = spec.rsplit("@", 1)[0] if "@" in spec else spec

        # task_exists handles ::overrides internally via parse_task_spec
        if task_exists(task_spec):
            valid_tasks.append(spec)
        elif suite_exists(task_spec.split("::")[0] if "::" in task_spec else task_spec):
            # It's a suite that wasn't expanded (shouldn't happen but handle it)
            valid_tasks.append(spec)
        else:
            invalid_tasks.append(spec)

    return valid_tasks, invalid_tasks


def _parse_int_override(value: Any) -> int | None:
    """Parse an override value as int, handling string conversion."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return int(value)


# Keys that are vLLM/backend-specific and should not be passed to ModelConfig
# These are handled separately by the runners
_BACKEND_ONLY_KEYS = {"load_format", "extra_loader_config", "attention_backend", "gpus_per_worker"}


def get_model_config(name: str, **overrides: Any) -> ModelConfig:
    """Get a model config by preset name with optional overrides.

    Args:
        name: Preset name (e.g., "llama3.1-8b") or HuggingFace model path.
        **overrides: Override specific config fields.

    Returns:
        ModelConfig instance.
    """
    from olmo_eval.core.constants.models import get_model_presets

    # Filter out backend-specific keys that don't belong in ModelConfig
    filtered_overrides = {k: v for k, v in overrides.items() if k not in _BACKEND_ONLY_KEYS}

    models = get_model_presets()
    if name in models:
        base = models[name]
        if filtered_overrides:
            # Parse max_model_len as int if present (inline overrides come as strings)
            max_model_len_override = filtered_overrides.get("max_model_len")
            effective_max_model_len = (
                _parse_int_override(max_model_len_override)
                if max_model_len_override is not None
                else base.max_model_len
            )

            # Create new config with overrides
            return ModelConfig(
                model=filtered_overrides.get("model", base.model),
                tokenizer=filtered_overrides.get("tokenizer", base.tokenizer),
                backend=filtered_overrides.get("backend", base.backend),
                revision=filtered_overrides.get("revision", base.revision),
                trust_remote_code=filtered_overrides.get(
                    "trust_remote_code", base.trust_remote_code
                ),
                dtype=filtered_overrides.get("dtype", base.dtype),
                max_model_len=effective_max_model_len,
                extra_args={**base.extra_args, **filtered_overrides.get("extra_args", {})},
            )
        return base

    # Treat as HuggingFace model path - parse max_model_len if present
    if "max_model_len" in filtered_overrides:
        filtered_overrides["max_model_len"] = _parse_int_override(
            filtered_overrides["max_model_len"]
        )
    return ModelConfig(model=name, **filtered_overrides)

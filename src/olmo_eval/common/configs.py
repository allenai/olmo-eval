"""Configuration types, presets, and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import DictConfig, ListConfig, OmegaConf

from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.harness.config import ProviderConfig


@dataclass
class RunConfig:
    """Top-level configuration for an evaluation run."""

    model: ProviderConfig
    tasks: list[str] = field(default_factory=list)
    output_dir: str = BEAKER_RESULT_DIR
    batch_size: int | Literal["auto"] = "auto"


def load_config(path: str) -> DictConfig | ListConfig:
    """Load a YAML configuration file."""
    return OmegaConf.load(path)


def expand_tasks(tasks: list[str]) -> list[str]:
    """Expand suites and specs to individual task names.

    Supports both Suite names from the named_tasks registry
    and individual task specs. Preserves priority suffixes (@priority)
    when expanding suites.

    Args:
        tasks: List of task specs or suite names, optionally with
               priority (@priority).

    Returns:
        Flattened list with suites expanded to their constituent tasks,
        with priorities propagated to each expanded task.
    """
    from olmo_eval.evals.suites import get_suite, suite_exists

    result = []
    for t in tasks:
        # Parse out priority suffix (e.g., "suite@high" -> "suite", "high")
        priority_suffix = ""
        base_spec = t
        if "@" in t:
            base_spec, priority = t.rsplit("@", 1)
            priority_suffix = f"@{priority}"

        # Check if the base spec (without priority) is a suite
        if suite_exists(base_spec):
            suite = get_suite(base_spec)
            # Propagate priority to each expanded task
            for expanded_task in suite.expand():
                result.append(f"{expanded_task}{priority_suffix}")
        else:
            result.append(t)
    return result


def validate_tasks(tasks: list[str]) -> tuple[list[str], list[str]]:
    """Validate that all tasks/suites exist and return expanded task list.

    Args:
        tasks: List of task specs or suite names, optionally with
               priority (@priority).

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
        # Strip priority suffix (e.g., "task@high" -> "task")
        task_spec = spec.rsplit("@", 1)[0] if "@" in spec else spec

        if task_exists(task_spec):
            valid_tasks.append(spec)
        elif suite_exists(task_spec):
            # It's a suite that wasn't expanded (shouldn't happen but handle it)
            valid_tasks.append(spec)
        else:
            invalid_tasks.append(spec)

    return valid_tasks, invalid_tasks


def validate_task_metrics(tasks: list[str]) -> tuple[list[str], list[str]]:
    """Check which tasks have metrics configured.

    Tasks without metrics cannot be scored, making the evaluation run useless.
    This validation helps catch configuration errors early.

    Args:
        tasks: List of task specs (should already be validated as existing).

    Returns:
        Tuple of (tasks_with_metrics, tasks_without_metrics).
    """
    from olmo_eval.evals.tasks import get_task

    with_metrics = []
    without_metrics = []

    for spec in tasks:
        # Strip priority suffix
        task_spec = spec.rsplit("@", 1)[0] if "@" in spec else spec

        try:
            task = get_task(task_spec)
            if task.config.metrics:
                with_metrics.append(spec)
            else:
                without_metrics.append(spec)
        except Exception:
            # If we can't instantiate the task, skip it (should be caught elsewhere)
            without_metrics.append(spec)

    return with_metrics, without_metrics


# Keys that are runner-specific and should not be passed to ProviderConfig
_BACKEND_ONLY_KEYS = {
    "attention_backend",
    "gpus",
}


def get_provider_config(name: str, **overrides: Any) -> ProviderConfig:
    """Get a provider config by preset name with optional overrides.

    Args:
        name: Preset name (e.g., "llama3.1-8b") or HuggingFace model path.
        **overrides: Override specific config fields.

    Returns:
        ProviderConfig instance.
    """
    from olmo_eval.common.constants.models import get_model_presets

    # Filter out backend-specific keys that don't belong in ProviderConfig
    filtered_overrides = {k: v for k, v in overrides.items() if k not in _BACKEND_ONLY_KEYS}

    presets = get_model_presets()
    if name in presets:
        base = presets[name]

        if filtered_overrides:
            # Build new config with overrides
            return ProviderConfig(
                kind=filtered_overrides.get("kind", base.kind),
                model=filtered_overrides.get("model", base.model),
                base_url=filtered_overrides.get("base_url", base.base_url),
                tokenizer=filtered_overrides.get("tokenizer", base.tokenizer),
                revision=filtered_overrides.get("revision", base.revision),
                trust_remote_code=filtered_overrides.get(
                    "trust_remote_code", base.trust_remote_code
                ),
                dtype=filtered_overrides.get("dtype", base.dtype),
                max_model_len=filtered_overrides.get("max_model_len", base.max_model_len),
                max_concurrency=filtered_overrides.get("max_concurrency", base.max_concurrency),
                required_secrets=tuple(
                    filtered_overrides.get("required_secrets", base.required_secrets)
                ),
                package=filtered_overrides.get("package", base.package),
                kwargs={**base.kwargs, **filtered_overrides.get("kwargs", {})},
            )
        return base

    # Not a preset - create ProviderConfig directly
    return ProviderConfig(model=name, **filtered_overrides)


# Backward compatibility alias
def get_model_config(name: str, **overrides: Any) -> ProviderConfig:
    """Alias for get_provider_config for backward compatibility."""
    return get_provider_config(name, **overrides)

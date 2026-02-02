"""Configuration types, presets, and utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from omegaconf import DictConfig, ListConfig, OmegaConf

from olmo_eval.core.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.core.literals import DtypeLiteral, ProviderLiteral


@dataclass
class ModelConfig:
    """Core model configuration for inference.

    For agent tasks, models can be specified in two ways:
    1. HuggingFace model/path with provider="vllm" - starts local vLLM server
    2. API endpoint with model_url - uses OpenAI-compatible API directly

    Example presets for API-based models:
        "gpt-4o": ModelConfig(model="gpt-4o", model_url="https://api.openai.com/v1")
        "claude-3": ModelConfig(model="claude-3-opus", model_url="https://api.anthropic.com")
    """

    model: str
    tokenizer: str | None = None  # Tokenizer path/identifier, defaults to model if None
    provider: ProviderLiteral = "vllm"
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: DtypeLiteral = "auto"
    max_model_len: int | None = None  # Override model's default context length (vLLM)
    extra_args: dict[str, Any] = field(default_factory=dict)
    # API endpoint for OpenAI-compatible APIs (agent tasks only)
    # When set, agent tasks use this URL directly instead of starting vLLM
    model_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        from dataclasses import asdict

        return asdict(self)


@dataclass
class RunConfig:
    """Top-level configuration for an evaluation run."""

    model: ModelConfig
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


# Keys that are vLLM/backend-specific and should not be passed to ModelConfig
# These are handled separately by the runners
_BACKEND_ONLY_KEYS = {
    "load_format",
    "extra_loader_config",
    "attention_backend",
    "gpus_per_worker",
    "gpus",
}


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
            return ModelConfig(
                model=filtered_overrides.get("model", base.model),
                tokenizer=filtered_overrides.get("tokenizer", base.tokenizer),
                provider=filtered_overrides.get("provider", base.provider),
                revision=filtered_overrides.get("revision", base.revision),
                trust_remote_code=filtered_overrides.get(
                    "trust_remote_code", base.trust_remote_code
                ),
                dtype=filtered_overrides.get("dtype", base.dtype),
                max_model_len=filtered_overrides.get("max_model_len", base.max_model_len),
                extra_args={**base.extra_args, **filtered_overrides.get("extra_args", {})},
            )
        return base

    # Check if model_url was provided in overrides
    model_url = filtered_overrides.pop("model_url", None)
    config = ModelConfig(model=name, **filtered_overrides)
    if model_url:
        config = ModelConfig(
            model=config.model,
            tokenizer=config.tokenizer,
            provider=config.provider,
            revision=config.revision,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            extra_args=config.extra_args,
            model_url=model_url,
        )
    return config

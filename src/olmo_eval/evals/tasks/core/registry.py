"""Task registry for registering and retrieving tasks by name.

Task specs follow the format: task_name[:variant1[:variant2...]][::key=value,...]

Examples:
    - "arc_easy" - base task
    - "arc_easy:mc" - task with multiple-choice variant
    - "arc_easy:olmes" - task with olmes regime (regimes are now variants)
    - "arc_easy:mc:olmes" - task with variant and regime
    - "mbpp:3shot:bpb:none" - task with stacked variants and regime
    - "gsm8k:olmes::temperature=0.6" - task with regime and override
    - "gsm8k::temperature=0.6,max_tokens=512" - task with overrides only
"""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from .base import Task, TaskConfig

T = TypeVar("T", bound=type[Task])

# Module-level registries
_tasks: dict[str, type[Task]] = {}
_configs: dict[str, Callable[[], TaskConfig]] = {}
_variants: dict[str, dict[str, dict[str, Any]]] = {}
_regimes: dict[str, dict[str, dict[str, Any]]] = {}


def register(
    name: str,
    config_factory: Callable[[], TaskConfig],
) -> Callable[[T], T]:
    """Register a task class with a name and config factory.

    Usage:
        @register("mmlu", lambda: TaskConfig(name="mmlu", data_source=DataSource(path="cais/mmlu")))
        class MMLUTask(Task):
            ...
    """

    def decorator(cls: T) -> T:
        if name in _tasks:
            raise ValueError(f"Task '{name}' already registered")
        _tasks[name] = cls
        _configs[name] = config_factory
        return cls

    return decorator


def register_variant(task_name: str, variant: str, **overrides: Any) -> None:
    """Register a variant (format modifier) for a task.

    Variants modify how a task is evaluated (e.g., :mc for multiple choice,
    :gen for generation). They are applied before regimes.

    Args:
        task_name: Name of the base task (must already be registered).
        variant: Name of the variant (e.g., "mc", "gen").
        **overrides: TaskConfig field overrides for this variant.

    Raises:
        ValueError: If the task is not registered.
    """
    if task_name not in _tasks:
        raise ValueError(
            f"Cannot register variant '{variant}' for unknown task '{task_name}'. "
            f"Register the task first using @register()."
        )
    _variants.setdefault(task_name, {})[variant] = overrides


def register_regime(task_name: str, regime: str, **overrides: Any) -> None:
    """Register a regime (configuration preset) for a task.

    Regimes are configuration presets that define evaluation settings
    (e.g., ::olmes for OLMo-style evaluation). They are applied after variants.

    Args:
        task_name: Name of the base task (must already be registered).
        regime: Name of the regime (e.g., "olmes").
        **overrides: TaskConfig field overrides for this regime.

    Raises:
        ValueError: If the task is not registered.
    """
    if task_name not in _tasks:
        raise ValueError(
            f"Cannot register regime '{regime}' for unknown task '{task_name}'. "
            f"Register the task first using @register()."
        )
    _regimes.setdefault(task_name, {})[regime] = overrides


def parse_overrides(override_str: str) -> dict[str, Any]:
    """Parse 'key=value,key=value' into dict with type coercion.

    Supports JSON values for complex configs (e.g., extra_loader_config={"distributed":true}).

    Args:
        override_str: Override string like "temperature=0.6,max_tokens=512"

    Returns:
        Dict with appropriately typed values.

    Examples:
        >>> parse_overrides("temperature=0.6,max_tokens=512")
        {"temperature": 0.6, "max_tokens": 512}
        >>> parse_overrides("provider=vllm")
        {"provider": "vllm"}
        >>> parse_overrides('extra_loader_config={"distributed":true}')
        {"extra_loader_config": {"distributed": True}}
    """
    import json

    if not override_str:
        return {}

    result: dict[str, Any] = {}
    decoder = json.JSONDecoder()
    i = 0

    while i < len(override_str):
        # Skip commas and whitespace
        while i < len(override_str) and override_str[i] in ", ":
            i += 1
        if i >= len(override_str):
            break

        # Find key=value
        eq_pos = override_str.find("=", i)
        if eq_pos == -1:
            break

        key = override_str[i:eq_pos].strip()
        i = eq_pos + 1

        # Parse value - use raw_decode for JSON, otherwise read until comma
        if i < len(override_str) and override_str[i] in "{[":
            value, end = decoder.raw_decode(override_str, i)
            i = end
        else:
            comma_pos = override_str.find(",", i)
            value_str = (override_str[i:comma_pos] if comma_pos != -1 else override_str[i:]).strip()
            i = comma_pos if comma_pos != -1 else len(override_str)

            # Type coercion
            if key in {
                "num_fewshot",
                "limit",
                "fewshot_seed",
                "max_tokens",
                "max_model_len",
                "top_k",
                "num_samples",
            }:
                value = int(value_str)
            elif key in {"temperature", "top_p"}:
                value = float(value_str)
            else:
                value = value_str

        result[key] = value

    return result


def parse_task_spec(spec: str) -> tuple[str, list[str], dict[str, Any]]:
    """Parse a task spec into (task_name, variants, overrides).

    Spec format: task_name[:variant1[:variant2...]][::key=value,...]

    Note: Regimes are now treated as variants (no special :: delimiter for regimes).
    The :: delimiter is used exclusively for inline overrides.

    Args:
        spec: Task specification string.

    Returns:
        Tuple of (task_name, variants, overrides). Variants is a list (may be empty).
        Overrides is a dict (may be empty).

    Examples:
        >>> parse_task_spec("arc_easy")
        ("arc_easy", [], {})
        >>> parse_task_spec("arc_easy:mc")
        ("arc_easy", ["mc"], {})
        >>> parse_task_spec("arc_easy:olmes")
        ("arc_easy", ["olmes"], {})
        >>> parse_task_spec("arc_easy:mc:olmes")
        ("arc_easy", ["mc", "olmes"], {})
        >>> parse_task_spec("gsm8k:olmes::temperature=0.6")
        ("gsm8k", ["olmes"], {"temperature": 0.6})
        >>> parse_task_spec("gsm8k::temperature=0.6,max_tokens=512")
        ("gsm8k", [], {"temperature": 0.6, "max_tokens": 512})
    """
    # Split on :: to separate overrides
    main_part, _, override_str = spec.partition("::")

    # Parse overrides
    overrides = parse_overrides(override_str)

    # Split main part on : to get task name and variants
    parts = main_part.split(":")
    task_name = parts[0]
    variants = parts[1:] if len(parts) > 1 else []

    return task_name, variants, overrides


def get_task(spec: str, config_overrides: dict[str, Any] | None = None) -> Task:
    """Instantiate a task by spec.

    Spec format: task_name[:variant1[:variant2...]][::key=value,...]

    Note: Regimes are now treated as variants. When looking up a variant,
    we check both the variants and regimes registries.

    Task names with colons (e.g., "humaneval:bpb") are checked first before
    parsing as base_task:variant.

    Args:
        spec: Task specification (e.g., "arc_easy", "arc_easy:mc:olmes", "gsm8k::temperature=0.6").
        config_overrides: Additional config overrides to apply (highest priority).

    Returns:
        Instantiated Task with config (and variant/override if specified).

    Raises:
        KeyError: If task_name is not registered.
    """
    # First, check if the spec (or spec without inline overrides) is a directly registered task
    main_part, _, override_str = spec.partition("::")
    inline_overrides = parse_overrides(override_str)

    # Try progressively shorter prefixes to find a registered task with colons in its name
    # e.g., for "humaneval:bpb:mc", check "humaneval:bpb:mc", then "humaneval:bpb", then "humaneval"
    parts = main_part.split(":")
    task_name = None
    variants: list[str] = []

    for i in range(len(parts), 0, -1):
        candidate = ":".join(parts[:i])
        if candidate in _tasks:
            task_name = candidate
            variants = parts[i:]
            break

    if task_name is None:
        # Fall back to original parsing (first part is task name)
        task_name = parts[0]
        variants = parts[1:] if len(parts) > 1 else []

    if task_name not in _tasks:
        available = ", ".join(sorted(_tasks.keys()))
        raise KeyError(f"Unknown task '{task_name}'. Available: {available}")

    config = _configs[task_name]()

    # Apply variant/regime overrides in order (check both registries)
    for variant in variants:
        # First check variants registry
        if task_name in _variants and variant in _variants[task_name]:
            config = replace(config, **_variants[task_name][variant])
        # Then check regimes registry (regimes are now accessed as variants)
        elif task_name in _regimes and variant in _regimes[task_name]:
            config = replace(config, **_regimes[task_name][variant])
        else:
            available_variants = list(_variants.get(task_name, {}).keys())
            available_regimes = list(_regimes.get(task_name, {}).keys())
            available = sorted(set(available_variants + available_regimes))
            raise KeyError(
                f"Unknown variant '{variant}' for task '{task_name}'. "
                f"Available: {', '.join(available) if available else 'none'}"
            )

    # Apply inline overrides from spec (e.g., ::temperature=0.6)
    if inline_overrides:
        # Only apply TaskConfig fields, not sampling params
        taskconfig_fields = {"num_fewshot", "limit", "fewshot_seed"}
        for key, value in inline_overrides.items():
            if key in taskconfig_fields:
                config = replace(config, **{key: value})

    # Apply additional config overrides (highest priority)
    if config_overrides:
        config = replace(config, **config_overrides)

    return _tasks[task_name](config)


def list_tasks() -> list[str]:
    """List all registered task names."""
    return sorted(_tasks.keys())


def list_variants(task_name: str | None = None) -> dict[str, list[str]]:
    """List available variants, optionally filtered by task.

    Args:
        task_name: If provided, only return variants for this task.

    Returns:
        Dict mapping task names to their available variants.
    """
    if task_name:
        return {task_name: list(_variants.get(task_name, {}).keys())}
    return {name: list(variants.keys()) for name, variants in _variants.items()}


def list_regimes(task_name: str | None = None) -> dict[str, list[str]]:
    """List available regimes, optionally filtered by task.

    Args:
        task_name: If provided, only return regimes for this task.

    Returns:
        Dict mapping task names to their available regimes.
    """
    if task_name:
        return {task_name: list(_regimes.get(task_name, {}).keys())}
    return {name: list(regimes.keys()) for name, regimes in _regimes.items()}


def task_exists(spec: str) -> bool:
    """Check if a task spec is valid (task exists).

    Args:
        spec: Task specification string.

    Returns:
        True if the base task exists, False otherwise.
    """
    task_name, _variants, _overrides = parse_task_spec(spec)
    return task_name in _tasks


def clear_registry() -> None:
    """Clear registry (useful for testing)."""
    _tasks.clear()
    _configs.clear()
    _variants.clear()
    _regimes.clear()

"""Task registry for registering and retrieving tasks by name.

Task specs follow the format: task_name[:variant][::regime]

Examples:
    - "arc_easy" - base task
    - "arc_easy:mc" - task with multiple-choice variant
    - "arc_easy::olmes" - task with olmes regime
    - "arc_easy:mc::olmes" - task with variant and regime
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
        @register("mmlu", lambda: TaskConfig(name="mmlu", hf_dataset="cais/mmlu"))
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


def parse_task_spec(spec: str) -> tuple[str, str | None, str | None]:
    """Parse a task spec into (task_name, variant, regime).

    Spec format: task_name[:variant][::regime]

    Args:
        spec: Task specification string.

    Returns:
        Tuple of (task_name, variant, regime). Variant and regime may be None.

    Examples:
        >>> parse_task_spec("arc_easy")
        ("arc_easy", None, None)
        >>> parse_task_spec("arc_easy:mc")
        ("arc_easy", "mc", None)
        >>> parse_task_spec("arc_easy::olmes")
        ("arc_easy", None, "olmes")
        >>> parse_task_spec("arc_easy:mc::olmes")
        ("arc_easy", "mc", "olmes")
    """
    # First split on :: to separate regime
    task_part, _, regime = spec.partition("::")
    regime = regime or None

    # Then split task_part on : to separate variant
    task_name, _, variant = task_part.partition(":")
    variant = variant or None

    return task_name, variant, regime


def get_task(spec: str) -> Task:
    """Instantiate a task by spec.

    Spec format: task_name[:variant][::regime]

    Args:
        spec: Task specification (e.g., "arc_easy", "arc_easy:mc::olmes").

    Returns:
        Instantiated Task with config (and variant/regime overrides if specified).

    Raises:
        KeyError: If task_name is not registered.
    """
    task_name, variant, regime = parse_task_spec(spec)

    if task_name not in _tasks:
        available = ", ".join(sorted(_tasks.keys()))
        raise KeyError(f"Unknown task '{task_name}'. Available: {available}")

    config = _configs[task_name]()

    # Apply variant overrides first (if specified and registered)
    if variant and task_name in _variants and variant in _variants[task_name]:
        config = replace(config, **_variants[task_name][variant])

    # Apply regime overrides second (if specified and registered)
    if regime and task_name in _regimes and regime in _regimes[task_name]:
        config = replace(config, **_regimes[task_name][regime])

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
    task_name, _, _ = parse_task_spec(spec)
    return task_name in _tasks


def clear_registry() -> None:
    """Clear registry (useful for testing)."""
    _tasks.clear()
    _configs.clear()
    _variants.clear()
    _regimes.clear()

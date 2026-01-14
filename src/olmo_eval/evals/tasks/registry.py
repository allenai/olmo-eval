"""Task registry for registering and retrieving tasks by name."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypeVar

from .base import Task, TaskConfig

T = TypeVar("T", bound=type[Task])

# Module-level registries
_tasks: dict[str, type[Task]] = {}
_configs: dict[str, Callable[[], TaskConfig]] = {}
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


def register_regime(task_name: str, regime: str, **overrides: Any) -> None:
    """Register a regime (configuration variant) for a task.

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


def get_task(spec: str) -> Task:
    """Instantiate a task by name or spec (task_name::regime).

    Args:
        spec: Task name or "task_name::regime" format.

    Returns:
        Instantiated Task with config (and regime overrides if specified).
    """
    task_name, _, regime = spec.partition("::")

    if task_name not in _tasks:
        available = ", ".join(sorted(_tasks.keys()))
        raise KeyError(f"Unknown task '{task_name}'. Available: {available}")

    config = _configs[task_name]()

    # Apply regime overrides if specified
    if regime and task_name in _regimes and regime in _regimes[task_name]:
        config = replace(config, **_regimes[task_name][regime])

    return _tasks[task_name](config)


def list_tasks() -> list[str]:
    """List all registered task names."""
    return sorted(_tasks.keys())


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


def clear_registry() -> None:
    """Clear registry (useful for testing)."""
    _tasks.clear()
    _configs.clear()
    _regimes.clear()

"""Task registry for registering and retrieving tasks by name."""

from collections.abc import Callable
from typing import TypeVar

from .base import Task, TaskConfig

T = TypeVar("T", bound=type[Task])

# Module-level registry (not a singleton class)
_tasks: dict[str, type[Task]] = {}
_configs: dict[str, Callable[[], TaskConfig]] = {}


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


def get_task(name: str) -> Task:
    """Instantiate a registered task by name."""
    if name not in _tasks:
        available = ", ".join(sorted(_tasks.keys()))
        raise KeyError(f"Unknown task '{name}'. Available: {available}")
    config = _configs[name]()
    task_cls = _tasks[name]
    return task_cls(config)


def list_tasks() -> list[str]:
    """List all registered task names."""
    return sorted(_tasks.keys())


def clear_registry() -> None:
    """Clear registry (useful for testing)."""
    _tasks.clear()
    _configs.clear()

"""Task framework for evaluation."""

from .base import Task, TaskConfig
from .registry import clear_registry, get_task, list_tasks, register

__all__ = [
    "Task",
    "TaskConfig",
    "register",
    "get_task",
    "list_tasks",
    "clear_registry",
]

# Import task modules to trigger registration (must be after exports to avoid circular import)
from . import arc as _arc  # noqa: F401, E402

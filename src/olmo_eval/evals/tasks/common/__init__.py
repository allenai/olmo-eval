"""Core task framework - base classes, registry, and configuration."""

from .base import Task, TaskConfig
from .registry import (
    clear_registry,
    get_base_task_name,
    get_task,
    get_task_dependencies,
    list_regimes,
    list_tasks,
    list_variants,
    parse_overrides,
    parse_task_spec,
    register,
    register_regime,
    register_subtasks,
    register_variant,
    task_exists,
)

__all__ = [
    "Task",
    "TaskConfig",
    "clear_registry",
    "get_base_task_name",
    "get_task",
    "get_task_dependencies",
    "list_regimes",
    "list_tasks",
    "list_variants",
    "parse_overrides",
    "parse_task_spec",
    "register",
    "register_regime",
    "register_subtasks",
    "register_variant",
    "task_exists",
]

"""Core task framework - base classes, registry, and configuration."""

from .base import SandboxEnv, Task, TaskConfig
from .format_helpers import format_mc, format_rc
from .registry import (
    clear_registry,
    get_base_task_name,
    get_sandbox_envs,
    get_task,
    get_task_dependencies,
    list_tasks,
    list_variants,
    parse_overrides,
    parse_task_spec,
    register,
    register_subtasks,
    register_variant,
    task_exists,
)

__all__ = [
    "SandboxEnv",
    "Task",
    "TaskConfig",
    "clear_registry",
    "format_mc",
    "format_rc",
    "get_base_task_name",
    "get_sandbox_envs",
    "get_task",
    "get_task_dependencies",
    "list_tasks",
    "list_variants",
    "parse_overrides",
    "parse_task_spec",
    "register",
    "register_subtasks",
    "register_variant",
    "task_exists",
]

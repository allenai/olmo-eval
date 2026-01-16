"""Task framework for evaluation."""

# Re-export from core for backward compatibility
from .core import (
    Task,
    TaskConfig,
    clear_registry,
    get_task,
    list_regimes,
    list_tasks,
    list_variants,
    parse_task_spec,
    register,
    register_regime,
    register_variant,
    task_exists,
)

__all__ = [
    "Task",
    "TaskConfig",
    "clear_registry",
    "get_task",
    "list_regimes",
    "list_tasks",
    "list_variants",
    "parse_task_spec",
    "register",
    "register_regime",
    "register_variant",
    "task_exists",
]

import importlib
import pkgutil
from pathlib import Path


def _discover_and_load_tasks() -> None:
    """Auto-discover and import all task modules to trigger registration."""
    package_dir = Path(__file__).parent

    for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(package_dir)]):
        # Skip the core subpackage and private modules
        if module_name == "core" or module_name.startswith("_"):
            continue

        # Import the module (triggers @register decorators)
        importlib.import_module(f".{module_name}", package=__package__)


# Auto-discover and load all task modules
_discover_and_load_tasks()

# Register common variants after all tasks are registered
from . import _variants  # noqa: F401, E402

# Register common regimes after all tasks and variants are registered
from . import _regimes  # noqa: F401, E402

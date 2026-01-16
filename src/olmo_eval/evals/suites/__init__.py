"""Evaluation suites for benchmarks.

This module provides a registry system for defining and managing collections of
related evaluation tasks. Suites can be nested, allowing complex benchmark
suites to be built from simpler components.

Example usage:
    >>> from olmo_eval.evals.suites import get_suite, list_suites
    >>>
    >>> # Get a specific suite
    >>> mmlu = get_suite("mmlu:mc")
    >>> print(mmlu.tasks)  # All MMLU tasks with MC format
    >>>
    >>> # Expand nested suites to get all individual tasks
    >>> print(mmlu.expand())
    >>>
    >>> # List all registered suites
    >>> for name in list_suites():
    ...     print(name)
"""

# Import and re-export core types and functions
import importlib
import pkgutil
from pathlib import Path

from olmo_eval.evals.suites.registry import (
    AggregationStrategy,
    Suite,
    format_tasks,
    get_suite,
    list_suites,
    make_suite,
    register,
    search_suites,
    suite_exists,
)


def _discover_and_load_suites() -> None:
    """Auto-discover and import all suite modules to trigger registration."""
    package_dir = Path(__file__).parent

    for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(package_dir)]):
        # Skip the registry module and private modules
        if module_name == "registry" or module_name.startswith("_"):
            continue

        # Import the module (triggers suite registration)
        importlib.import_module(f".{module_name}", package=__package__)


# Auto-discover and load all suite modules
_discover_and_load_suites()

__all__ = [
    # Core types
    "AggregationStrategy",
    "Suite",
    # Registry functions
    "get_suite",
    "list_suites",
    "search_suites",
    "suite_exists",
    # Suite creation helpers
    "make_suite",
    "register",
    "format_tasks",
]

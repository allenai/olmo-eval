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

# Import suite definition modules to trigger registration
from olmo_eval.evals.suites import (  # noqa: F401
    code,
    core_tasks,
    long_context,
    math,
    mmlu,
    multiturn,
    olmo,
    reasoning,
)

# Import and re-export core types and functions
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

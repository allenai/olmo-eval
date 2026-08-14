"""Suite types and registry for evaluation benchmarks.

This module provides the foundational types for defining and managing
collections of related evaluation tasks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache


@lru_cache(maxsize=256)
def _expand_tasks(suite: Suite) -> tuple[str, ...]:
    """Recursively expand suite to individual task names (cached)."""
    expanded: list[str] = []
    for task in suite.tasks:
        if isinstance(task, Suite):
            expanded.extend(_expand_tasks(task))
        else:
            expanded.append(task)
    return tuple(expanded)


class AggregationStrategy(StrEnum):
    """How to combine results from tasks in a suite.

    Attributes:
        NONE: No aggregation - just collect individual task results.
        AVERAGE: Compute simple average of all task scores.
        AVERAGE_OF_AVERAGES: Average over child suite averages.
        WEIGHTED_AVERAGE: Average direct children using explicit weights.
        DISPLAY_ONLY: Display child results without computing suite average.
    """

    NONE = "none"
    AVERAGE = "average"
    AVERAGE_OF_AVERAGES = "average_of_averages"
    WEIGHTED_AVERAGE = "weighted_average"
    DISPLAY_ONLY = "display_only"


@dataclass(frozen=True, slots=True)
class Suite:
    """A named collection of evaluation tasks.

    Suites can contain individual task names (strings) or references
    to other suites, enabling hierarchical benchmark definitions.

    Attributes:
        name: Unique identifier for this suite.
        tasks: Tuple of task names or nested Suite references.
        aggregation: Strategy for combining task results.
        description: Optional human-readable description.
        weights: Direct-child weights for ``WEIGHTED_AVERAGE`` suites.
    """

    name: str
    tasks: tuple[str | Suite, ...]
    aggregation: AggregationStrategy = AggregationStrategy.AVERAGE
    description: str = ""
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.aggregation == AggregationStrategy.WEIGHTED_AVERAGE:
            if self.weights is None:
                raise ValueError("weights are required when aggregation='weighted_average'")
            if len(self.weights) != len(self.tasks):
                raise ValueError(
                    f"weights must match tasks one-for-one: got {len(self.weights)} weights for {len(self.tasks)} tasks"
                )
            if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
                raise ValueError("weights must be finite and non-negative")
            if not any(weight > 0 for weight in self.weights):
                raise ValueError("at least one weight must be positive")
        elif self.weights is not None:
            raise ValueError("weights can only be set when aggregation='weighted_average'")

    @property
    def expanded_tasks(self) -> tuple[str, ...]:
        """Recursively expand all nested suites to individual task names."""
        return _expand_tasks(self)

    def expand(self) -> tuple[str, ...]:
        """Alias for expanded_tasks property."""
        return self.expanded_tasks

    def __repr__(self) -> str:
        n_tasks = len(self.tasks)
        n_expanded = len(self.expanded_tasks)
        if n_tasks == n_expanded:
            return f"Suite({self.name!r}, {n_tasks} tasks)"
        return f"Suite({self.name!r}, {n_tasks} items -> {n_expanded} tasks)"


# =============================================================================
# Suite Registry
# =============================================================================

_REGISTRY: dict[str, Suite] = {}


def register(suite: Suite) -> Suite:
    """Register a suite in the global registry.

    Args:
        suite: The Suite to register.

    Returns:
        The same Suite, for chaining.

    Raises:
        ValueError: If a suite with the same name is already registered,
            or if the name collides with a registered task spec.
    """
    if suite.name in _REGISTRY:
        raise ValueError(f"Suite {suite.name!r} is already registered")

    from olmo_eval.evals.tasks.common.registry import task_exists

    if task_exists(suite.name):
        raise ValueError(
            f"Suite name {suite.name!r} collides with a registered task spec. "
            f"Choose a different suite name to avoid ambiguity."
        )

    _REGISTRY[suite.name] = suite
    return suite


def get_suite(name: str) -> Suite:
    """Retrieve a suite by name.

    Args:
        name: The registered name of the suite.

    Returns:
        The Suite instance.

    Raises:
        KeyError: If no suite with that name exists.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Suite {name!r} not found. Use list_suites() to see available suites.")
    return _REGISTRY[name]


def list_suites() -> tuple[str, ...]:
    """Return all registered suite names."""
    return tuple(sorted(_REGISTRY.keys()))


def search_suites(pattern: str | re.Pattern[str]) -> tuple[str, ...]:
    """Search for suites matching a pattern.

    Args:
        pattern: Exact string match or regex pattern.

    Returns:
        Tuple of matching suite names.
    """
    compiled = re.compile(re.escape(pattern)) if isinstance(pattern, str) else pattern
    return tuple(name for name in _REGISTRY if compiled.search(name))


def suite_exists(name: str) -> bool:
    """Check if a suite is registered."""
    return name in _REGISTRY


# =============================================================================
# Helper for creating suites
# =============================================================================


def make_suite(
    name: str,
    tasks: tuple[str | Suite, ...],
    aggregation: AggregationStrategy = AggregationStrategy.AVERAGE,
    description: str = "",
    weights: tuple[float, ...] | None = None,
) -> Suite:
    """Create and register a suite."""
    suite = Suite(name=name, tasks=tasks, aggregation=aggregation, description=description, weights=weights)
    return register(suite)


def format_tasks(
    categories: tuple[str, ...],
    template: str,
) -> tuple[str, ...]:
    """Format task names from categories using a template.

    Args:
        categories: Base category names.
        template: Format string with {} placeholder for category.

    Returns:
        Tuple of formatted task names.
    """
    return tuple(template.format(cat) for cat in categories)
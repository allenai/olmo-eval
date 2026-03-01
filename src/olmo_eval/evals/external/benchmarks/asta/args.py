"""Arguments for ASTA-bench evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

AstaSandboxType = Literal["local", "docker"]
AstaSolver = Literal["react", "basic"]

# Known ASTA-bench tasks by category
ASTA_TASKS = {
    "literature": ["paper_finder", "sqa", "litqa2", "paper_finder_litqa2", "arxivdigestables"],
    "code": ["core_bench", "ds1000", "super"],
    "data_analysis": ["discoverybench"],
    "discovery": ["e2e_discovery", "e2e_discovery_hard"],
}

# Flat set for validation
KNOWN_TASKS = {task for tasks in ASTA_TASKS.values() for task in tasks}


def _parse_optional(data: dict, key: str, type_fn: type) -> Any:
    """Parse an optional value from a dict with type conversion."""
    value = data.get(key)
    return type_fn(value) if value is not None else None


@dataclass
class AstaArgs:
    """Arguments for asta_bench evaluation."""

    # Dataset selection
    split: str = "validation"
    tasks: list[str] | None = None
    limit: int | None = None

    # Agent configuration
    solver: AstaSolver = "react"

    # Parallelism (conservative defaults for memory)
    max_samples: int = 1
    max_sandboxes: int = 1
    max_connections: int = 8

    # Sandbox mode
    sandbox_type: AstaSandboxType = "local"

    # Model overrides
    temperature: float | None = None
    max_tokens: int | None = None

    # Extra inspect args (passed through to inspect eval)
    # Use for task-specific flags like -T with_search_tools=1
    extra_args: list[str] = field(default_factory=list)

    # Trajectory logging
    dump_trajectories: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AstaArgs:
        # Handle tasks which can be comma-separated string or list
        tasks = data.get("tasks")
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]

        # Validate tasks and warn about unknown ones
        if tasks:
            unknown = [t for t in tasks if t not in KNOWN_TASKS]
            if unknown:
                known_list = ", ".join(sorted(KNOWN_TASKS))
                logger.warning(
                    f"Unknown ASTA task(s): {', '.join(unknown)}. Known tasks: {known_list}"
                )

        # Handle extra_args which can be comma-separated string or list
        extra_args = data.get("extra_args", [])
        if isinstance(extra_args, str):
            extra_args = [a.strip() for a in extra_args.split(",") if a.strip()]

        return cls(
            split=data.get("split", "validation"),
            tasks=tasks,
            limit=_parse_optional(data, "limit", int),
            solver=data.get("solver", "react"),
            max_samples=int(data.get("max_samples", 1)),
            max_sandboxes=int(data.get("max_sandboxes", 1)),
            max_connections=int(data.get("max_connections", 8)),
            sandbox_type=data.get("sandbox_type", "local"),
            temperature=_parse_optional(data, "temperature", float),
            max_tokens=_parse_optional(data, "max_tokens", int),
            extra_args=extra_args,
            dump_trajectories=data.get("dump_trajectories", True),
        )

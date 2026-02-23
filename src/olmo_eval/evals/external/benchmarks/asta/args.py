"""Arguments for ASTA-bench evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AstaSandboxType = Literal["local", "docker"]
AstaSolver = Literal["react", "basic"]


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

    # Scoring configuration
    scorer_model: str = "openai/gpt-4o-mini"

    # Extra inspect args (passed through to inspect eval)
    # Use for task-specific flags like -T with_search_tools=1
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AstaArgs:
        # Handle tasks which can be comma-separated string or list
        tasks = data.get("tasks")
        if isinstance(tasks, str):
            tasks = [t.strip() for t in tasks.split(",") if t.strip()]

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
            scorer_model=data.get("scorer_model", "openai/gpt-4o-mini"),
            extra_args=extra_args,
        )

"""Core types for agent execution.

This module defines the result types used by AgentTask
for multi-turn agent evaluations.
"""

from dataclasses import dataclass, field
from typing import Any

from olmo_eval.core.types import AgentTrajectory


@dataclass
class AgentExecutionResult:
    """Result from executing an agent on a single instance.

    Attributes:
        trajectory: The complete agent trajectory with all turns.
        final_answer: The extracted final answer from the agent, if any.
        success: Whether the execution completed without errors.
        error: Error message if execution failed.
        metadata: Additional execution metadata (timing, token counts, etc.).
    """

    trajectory: AgentTrajectory
    final_answer: str | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

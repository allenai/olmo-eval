"""HarnessResult: Result from Harness.run() multi-turn execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.common.types import LMOutput
    from olmo_eval.common.types.trajectory import AgentTrajectory


@dataclass
class HarnessResult:
    """Result from Harness.run() execution.

    Contains the final output and optionally a trajectory of agent-tool
    interactions for multi-turn execution.

    Attributes:
        final_output: The final LMOutput from the last assistant turn.
        trajectory: Complete record of turns (None for single-turn without tools).
        max_turns_reached: Whether execution stopped due to reaching max_turns.
        error: Error message if execution failed.
        metadata: Additional metadata about the execution.
    """

    final_output: LMOutput
    trajectory: AgentTrajectory | None = None
    max_turns_reached: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution completed successfully.

        Returns:
            True if no error and max_turns was not reached.
        """
        return self.error is None and not self.max_turns_reached

    @property
    def final_text(self) -> str:
        """Get the text from the final output.

        Returns:
            Text content of the final output.
        """
        return self.final_output.text

    @property
    def total_tool_calls(self) -> int:
        """Count total tool calls in the trajectory.

        Returns:
            Total number of tool calls made (0 if no trajectory).
        """
        return self.trajectory.total_tool_calls if self.trajectory else 0

    @property
    def num_turns(self) -> int:
        """Get the number of turns in the trajectory.

        Returns:
            Number of turns (1 if no trajectory).
        """
        return self.trajectory.num_turns if self.trajectory else 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the result.
        """
        return {
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "final_output": {
                "text": self.final_output.text,
                "extracted_answer": self.final_output.extracted_answer,
                "metadata": self.final_output.metadata,
                "tool_calls": (
                    [tc.to_dict() for tc in self.final_output.tool_calls]
                    if self.final_output.tool_calls
                    else None
                ),
            },
            "max_turns_reached": self.max_turns_reached,
            "error": self.error,
            "metadata": self.metadata,
        }

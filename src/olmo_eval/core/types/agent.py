"""Nested metric types for agent evaluation.

This module provides structured metric containers for different aspects
of agent evaluation including tool use, trajectory, reliability,
execution, and LLM judge evaluation.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolMetrics:
    """Metrics for tool calling accuracy."""

    call_accuracy: float = 0.0
    argument_accuracy: float = 0.0
    sequence_accuracy: float = 0.0
    num_tool_calls: int = 0
    num_correct_calls: int = 0
    num_correct_arguments: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "call_accuracy": self.call_accuracy,
            "argument_accuracy": self.argument_accuracy,
            "sequence_accuracy": self.sequence_accuracy,
            "num_tool_calls": self.num_tool_calls,
            "num_correct_calls": self.num_correct_calls,
            "num_correct_arguments": self.num_correct_arguments,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryMetrics:
    """Metrics for trajectory evaluation."""

    response_check: float = 0.0  # Did the tool call sequence match?
    state_check: float = 0.0  # Did the final state match?
    efficiency: float = 0.0  # minimal_steps / actual_steps
    total_steps: int = 0
    minimal_steps: int = 0
    combined_score: float = 0.0  # Both response AND state passed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "response_check": self.response_check,
            "state_check": self.state_check,
            "efficiency": self.efficiency,
            "total_steps": self.total_steps,
            "minimal_steps": self.minimal_steps,
            "combined_score": self.combined_score,
        }


@dataclass(frozen=True, slots=True)
class ReliabilityMetrics:
    """Metrics for multi-trial reliability evaluation."""

    num_trials: int = 0
    pass_at_k: float = 0.0  # At least one success in k trials
    pass_pow_k: float = 0.0  # All k trials succeed
    k: int = 1
    success_count: int = 0
    failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_trials": self.num_trials,
            "pass_at_k": self.pass_at_k,
            "pass_pow_k": self.pass_pow_k,
            "k": self.k,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
        }


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    """Metrics for task execution."""

    total_runs: int = 0
    successful_runs: int = 0
    processing_errors: int = 0
    instruction_errors: int = 0
    timeout_errors: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "processing_errors": self.processing_errors,
            "instruction_errors": self.instruction_errors,
            "timeout_errors": self.timeout_errors,
            "success_rate": self.success_rate,
        }


@dataclass(frozen=True, slots=True)
class JudgeMetrics:
    """Metrics from LLM-as-judge evaluation."""

    accuracy: float = 0.0
    not_attempted_rate: float = 0.0
    judge_model: str = ""
    grade_distribution: dict[str, int] = field(default_factory=dict)
    num_evaluated: int = 0
    num_correct: int = 0
    num_incorrect: int = 0
    num_not_attempted: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "not_attempted_rate": self.not_attempted_rate,
            "judge_model": self.judge_model,
            "grade_distribution": dict(self.grade_distribution),
            "num_evaluated": self.num_evaluated,
            "num_correct": self.num_correct,
            "num_incorrect": self.num_incorrect,
            "num_not_attempted": self.num_not_attempted,
        }


@dataclass(frozen=True, slots=True)
class AgentMetrics:
    """Container for all agent evaluation metrics.

    This is the top-level metrics container that can be added to
    StoredTaskResult for agent evaluation tasks.
    """

    tool: ToolMetrics | None = None
    trajectory: TrajectoryMetrics | None = None
    reliability: ReliabilityMetrics | None = None
    execution: ExecutionMetrics | None = None
    judge: JudgeMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result: dict[str, Any] = {}
        if self.tool is not None:
            result["tool"] = self.tool.to_dict()
        if self.trajectory is not None:
            result["trajectory"] = self.trajectory.to_dict()
        if self.reliability is not None:
            result["reliability"] = self.reliability.to_dict()
        if self.execution is not None:
            result["execution"] = self.execution.to_dict()
        if self.judge is not None:
            result["judge"] = self.judge.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentMetrics":
        """Create from dictionary.

        Args:
            data: Dictionary with metric data.

        Returns:
            A new AgentMetrics instance.
        """
        tool = ToolMetrics(**data["tool"]) if "tool" in data else None
        trajectory = TrajectoryMetrics(**data["trajectory"]) if "trajectory" in data else None
        reliability = ReliabilityMetrics(**data["reliability"]) if "reliability" in data else None
        execution = ExecutionMetrics(**data["execution"]) if "execution" in data else None

        judge = None
        if "judge" in data:
            judge_data = data["judge"].copy()
            judge = JudgeMetrics(**judge_data)

        return cls(
            tool=tool,
            trajectory=trajectory,
            reliability=reliability,
            execution=execution,
            judge=judge,
        )

"""Core types for agent execution.

This module defines the configuration and result types used by AgentTask
for multi-turn agent evaluations.
"""

from dataclasses import dataclass, field
from typing import Any

from olmo_eval.core.types import AgentTrajectory


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for agent execution.

    Attributes:
        model: The model identifier to use for agent inference.
        model_url: The API endpoint URL for the model.
        system_prompt: Optional system prompt for the agent.
        max_turns: Maximum number of agent turns before stopping.
        max_concurrency: Maximum number of concurrent agent executions.
        temperature: Sampling temperature for agent responses.
        max_tokens: Maximum tokens per agent response.
    """

    model: str
    model_url: str = ""
    system_prompt: str = ""
    max_turns: int = 10
    max_concurrency: int = 1
    temperature: float = 0.0
    max_tokens: int = 4096


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

"""Agent execution infrastructure for multi-turn evaluations.

This module provides types and utilities for running agent-based evaluations
that involve multiple turns of interaction with tools.
"""

from .executor import AgentExecutionResult

__all__ = [
    "AgentExecutionResult",
]

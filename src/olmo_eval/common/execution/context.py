"""Scoring context for passing execution environment to scorers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import ExecutionEnvironment


@dataclass
class ScoringContext:
    """Context passed to scorers during evaluation.

    This provides access to shared resources like execution environments
    that scorers may need. The context is created by the worker and passed
    through the scoring chain.

    Attributes:
        execution_env: Optional execution environment for code execution.
            If None, scorers requiring execution will fall back to unsafe
            local execution or raise an error.
    """

    execution_env: ExecutionEnvironment | None = None

    @property
    def has_execution_env(self) -> bool:
        """Check if an execution environment is available."""
        return self.execution_env is not None and self.execution_env.is_running

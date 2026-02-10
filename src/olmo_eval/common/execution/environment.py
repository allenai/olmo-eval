"""Execution environment protocol for sandboxed code execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecutionResult:
    """Result from executing code in an environment.

    Attributes:
        success: Whether execution completed without errors.
        output: Combined stdout/stderr from execution.
        exit_code: Exit code from the execution (0 = success).
        error: Error message if execution failed to start.
    """

    success: bool
    output: str = ""
    exit_code: int = 0
    error: str | None = None


@runtime_checkable
class ExecutionEnvironment(Protocol):
    """Protocol for code execution environments.

    This abstraction allows scorers to execute code without depending
    on specific sandbox implementations. Implementations include:
    - SandboxExecutor (Docker/Podman via SWE-ReX)
    - LocalExecutionEnvironment (unsafe, for testing only)

    The environment must be started before use and stopped after.
    """

    @property
    def is_running(self) -> bool:
        """Check if the environment is ready for execution."""
        ...

    async def execute(self, command: str, timeout: float | None = None) -> str:
        """Execute a command and return the output.

        Args:
            command: Shell command to execute.
            timeout: Optional timeout in seconds.

        Returns:
            Command output (stdout + stderr combined).
        """
        ...

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute code in the specified language.

        This is a higher-level method that handles writing code to a file,
        executing it, and capturing the result.

        Args:
            code: Source code to execute.
            language: Programming language (default: "python").
            timeout: Optional timeout in seconds.

        Returns:
            ExecutionResult with success status and output.
        """
        ...

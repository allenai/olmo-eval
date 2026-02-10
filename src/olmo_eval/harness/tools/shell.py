"""Shell execution tools for sandboxed command execution.

This module provides tools for executing shell commands in a sandboxed
environment. These tools are marked with requires_sandbox=True and
are executed via the SandboxExecutor.
"""

from __future__ import annotations

from .registry import registered_tool


@registered_tool(
    name="execute_bash",
    description="Execute a bash command in a sandboxed environment and return the output.",
    requires_sandbox=True,
)
async def execute_bash(command: str) -> str:
    """Execute a bash command in a sandboxed environment.

    This tool executes arbitrary bash commands in an isolated container
    environment. Use this to run code, install packages, manipulate files,
    and perform other shell operations.

    Args:
        command: The bash command to execute.

    Returns:
        The command output (stdout + stderr combined).

    Note:
        This is a placeholder implementation. Actual execution is delegated
        to the SandboxExecutor by the backend when sandbox is enabled.
    """
    raise NotImplementedError(
        "execute_bash requires sandbox execution. Ensure sandbox is enabled in HarnessConfig."
    )

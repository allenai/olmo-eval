"""Shell execution tools for sandboxed command execution.

This module provides tools for executing shell commands in a sandboxed
environment. These tools specify capabilities and are executed via the
SandboxManager.
"""

from __future__ import annotations

from olmo_eval.harness.sandbox import Capability

from .registry import registered_tool


@registered_tool(
    name="execute_bash",
    description="Execute a bash command in a sandboxed environment and return the output.",
    sandbox=Capability.BASH,
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


@registered_tool(
    name="execute_bash_session",
    description="Execute a bash command in a persistent shell session. "
    "Working directory changes, exported variables, and aliases persist between calls.",
    sandbox=Capability.BASH,
    session=True,
)
async def execute_bash_session(command: str) -> str:
    """Execute a bash command in a persistent shell session.

    Unlike execute_bash, this tool maintains shell state between calls:
    - Working directory changes persist (cd /tmp stays in /tmp)
    - Exported variables persist (export FOO=bar available in later calls)
    - Aliases and shell functions persist

    Use this when you need to build up shell state across multiple commands.

    Args:
        command: The bash command to execute.

    Returns:
        The command output (stdout + stderr combined).
    """
    raise NotImplementedError(
        "execute_bash_session requires sandbox execution. "
        "Ensure sandbox is enabled in HarnessConfig."
    )

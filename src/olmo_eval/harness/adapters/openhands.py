"""SWE-ReX → OpenHands SDK Tool Adapter.

This module provides a bridge between SWE-ReX runtimes and the OpenHands SDK,
enabling OpenHands agents to execute commands in SWE-ReX managed containers.

Usage:
    import asyncio
    from swerex.deployment.docker import DockerDeployment
    from openhands.sdk import LLM, Conversation

    from olmo_eval.harness.adapters.openhands import create_swerex_agent

    async def main():
        # Start Docker deployment
        deployment = DockerDeployment(image="python:3.11")
        await deployment.start()

        # Create agent with SWE-ReX backend
        llm = LLM(model="anthropic/claude-sonnet-4-20250514", api_key="...")
        agent = create_swerex_agent(llm, deployment.runtime)

        # Run conversation
        conversation = Conversation(agent=agent, workspace="/workspace")
        conversation.send_message("Create and run hello.py")
        conversation.run()

        # Cleanup
        await deployment.stop()

    if __name__ == "__main__":
        asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from collections.abc import Coroutine, Sequence
from typing import TYPE_CHECKING

from openhands.sdk import LLM, Agent, Tool
from openhands.sdk.tool import (
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.terminal.definition import (
    TOOL_DESCRIPTION,
    TerminalAction,
    TerminalObservation,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState
    from swerex.runtime.abstract import AbstractRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "SweRexTerminalExecutor",
    "SweRexTerminalTool",
    "create_swerex_tools",
    "register_swerex_tools",
    "create_swerex_agent",
]

# Counter for unique tool registration names
_tool_registration_counter = 0


def _run_coroutine_sync[T](coro: Coroutine[None, None, T]) -> T:
    """Run a coroutine synchronously, handling nested event loop scenarios.

    When called from outside an event loop, uses asyncio.run().
    When called from within a running event loop (e.g., from an async context),
    runs the coroutine in a separate thread to avoid the "cannot call asyncio.run()
    from a running event loop" error.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - safe to use asyncio.run()
        return asyncio.run(coro)

    # Already in an event loop - run in a thread pool with its own loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


class SweRexTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Terminal executor that delegates to a SWE-ReX runtime.

    This executor wraps a SWE-ReX AbstractRuntime to execute bash commands,
    bridging the OpenHands SDK's terminal tool interface with SWE-ReX's
    session-based command execution.

    The executor manages bash session lifecycle lazily - the session is created
    on first command execution and persists across subsequent calls, maintaining
    shell state (cd, export, aliases, etc.).
    """

    def __init__(
        self,
        runtime: AbstractRuntime,
        session_name: str = "default",
        timeout: float = 120.0,
        working_dir: str | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            runtime: SWE-ReX runtime instance (from a started deployment).
            session_name: Name for the bash session (default: "default").
            timeout: Default command timeout in seconds (default: 120).
            working_dir: Initial working directory (optional).
        """
        self._runtime = runtime
        self._session_name = session_name
        self._timeout = timeout
        self._working_dir = working_dir
        self._session_created = False
        self._session_lock = asyncio.Lock()

    async def _ensure_session(self) -> None:
        """Create the bash session if it doesn't exist."""
        if self._session_created:
            return

        async with self._session_lock:
            if self._session_created:
                return

            from swerex.runtime.abstract import CreateBashSessionRequest

            await self._runtime.create_session(CreateBashSessionRequest(session=self._session_name))
            self._session_created = True

            # Set initial working directory if specified
            if self._working_dir:
                from swerex.runtime.abstract import BashAction

                await self._runtime.run_in_session(
                    BashAction(
                        command=f"cd {self._working_dir}",
                        session=self._session_name,
                        timeout=10.0,
                        check="silent",
                    )
                )

    async def _close_session(self) -> None:
        """Close the bash session."""
        if not self._session_created:
            return

        async with self._session_lock:
            if not self._session_created:
                return

            try:
                from swerex.runtime.abstract import CloseBashSessionRequest

                await self._runtime.close_session(
                    CloseBashSessionRequest(session=self._session_name)
                )
            except Exception as e:
                logger.debug(f"Failed to close session: {e}")
            finally:
                self._session_created = False

    async def _reset_session(self) -> None:
        """Close and recreate the session."""
        await self._close_session()
        await self._ensure_session()

    def __call__(
        self,
        action: TerminalAction,
        conversation: LocalConversation | None = None,
    ) -> TerminalObservation:
        """Execute a terminal action synchronously.

        OpenHands executors are called from sync context, so we bridge to
        the async SWE-ReX runtime. Handles the case where we're already
        inside an event loop (e.g., when called from an async context).

        Args:
            action: The terminal action to execute.
            conversation: Conversation context (unused, for interface compatibility).

        Returns:
            TerminalObservation with command output and exit code.
        """
        return _run_coroutine_sync(self._execute_async(action))

    async def _execute_async(self, action: TerminalAction) -> TerminalObservation:
        """Execute the terminal action asynchronously.

        Args:
            action: The terminal action to execute.

        Returns:
            TerminalObservation with command output and exit code.
        """
        from swerex.runtime.abstract import BashAction

        # Handle special cases
        if action.command == "C-c":
            return TerminalObservation.from_text(
                text="[Interrupt (C-c) not directly supported in SWE-ReX sessions. "
                "Consider using 'kill' command or starting a new session.]",
                command=action.command,
                exit_code=0,
            )

        if getattr(action, "is_input", False):
            return TerminalObservation.from_text(
                text="[Interactive input not supported. SWE-ReX run_in_session "
                "does not support sending input to running processes.]",
                command=action.command,
                exit_code=1,
                is_error=True,
            )

        if getattr(action, "reset", False):
            await self._reset_session()
            return TerminalObservation.from_text(
                text="[Session reset successfully]",
                command=action.command,
                exit_code=0,
            )

        # Ensure session exists
        await self._ensure_session()

        # Get timeout from action or use default
        timeout = getattr(action, "timeout", None) or self._timeout

        try:
            observation = await self._runtime.run_in_session(
                BashAction(
                    command=action.command,
                    session=self._session_name,
                    timeout=timeout,
                    check="silent",
                )
            )

            output = observation.output or ""
            exit_code = observation.exit_code if observation.exit_code is not None else 0

            # Check for timeout in failure reason
            if observation.failure_reason and "timeout" in observation.failure_reason.lower():
                return TerminalObservation.from_text(
                    text=f"{output}\n[Command timed out after {timeout}s]",
                    command=action.command,
                    exit_code=-1,
                    timeout=True,
                )

            return TerminalObservation.from_text(
                text=output,
                command=action.command,
                exit_code=exit_code,
            )

        except TimeoutError:
            return TerminalObservation.from_text(
                text=f"[Command timed out after {timeout}s]",
                command=action.command,
                exit_code=-1,
                timeout=True,
            )
        except Exception as e:
            logger.error(f"SWE-ReX execution error: {e}")
            return TerminalObservation.from_text(
                text=f"[Infrastructure error: {e}]",
                command=action.command,
                exit_code=-1,
                is_error=True,
            )


class SweRexTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """Terminal tool backed by a SWE-ReX runtime.

    This is a concrete implementation of ToolDefinition that executes terminal
    commands via a SWE-ReX runtime instead of local shell execution.
    """

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        executor: SweRexTerminalExecutor | None = None,
        runtime: AbstractRuntime | None = None,
        working_dir: str = "/workspace",
        session_name: str = "default",
        timeout: float = 120.0,
    ) -> Sequence[SweRexTerminalTool]:
        """Create SweRexTerminalTool instances.

        Can be called either with a pre-created executor, or with runtime
        parameters to create a new executor.

        Args:
            conv_state: Conversation state (unused, for interface compatibility).
            executor: Pre-created SweRexTerminalExecutor (takes precedence).
            runtime: SWE-ReX runtime to create executor from.
            working_dir: Initial working directory.
            session_name: Name for the bash session.
            timeout: Default command timeout in seconds.

        Returns:
            List containing a single SweRexTerminalTool instance.
        """
        if executor is None:
            if runtime is None:
                raise ValueError("Either executor or runtime must be provided")
            executor = SweRexTerminalExecutor(
                runtime=runtime,
                session_name=session_name,
                timeout=timeout,
                working_dir=working_dir,
            )

        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=TerminalAction,
                observation_type=TerminalObservation,
                annotations=ToolAnnotations(
                    title="terminal",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=executor,
            )
        ]


def _create_swerex_terminal_tool(
    executor: SweRexTerminalExecutor,
) -> SweRexTerminalTool:
    """Create a SWE-ReX terminal tool instance.

    Args:
        executor: The SWE-ReX terminal executor.

    Returns:
        SweRexTerminalTool configured for terminal execution.
    """
    return SweRexTerminalTool(
        description=TOOL_DESCRIPTION,
        action_type=TerminalAction,
        observation_type=TerminalObservation,
        annotations=ToolAnnotations(
            title="terminal",
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=True,
        ),
        executor=executor,
    )


def create_swerex_tools(
    runtime: AbstractRuntime,
    working_dir: str = "/workspace",
    session_name: str = "default",
    timeout: float = 120.0,
) -> list[SweRexTerminalTool]:
    """Create OpenHands tools backed by a SWE-ReX runtime.

    This factory function creates ready-to-use tool instances without
    requiring a ConversationState.

    Args:
        runtime: SWE-ReX runtime instance (from a started deployment).
        working_dir: Initial working directory (default: "/workspace").
        session_name: Name for the bash session (default: "default").
        timeout: Default command timeout in seconds (default: 120).

    Returns:
        List of tool definitions ready for use with OpenHands Agent.

    Note:
        Currently only provides TerminalTool. File editing can be done
        via terminal commands (cat, echo, sed) in the sandbox.
    """
    executor = SweRexTerminalExecutor(
        runtime=runtime,
        session_name=session_name,
        timeout=timeout,
        working_dir=working_dir,
    )
    return [_create_swerex_terminal_tool(executor)]


def register_swerex_tools(
    runtime: AbstractRuntime,
    working_dir: str = "/workspace",
    session_name: str = "default",
    timeout: float = 120.0,
) -> str:
    """Register SWE-ReX tools with the OpenHands tool registry.

    This function creates SWE-ReX-backed tools and registers them with
    OpenHands' tool registry, returning the registered tool name that
    can be used with Agent(tools=[Tool(name=...)]).

    Args:
        runtime: SWE-ReX runtime instance (from a started deployment).
        working_dir: Initial working directory (default: "/workspace").
        session_name: Name for the bash session (default: "default").
        timeout: Default command timeout in seconds (default: 120).

    Returns:
        The registered tool name to use with Tool(name=...).
    """
    global _tool_registration_counter
    _tool_registration_counter += 1
    tool_name = f"SweRexTerminal_{_tool_registration_counter}"

    # Create the executor once, captured by the factory closure
    executor = SweRexTerminalExecutor(
        runtime=runtime,
        session_name=session_name,
        timeout=timeout,
        working_dir=working_dir,
    )

    def tool_factory(
        conv_state: ConversationState,
    ) -> list[SweRexTerminalTool]:
        """Factory function for OpenHands tool registry."""
        return [_create_swerex_terminal_tool(executor)]

    register_tool(tool_name, tool_factory)
    return tool_name


def create_swerex_agent(
    llm: LLM,
    runtime: AbstractRuntime,
    working_dir: str = "/workspace",
    session_name: str = "default",
    timeout: float = 120.0,
) -> Agent:
    """Create an OpenHands Agent with SWE-ReX-backed tools.

    This factory function creates a fully configured OpenHands Agent
    that executes commands in a SWE-ReX managed container.

    Args:
        llm: OpenHands LLM instance configured with model and API settings.
        runtime: SWE-ReX runtime instance (from a started deployment).
        working_dir: Initial working directory (default: "/workspace").
        session_name: Name for the bash session (default: "default").
        timeout: Default command timeout in seconds (default: 120).

    Returns:
        Configured OpenHands Agent instance.

    Example:
        >>> from swerex.deployment.docker import DockerDeployment
        >>> from openhands.sdk import LLM, Conversation
        >>>
        >>> async def main():
        ...     deployment = DockerDeployment(image="python:3.11")
        ...     await deployment.start()
        ...
        ...     llm = LLM(model="anthropic/claude-sonnet-4-20250514", api_key="...")
        ...     agent = create_swerex_agent(llm, deployment.runtime)
        ...
        ...     conv = Conversation(agent=agent, workspace="/workspace")
        ...     conv.send_message("List files")
        ...     conv.run()
        ...
        ...     await deployment.stop()
    """
    # Register tools and get the tool name
    tool_name = register_swerex_tools(
        runtime=runtime,
        working_dir=working_dir,
        session_name=session_name,
        timeout=timeout,
    )

    return Agent(
        llm=llm,
        tools=[Tool(name=tool_name)],
    )

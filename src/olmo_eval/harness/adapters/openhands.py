"""OpenHands SDK Tool Adapter.

This module provides a bridge between sandbox runtimes and the OpenHands SDK,
enabling OpenHands agents to execute commands in sandbox-managed containers.

Usage:
    import asyncio
    from swerex.deployment.docker import DockerDeployment
    from openhands.sdk import LLM, Conversation

    from olmo_eval.harness.adapters.openhands import create_sandbox_agent

    async def main():
        # Start Docker deployment
        deployment = DockerDeployment(image="python:3.11")
        await deployment.start()

        # Create agent with sandbox backend
        llm = LLM(model="anthropic/claude-sonnet-4-20250514", api_key="...")
        agent = create_sandbox_agent(llm, deployment.runtime)

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
from typing import TYPE_CHECKING, Any

from openhands.sdk import (  # type: ignore[ty:unresolved-import]
    LLM,
    Action,
    Agent,
    ImageContent,
    Observation,
    TextContent,
    Tool,
)
from openhands.sdk.tool import (  # type: ignore[ty:unresolved-import]
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.terminal.definition import (  # type: ignore[ty:unresolved-import]
    TOOL_DESCRIPTION,
    TerminalAction,
    TerminalObservation,
)
from openhands.tools.terminal.metadata import (  # type: ignore[ty:unresolved-import]
    CmdOutputMetadata,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.impl.local_conversation import (  # type: ignore[ty:unresolved-import]
        LocalConversation,
    )
    from openhands.sdk.conversation.state import (  # type: ignore[ty:unresolved-import]
        ConversationState,
    )
    from swerex.runtime.abstract import AbstractRuntime  # type: ignore[ty:unresolved-import]

    from olmo_eval.harness.tools import Tool as HarnessTool

logger = logging.getLogger(__name__)

__all__ = [
    "SandboxTerminalExecutor",
    "SandboxTerminalTool",
    "create_sandbox_tools",
    "register_sandbox_tools",
    "create_sandbox_agent",
    "HarnessToolObservation",
    "HarnessToolExecutor",
    "HarnessToolDefinition",
]


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
    def _run() -> T:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


class SandboxTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    """Terminal executor that delegates to a sandbox runtime.

    This executor wraps a sandbox AbstractRuntime to execute bash commands,
    bridging the OpenHands SDK's terminal tool interface with the sandbox's
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
            runtime: Sandbox runtime instance (from a started deployment).
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

            from swerex.runtime.abstract import (  # type: ignore[ty:unresolved-import]
                CreateBashSessionRequest,
            )

            await self._runtime.create_session(CreateBashSessionRequest(session=self._session_name))
            self._session_created = True

            # Set initial working directory if specified
            if self._working_dir:
                from swerex.runtime.abstract import BashAction  # type: ignore[ty:unresolved-import]

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
                from swerex.runtime.abstract import (  # type: ignore[ty:unresolved-import]
                    CloseBashSessionRequest,
                )

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
        the async sandbox runtime. Handles the case where we're already
        inside an event loop (e.g., when called from an async context).

        Args:
            action: The terminal action to execute.
            conversation: Conversation context (unused, for interface compatibility).

        Returns:
            TerminalObservation with command output and exit code.
        """
        return _run_coroutine_sync(self._execute_async(action))

    def _make_observation(
        self,
        text: str,
        command: str,
        exit_code: int,
        is_error: bool = False,
        timeout: bool = False,
        working_dir: str | None = None,
    ) -> TerminalObservation:
        """Create a TerminalObservation with properly populated metadata.

        Args:
            text: The output text from the command.
            command: The command that was executed.
            exit_code: The exit code of the command.
            is_error: Whether this observation represents an error.
            timeout: Whether the command timed out.
            working_dir: The working directory (defaults to self._working_dir).

        Returns:
            TerminalObservation with metadata populated.
        """
        metadata = CmdOutputMetadata(
            exit_code=exit_code,
            working_dir=working_dir or self._working_dir,
        )
        return TerminalObservation(
            content=[{"type": "text", "text": text}],
            command=command,
            exit_code=exit_code,
            is_error=is_error,
            timeout=timeout,
            metadata=metadata,
        )

    async def _execute_async(self, action: TerminalAction) -> TerminalObservation:
        """Execute the terminal action asynchronously.

        Args:
            action: The terminal action to execute.

        Returns:
            TerminalObservation with command output and exit code.
        """
        from swerex.runtime.abstract import BashAction  # type: ignore[ty:unresolved-import]

        # Handle special cases
        if action.command == "C-c":
            return self._make_observation(
                text="[Interrupt (C-c) not directly supported in sandbox sessions. "
                "Consider using 'kill' command or starting a new session.]",
                command=action.command,
                exit_code=0,
            )

        if getattr(action, "is_input", False):
            return self._make_observation(
                text="[Interactive input not supported. Sandbox run_in_session "
                "does not support sending input to running processes.]",
                command=action.command,
                exit_code=1,
                is_error=True,
            )

        if getattr(action, "reset", False):
            await self._reset_session()
            return self._make_observation(
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
                return self._make_observation(
                    text=f"{output}\n[Command timed out after {timeout}s]",
                    command=action.command,
                    exit_code=-1,
                    timeout=True,
                )

            return self._make_observation(
                text=output,
                command=action.command,
                exit_code=exit_code,
            )

        except TimeoutError:
            return self._make_observation(
                text=f"[Command timed out after {timeout}s]",
                command=action.command,
                exit_code=-1,
                timeout=True,
            )
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            return self._make_observation(
                text=f"[Infrastructure error: {e}]",
                command=action.command,
                exit_code=-1,
                is_error=True,
            )


class SandboxTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    """Terminal tool backed by a sandbox runtime.

    This is a concrete implementation of ToolDefinition that executes terminal
    commands via a sandbox runtime instead of local shell execution.
    """

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        executor: SandboxTerminalExecutor | None = None,
        runtime: AbstractRuntime | None = None,
        working_dir: str = "/workspace",
        session_name: str = "default",
        timeout: float = 120.0,
    ) -> Sequence[SandboxTerminalTool]:
        """Create SandboxTerminalTool instances.

        Can be called either with a pre-created executor, or with runtime
        parameters to create a new executor.

        Args:
            conv_state: Conversation state (unused, for interface compatibility).
            executor: Pre-created SandboxTerminalExecutor (takes precedence).
            runtime: Sandbox runtime to create executor from.
            working_dir: Initial working directory.
            session_name: Name for the bash session.
            timeout: Default command timeout in seconds.

        Returns:
            List containing a single SandboxTerminalTool instance.
        """
        if executor is None:
            if runtime is None:
                raise ValueError("Either executor or runtime must be provided")
            executor = SandboxTerminalExecutor(
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


def _create_sandbox_terminal_tool(
    executor: SandboxTerminalExecutor,
) -> SandboxTerminalTool:
    """Create a sandbox terminal tool instance.

    Args:
        executor: The sandbox terminal executor.

    Returns:
        SandboxTerminalTool configured for terminal execution.
    """
    return SandboxTerminalTool(
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


def create_sandbox_tools(
    runtime: AbstractRuntime,
    working_dir: str = "/workspace",
    session_name: str = "default",
    timeout: float = 120.0,
) -> list[SandboxTerminalTool]:
    """Create OpenHands tools backed by a sandbox runtime.

    This factory function creates ready-to-use tool instances without
    requiring a ConversationState.

    Args:
        runtime: Sandbox runtime instance (from a started deployment).
        working_dir: Initial working directory (default: "/workspace").
        session_name: Name for the bash session (default: "default").
        timeout: Default command timeout in seconds (default: 120).

    Returns:
        List of tool definitions ready for use with OpenHands Agent.

    Note:
        Currently only provides TerminalTool. File editing can be done
        via terminal commands (cat, echo, sed) in the sandbox.
    """
    executor = SandboxTerminalExecutor(
        runtime=runtime,
        session_name=session_name,
        timeout=timeout,
        working_dir=working_dir,
    )
    return [_create_sandbox_terminal_tool(executor)]


def register_sandbox_tools(
    runtime: AbstractRuntime,
    working_dir: str = "/workspace",
    session_name: str = "default",
    timeout: float = 120.0,
) -> str:
    """Register sandbox tools with the OpenHands tool registry.

    This function creates sandbox-backed tools and registers them with
    OpenHands' tool registry, returning the registered tool name that
    can be used with Agent(tools=[Tool(name=...)]).

    Args:
        runtime: Sandbox runtime instance (from a started deployment).
        working_dir: Initial working directory (default: "/workspace").
        session_name: Name for the bash session (default: "default").
        timeout: Default command timeout in seconds (default: 120).

    Returns:
        The registered tool name to use with Tool(name=...).
    """
    executor = SandboxTerminalExecutor(
        runtime=runtime,
        session_name=session_name,
        timeout=timeout,
        working_dir=working_dir,
    )

    tool_name = f"SandboxTerminal_{id(executor)}"

    def tool_factory(
        conv_state: ConversationState,
    ) -> list[SandboxTerminalTool]:
        """Factory function for OpenHands tool registry."""
        return [_create_sandbox_terminal_tool(executor)]

    register_tool(tool_name, tool_factory)
    return tool_name


# ---------------------------------------------------------------------------
# Harness Tool Translation
# ---------------------------------------------------------------------------


def _json_type_to_python(json_type: str) -> type:
    """Convert JSON schema type to Python type for Pydantic field."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, str)


def _create_action_class_for_tool(tool: HarnessTool) -> type[Action]:
    """Dynamically create an Action subclass with fields matching the tool's parameters.

    OpenHands generates tool schemas from Action class fields, so we need
    action classes with the correct parameter names and types.
    """
    from pydantic import Field, create_model  # type: ignore[ty:unresolved-import]

    field_definitions: dict[str, Any] = {}
    properties = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))

    for param_name, param_schema in properties.items():
        python_type = _json_type_to_python(param_schema.get("type", "string"))
        description = param_schema.get("description", "")

        if param_name in required:
            # Required field
            field_definitions[param_name] = (python_type, Field(description=description))
        else:
            # Optional field with default
            default = param_schema.get("default", None)
            field_definitions[param_name] = (
                python_type | None,
                Field(default=default, description=description),
            )

    # Create dynamic Pydantic model inheriting from Action
    action_class = create_model(
        f"{tool.name}Action",
        __base__=Action,
        **field_definitions,
    )
    return action_class


class HarnessToolObservation(Observation):
    """Observation wrapping harness tool string result."""

    content: str
    is_error: bool = False

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        """Convert observation to LLM-consumable content."""
        return [TextContent(type="text", text=self.content)]


class HarnessToolExecutor(ToolExecutor[Action, HarnessToolObservation]):
    """Executor that bridges harness Tool to OpenHands execution model."""

    def __init__(
        self,
        tool: HarnessTool,
        sandbox_executor: SandboxTerminalExecutor | None = None,
    ) -> None:
        self._tool = tool
        self._sandbox_executor = sandbox_executor

    def _extract_arguments(self, action: Action) -> dict[str, Any]:
        """Extract tool arguments from a dynamic action class instance."""
        # Get all fields from the action that aren't inherited from Action base
        base_fields = set(Action.model_fields.keys())
        return {
            name: getattr(action, name)
            for name in type(action).model_fields
            if name not in base_fields
        }

    def __call__(
        self,
        action: Action,
        conversation: LocalConversation | None = None,
    ) -> HarnessToolObservation:
        """Execute the harness tool."""
        try:
            arguments = self._extract_arguments(action)
            if self._tool.sandbox and self._sandbox_executor:
                # Execute via sandbox
                result = self._execute_in_sandbox(arguments)
            else:
                # Direct execution
                result = self._tool.execute(**arguments)
                if asyncio.iscoroutine(result):
                    result = _run_coroutine_sync(result)
            return HarnessToolObservation(content=str(result))
        except Exception as e:
            logger.error(f"Harness tool {self._tool.name} failed: {e}")
            return HarnessToolObservation(content=str(e), is_error=True)

    def _execute_in_sandbox(self, arguments: dict[str, Any]) -> str:
        """Execute tool via sandbox runtime."""
        # The harness bash tools expect 'command' argument
        command = arguments.get("command", "")
        action = TerminalAction(command=command)
        obs = self._sandbox_executor(action)
        if not obs.content:
            return ""
        item = obs.content[0]
        # Handle both dict and TextContent object forms
        if isinstance(item, dict):
            return item.get("text", "")
        return getattr(item, "text", str(item))


class HarnessToolDefinition(ToolDefinition[Action, HarnessToolObservation]):
    """OpenHands ToolDefinition wrapping a harness Tool."""

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
    ) -> Sequence[HarnessToolDefinition]:
        """Create method required by ToolDefinition.

        This is not used directly - use from_tool() instead.
        """
        raise NotImplementedError("Use HarnessToolDefinition.from_tool() instead")

    @classmethod
    def from_tool(
        cls,
        tool: HarnessTool,
        sandbox_executor: SandboxTerminalExecutor | None = None,
    ) -> HarnessToolDefinition:
        """Create from a harness Tool.

        Args:
            tool: Harness Tool instance.
            sandbox_executor: Executor for sandbox tools.

        Returns:
            HarnessToolDefinition wrapping the harness tool.
        """
        if tool.sandbox and sandbox_executor is None:
            raise ValueError(f"Sandbox executor required for tool {tool.name}")

        executor = HarnessToolExecutor(tool, sandbox_executor)

        # Create dynamic action class with correct parameter fields
        action_class = _create_action_class_for_tool(tool)

        # OpenHands derives tool names from class.__name__, so create a unique subclass.
        tool_class = type(tool.name, (cls,), {})

        return tool_class(
            description=tool.description,
            action_type=action_class,
            observation_type=HarnessToolObservation,
            annotations=ToolAnnotations(
                title=tool.name,
                readOnlyHint=not bool(tool.sandbox),
                destructiveHint=bool(tool.sandbox),
                idempotentHint=False,
                openWorldHint=True,
            ),
            executor=executor,
        )


def _register_harness_tool(
    tool: HarnessTool,
    sandbox_executor: SandboxTerminalExecutor | None = None,
) -> str:
    """Register a harness tool with OpenHands registry.

    Args:
        tool: Harness Tool to register.
        sandbox_executor: Executor for sandbox tools (required if tool.sandbox).

    Returns:
        Registered tool name.
    """
    tool_def = HarnessToolDefinition.from_tool(tool, sandbox_executor)
    tool_name = f"{tool.name}_{id(tool_def)}"

    def tool_factory(conv_state: ConversationState) -> list[ToolDefinition]:
        return [tool_def]

    register_tool(tool_name, tool_factory)
    return tool_name


def create_sandbox_agent(
    llm: LLM,
    runtime: AbstractRuntime | None = None,
    working_dir: str = "/workspace",
    session_name: str = "default",
    timeout: float = 120.0,
    system_prompt: str | None = None,
    harness_tools: tuple[HarnessTool, ...] | None = None,
) -> Agent:
    """Create an OpenHands Agent with the specified tools.

    Args:
        llm: OpenHands LLM instance.
        runtime: Sandbox runtime (required if any tools need sandbox).
        working_dir: Initial working directory for sandbox.
        session_name: Bash session name.
        timeout: Command timeout in seconds.
        system_prompt: Optional system prompt suffix.
        harness_tools: Tools to make available to the agent.

    Returns:
        Configured OpenHands Agent.

    Example:
        >>> from swerex.deployment.docker import DockerDeployment
        >>> from openhands.sdk import LLM, Conversation
        >>>
        >>> async def main():
        ...     deployment = DockerDeployment(image="python:3.11")
        ...     await deployment.start()
        ...
        ...     llm = LLM(model="anthropic/claude-sonnet-4-20250514", api_key="...")
        ...     agent = create_sandbox_agent(llm, deployment.runtime)
        ...
        ...     conv = Conversation(agent=agent, workspace="/workspace")
        ...     conv.send_message("List files")
        ...     conv.run()
        ...
        ...     await deployment.stop()
    """
    from openhands.sdk import AgentContext  # type: ignore[ty:unresolved-import]

    tools_list: list[Tool] = []
    sandbox_executor: SandboxTerminalExecutor | None = None

    if harness_tools:
        # Check if any tools need sandbox
        needs_sandbox = any(t.sandbox for t in harness_tools)

        if needs_sandbox:
            if runtime is None:
                raise ValueError("Sandbox runtime required for sandbox tools")
            sandbox_executor = SandboxTerminalExecutor(
                runtime=runtime,
                session_name=session_name,
                timeout=timeout,
                working_dir=working_dir,
            )

        for tool in harness_tools:
            if tool.sandbox:
                # Sandbox tool - execute via sandbox runtime
                tool_name = _register_harness_tool(tool, sandbox_executor=sandbox_executor)
            else:
                # Non-sandbox tool - execute directly
                tool_name = _register_harness_tool(tool)
            tools_list.append(Tool(name=tool_name))

    agent_context = None
    if system_prompt:
        agent_context = AgentContext(system_message_suffix=system_prompt)

    return Agent(
        llm=llm,
        tools=tools_list,
        agent_context=agent_context,
    )

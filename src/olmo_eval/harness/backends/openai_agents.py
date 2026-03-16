"""OpenAI Agents SDK backend."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.common.types.tools import ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.backends import Backend, register_backend
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.harness.tools import Tool
from olmo_eval.inference.base import InferenceProvider

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import ExecutorBinding, SandboxManager

logger = logging.getLogger(__name__)

_current_binding: ContextVar[ExecutorBinding | None] = ContextVar("_current_binding", default=None)


@register_backend("openai_agents")
class OpenAIAgentsBackend(Backend):
    """Backend that delegates execution to OpenAI Agents SDK.

    This backend converts Harness tools to the agents SDK format
    and uses the SDK's Runner for execution.
    """

    name = "openai_agents"
    required_extras = ("agents",)

    def __init__(self) -> None:
        self._cached_agent: Any = None  # Agent type from agents SDK
        self._cached_config: HarnessConfig | None = None
        self._cached_provider_id: int | None = None
        self._cached_has_sandbox: bool = False
        self._sandbox_manager: SandboxManager | None = None

    def clear_cache(self) -> None:
        """Clear cached agent to allow recreation with new config/provider."""
        self._cached_agent = None
        self._cached_config = None
        self._cached_provider_id = None
        self._cached_has_sandbox = False

    async def initialize(self, config: HarnessConfig) -> None:
        """Initialize sandbox manager if needed.

        Called during worker startup to create the sandbox before processing.
        """
        needs_sandbox = config.sandboxes and config.has_sandbox_tools

        if needs_sandbox and self._sandbox_manager is None:
            from olmo_eval.harness.sandbox import SandboxManager

            self._sandbox_manager = SandboxManager(config.sandboxes, owner=config.name)
            await self._sandbox_manager.start()
            logger.info(
                f"Sandbox manager started with {self._sandbox_manager.executor_count} executor(s)"
            )

    async def cleanup(self) -> None:
        """Clean up resources including sandbox manager."""
        if self._sandbox_manager is not None:
            await self._sandbox_manager.stop()
            self._sandbox_manager = None
        self.clear_cache()

    def _convert_tools(
        self,
        tools: Sequence[Tool],
        function_tool: Any,
        sandbox_manager: SandboxManager | None = None,
    ) -> list[Any]:
        """Convert harness tools to agents SDK format.

        Args:
            tools: Sequence of Tool instances to convert.
            function_tool: The function_tool decorator from agents SDK.
            sandbox_manager: Optional sandbox manager for tools that require it.

        Returns:
            List of agents SDK tool objects.
        """
        agent_tools = []
        for tool in tools:
            execute_fn = tool.execute

            # Wrap sandboxed tools to use the manager
            if tool.sandbox and sandbox_manager is not None:
                execute_fn = self._wrap_sandboxed_tool(tool, sandbox_manager)

            # Use function_tool decorator to wrap the execute function
            wrapped = function_tool(strict_mode=False)(execute_fn)
            # Override name and description
            wrapped.name = tool.name
            if hasattr(wrapped, "description"):
                wrapped.description = tool.description
            agent_tools.append(wrapped)
        return agent_tools

    def _wrap_sandboxed_tool(
        self,
        tool: Tool,
        manager: SandboxManager,
    ) -> Any:
        """Create a wrapper function that executes the tool via sandbox manager.

        Args:
            tool: The tool requiring sandbox execution.
            manager: The sandbox manager to use for routing.

        Returns:
            An async function that executes commands via the sandbox.
        """
        required_caps = tool.sandbox

        if tool.session:

            async def sandboxed_execute(command: str) -> str:
                """Execute command in sandbox session."""
                binding = _current_binding.get()
                if binding is None:
                    raise RuntimeError("No binding set for session tool")
                result = await binding.execute_in_session(command)
                output = result.output
                if result.exit_code != 0:
                    output += f"\n[Exit code: {result.exit_code}]"
                return output
        else:

            async def sandboxed_execute(command: str) -> str:
                """Execute command in sandbox."""
                return await manager.execute(command, capabilities=required_caps)

        return sandboxed_execute

    def _create_agent(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        sandbox_manager: SandboxManager | None = None,
    ) -> Any:
        """Create a new agent with the given configuration.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration.
            sandbox_manager: Optional sandbox manager for sandboxed tools.

        Returns:
            An Agent instance from the agents SDK.
        """
        from agents import (
            Agent,
            OpenAIChatCompletionsModel,
            function_tool,
            set_tracing_disabled,
        )

        from olmo_eval.inference.utils import patch_openai_agents_for_vllm

        # Disable trace export to OpenAI's backend (we don't have OPENAI_API_KEY set)
        set_tracing_disabled(True)

        patch_openai_agents_for_vllm()

        # Create model using provider's OpenAI client
        client = provider.get_openai_client()
        logger.debug(
            f"Creating agent with client: {type(client).__name__}, "
            f"base_url={getattr(client, 'base_url', 'unknown')}, "
            f"model={provider.model_name}"
        )

        model = OpenAIChatCompletionsModel(
            openai_client=client,
            model=provider.model_name,
        )

        agent_tools = self._convert_tools(config.resolved_tools, function_tool, sandbox_manager)

        agent = Agent(
            name=self.name,
            instructions=config.system_prompt or "",
            model=model,
            tools=agent_tools,
        )

        return agent

    def _get_or_create_agent(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        sandbox_manager: SandboxManager | None = None,
    ) -> Any:
        """Get cached agent or create a new one if config/provider changed.

        Agents are cached based on config, provider, and whether sandbox is used.
        The sandbox manager is stable across runs, so caching works.
        """
        has_sandbox = sandbox_manager is not None
        if (
            self._cached_agent is not None
            and self._cached_config == config
            and self._cached_provider_id == id(provider)
            and self._cached_has_sandbox == has_sandbox
        ):
            return self._cached_agent

        agent = self._create_agent(provider, config, sandbox_manager)

        self._cached_agent = agent
        self._cached_config = config
        self._cached_provider_id = id(provider)
        self._cached_has_sandbox = has_sandbox

        return agent

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        """Execute using OpenAI Agents SDK.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request.
            sampling_params: Optional sampling parameters.
            trace_metadata: Optional metadata for tracing (e.g., instance_id, task_id).
            **kwargs: Backend-specific options:
                - enable_compaction: Enable context compaction (default: True).

        Returns:
            HarnessResult with trajectory from SDK execution.
        """
        enable_compaction = kwargs.get("enable_compaction", True)
        try:
            from agents import Runner, trace
        except ImportError as e:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            ) from e

        # Create compaction session if enabled
        session = None
        if enable_compaction:
            try:
                from agents import SQLiteSession
                from agents.memory import (
                    OpenAIResponsesCompactionSession,
                )

                session_id = (trace_metadata or {}).get("task_id", "default")
                # Use an in-memory SQLite session as the underlying storage
                underlying = SQLiteSession(session_id, db_path=":memory:")
                session = OpenAIResponsesCompactionSession(
                    session_id=session_id,
                    underlying_session=underlying,
                )
            except ImportError:
                logger.warning("Context compaction not available - agents.memory not found")

        # Check if we need sandbox execution
        needs_sandbox = config.sandboxes and config.has_sandbox_tools

        # Lazily create and cache the sandbox manager
        if needs_sandbox and self._sandbox_manager is None:
            from olmo_eval.harness.sandbox import SandboxManager

            self._sandbox_manager = SandboxManager(config.sandboxes, owner=config.name)
            await self._sandbox_manager.start()
            logger.info(
                f"Sandbox manager started with {self._sandbox_manager.executor_count} executor(s)"
            )

        # Use cached agent (tools read from ContextVar at execution time)
        agent = self._get_or_create_agent(provider, config, self._sandbox_manager)

        # Acquire binding if session tools are used
        has_session_tools = any(t.session for t in config.resolved_tools if t.sandbox)
        binding_token = None

        if has_session_tools and self._sandbox_manager:
            session_caps = frozenset().union(
                *(t.sandbox for t in config.resolved_tools if t.session and t.sandbox)
            )
            binding = await self._sandbox_manager.acquire_binding(session_caps)
            binding_token = _current_binding.set(binding)

        # Get the input message
        input_text = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break

        # Track if max turns was reached
        max_turns_reached = False
        max_turns = config.max_turns or 10

        # Build trace name from config and metadata
        instance_id = (trace_metadata or {}).get("instance_id", "")
        if instance_id:
            trace_name = f"{config.name}:{instance_id}" if config.name else f"Agent:{instance_id}"
        else:
            trace_name = f"Agent: {config.name}" if config.name else "Agent run"

        # Run agent within trace context for observability
        with trace(trace_name, metadata=trace_metadata):
            try:
                run_kwargs: dict[str, Any] = {
                    "starting_agent": agent,
                    "input": input_text,
                    "max_turns": max_turns,
                }
                if session is not None:
                    run_kwargs["session"] = session

                result = await Runner.run(**run_kwargs)
            except Exception as e:
                # Handle MaxTurnsExceeded - return a result with the error instead of raising
                if type(e).__name__ == "MaxTurnsExceeded":
                    return HarnessResult(
                        trajectory=AgentTrajectory(turns=()),
                        final_output=LMOutput(text="[Max turns exceeded]"),
                        max_turns_reached=True,
                        error=f"Max turns ({max_turns}) exceeded",
                    )

                # Log full traceback for debugging connection issues
                import traceback

                logger.error(f"Agent run failed: {e}\n{traceback.format_exc()}")
                raise
            finally:
                # Release binding after run
                binding = _current_binding.get()
                if binding is not None:
                    await binding.release()
                    if binding_token is not None:
                        _current_binding.reset(binding_token)

        # Convert result to HarnessResult
        trajectory = self._convert_trajectory(result)
        final_text = result.final_output if hasattr(result, "final_output") else ""

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text or ""),
            max_turns_reached=max_turns_reached,
            error="Max turns exceeded" if max_turns_reached else None,
        )

    def _convert_trajectory(self, result: Any) -> AgentTrajectory:
        """Convert agents SDK result to AgentTrajectory.

        Args:
            result: Result from Runner.run().

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        # Get items from new_items (primary source in agents SDK)
        items = getattr(result, "new_items", None) or []
        if not items:
            # Fallback to to_input_list() for full conversation history
            if hasattr(result, "to_input_list"):
                try:
                    input_list = result.to_input_list()
                    if input_list:
                        return self._convert_input_list_to_trajectory(input_list)
                except Exception:
                    pass
            return AgentTrajectory(turns=tuple(turns))

        for item in items:
            item_class = type(item).__name__

            if item_class == "MessageOutputItem":
                raw = getattr(item, "raw_item", None)
                content = ""
                if raw is not None:
                    raw_content = getattr(raw, "content", None)
                    if raw_content:
                        for part in raw_content:
                            if hasattr(part, "text"):
                                content += part.text
                if content:
                    turns.append(AgentTurn.assistant(content=content))

            elif item_class == "ToolCallItem":
                raw = getattr(item, "raw_item", None)
                if raw is not None:
                    call_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or ""
                    name = getattr(raw, "name", "") or ""
                    arguments = getattr(raw, "arguments", "{}") or "{}"
                    raw_dict = raw.model_dump() if hasattr(raw, "model_dump") else {}
                    tool_call = ToolCall.create(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        metadata=raw_dict,
                    )
                    turns.append(AgentTurn.assistant(content="", tool_calls=[tool_call]))

            elif item_class == "ToolCallOutputItem":
                output = getattr(item, "output", None)
                raw = getattr(item, "raw_item", None)
                # Extract tool_call_id from raw_item
                tool_call_id = ""
                if raw is not None:
                    tool_call_id = getattr(raw, "call_id", "") or getattr(raw, "id", "") or ""
                content = str(output) if output is not None else ""
                tool_result = ToolResult(
                    tool_call_id=tool_call_id,
                    content=content,
                )
                turns.append(AgentTurn.tool([tool_result]))

        return AgentTrajectory(turns=tuple(turns))

    def _convert_input_list_to_trajectory(self, input_list: list[Any]) -> AgentTrajectory:
        """Convert input list (from to_input_list()) to AgentTrajectory.

        This is a fallback for when new_items is empty but we have the full
        conversation history available via to_input_list().

        Args:
            input_list: List of input items from result.to_input_list().

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        for item in input_list:
            # Items can be dicts or objects
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                tool_calls = item.get("tool_calls", [])

                if role == "assistant":
                    if tool_calls:
                        converted_calls = []
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                call_id = tc.get("id", "")
                                func = tc.get("function", {})
                                is_dict = isinstance(func, dict)
                                name = func.get("name", "") if is_dict else ""
                                args = func.get("arguments", "{}") if is_dict else "{}"
                            else:
                                call_id = getattr(tc, "id", "")
                                func = getattr(tc, "function", None)
                                name = getattr(func, "name", "") if func else ""
                                args = getattr(func, "arguments", "{}") if func else "{}"
                            converted_calls.append(
                                ToolCall.create(call_id=call_id, name=name, arguments=args)
                            )
                        turns.append(
                            AgentTurn.assistant(content=content, tool_calls=converted_calls)
                        )
                    elif content:
                        turns.append(AgentTurn.assistant(content=content))

                elif role == "tool":
                    tool_call_id = item.get("tool_call_id", "")
                    tool_result = ToolResult(tool_call_id=tool_call_id, content=content)
                    turns.append(AgentTurn.tool([tool_result]))

                elif role == "user":
                    turns.append(AgentTurn.user(content=content))
            else:
                # Handle object-based items
                item_type = type(item).__name__
                role = getattr(item, "role", None) or getattr(item, "type", "")

                is_assistant = item_type in ("ResponseOutputMessage", "MessageOutputItem")
                if is_assistant or role == "assistant":
                    content = ""
                    raw_content = getattr(item, "content", None)
                    if isinstance(raw_content, str):
                        content = raw_content
                    elif raw_content:
                        for part in raw_content:
                            if hasattr(part, "text"):
                                content += part.text
                    if content:
                        turns.append(AgentTurn.assistant(content=content))

        return AgentTrajectory(turns=tuple(turns))

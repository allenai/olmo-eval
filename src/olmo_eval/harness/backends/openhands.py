"""OpenHands SDK backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.common.types.tools import ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.backends import Backend, register_backend
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.inference.base import InferenceProvider

if TYPE_CHECKING:
    from olmo_eval.harness.sandbox import SandboxManager

logger = logging.getLogger(__name__)


@register_backend("openhands")
class OpenHandsBackend(Backend):
    """Backend that delegates execution to OpenHands SDK.

    This backend uses the openhands-ai SDK to run agents with built-in
    tools like bash execution and file editing.
    """

    name = "openhands"
    required_extras = ("openhands",)

    def __init__(self) -> None:
        self._sandbox_manager: SandboxManager | None = None

    async def initialize(self, config: HarnessConfig) -> None:
        """Initialize sandbox manager if needed."""
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

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> HarnessResult:
        """Execute using OpenHands SDK.

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
        try:
            from openhands.sdk import (  # type: ignore[import-not-found]
                LLM,
                Agent,
                Conversation,
                Tool,
            )
            from openhands.tools.file_editor import FileEditorTool  # type: ignore[import-not-found]
            from openhands.tools.terminal import TerminalTool  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenHands SDK not installed. Install with: pip install openhands-ai"
            ) from e

        max_turns = config.max_turns or 10

        # Get the input message
        input_text = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break

        # Configure LLM using the provider
        client = provider.get_openai_client()
        llm = LLM(
            model=provider.model_name,
            api_key=client.api_key,
            base_url=str(client.base_url) if client.base_url else None,
        )

        # Configure tools - use OpenHands built-in tools
        tools = [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
        ]

        # Create agent
        agent = Agent(
            llm=llm,
            tools=tools,
            instructions=config.system_prompt or "",
        )

        # Configure workspace
        workspace = None
        if self._sandbox_manager is not None:
            # Use sandbox working directory if available
            executors = self._sandbox_manager._executors
            executor = executors[0] if executors else None
            if executor:
                workspace = getattr(executor, "working_dir", None)

        # Create conversation and run
        conversation = Conversation(
            agent=agent,
            workspace=workspace or "/tmp",
        )

        # Send message and run
        max_turns_reached = False
        try:
            conversation.send_message(input_text)
            conversation.run(max_iterations=max_turns)
        except Exception as e:
            error_name = type(e).__name__
            if "MaxIterations" in error_name or "MaxTurns" in error_name:
                max_turns_reached = True
            else:
                raise

        # Convert conversation state to trajectory
        trajectory = self._convert_conversation_to_trajectory(conversation)

        # Get final output from conversation state
        final_text = ""
        if hasattr(conversation, "state") and conversation.state:
            # Extract last assistant message from state
            for msg in reversed(conversation.state):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    final_text = msg.get("content", "")
                    break

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text),
            max_turns_reached=max_turns_reached,
            error="Max turns exceeded" if max_turns_reached else None,
        )

    def _convert_conversation_to_trajectory(self, conversation: Any) -> AgentTrajectory:
        """Convert OpenHands conversation state to AgentTrajectory.

        Args:
            conversation: The conversation object with state.

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        # Get state from conversation (list of message dicts)
        state = getattr(conversation, "state", []) or []

        for msg in state:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "user":
                turns.append(AgentTurn.user(content=content))
            elif role == "assistant":
                if tool_calls:
                    # Convert tool calls
                    converted_calls = []
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            call = ToolCall.create(
                                call_id=tc.get("id", ""),
                                name=tc.get("function", {}).get("name", ""),
                                arguments=tc.get("function", {}).get("arguments", "{}"),
                            )
                            converted_calls.append(call)
                    turns.append(AgentTurn.assistant(content=content, tool_calls=converted_calls))
                else:
                    turns.append(AgentTurn.assistant(content=content))
            elif role == "tool":
                tool_result = ToolResult(
                    tool_call_id=msg.get("tool_call_id", ""),
                    content=content,
                )
                turns.append(AgentTurn.tool([tool_result]))

        return AgentTrajectory(turns=tuple(turns))

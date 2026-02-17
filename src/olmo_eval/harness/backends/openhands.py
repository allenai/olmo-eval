"""OpenHands SDK backend."""

from __future__ import annotations

import json
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
            from openhands.sdk import (
                LLM,
                Agent,
                Conversation,
                Tool,
            )
            from openhands.sdk.conversation.state import (
                ConversationExecutionStatus,
            )
            from openhands.sdk.event import (
                ActionEvent,
                MessageEvent,
                ObservationEvent,
            )
            from openhands.sdk.event.conversation_error import (
                ConversationErrorEvent,
            )
            from openhands.tools.file_editor import FileEditorTool
            from openhands.tools.terminal import TerminalTool
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
        # OpenHands uses LiteLLM internally, which requires provider prefix
        # For OpenAI-compatible endpoints (like vLLM), prefix with "openai/"
        client = provider.get_openai_client()
        llm = LLM(
            model=f"openai/{provider.model_name}",
            api_key=client.api_key or "dummy",  # vLLM doesn't require auth
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

        # Create conversation with visualizer=None to disable console output
        conversation = Conversation(
            agent=agent,
            workspace=workspace or "/tmp",
            max_iteration_per_run=max_turns,
            visualizer=None,  # Disable console output
        )

        # Send message and run (run() takes no arguments)
        conversation.send_message(input_text)
        conversation.run()

        # Check if max iterations was reached
        max_turns_reached = False
        error_msg = None
        if conversation.state.execution_status == ConversationExecutionStatus.ERROR:
            # Check events for MaxIterationsReached error
            for event in reversed(conversation.state.events):
                if isinstance(event, ConversationErrorEvent):
                    if event.code == "MaxIterationsReached":
                        max_turns_reached = True
                        error_msg = "Max turns exceeded"
                    else:
                        error_msg = f"{event.code}: {event.detail}"
                    break

        # Convert conversation events to trajectory
        trajectory = self._convert_events_to_trajectory(
            conversation.state.events,
            MessageEvent,
            ActionEvent,
            ObservationEvent,
        )

        # Get final output from last agent message
        final_text = ""
        for event in reversed(conversation.state.events):
            if isinstance(event, MessageEvent) and event.source == "agent":
                # Extract text content from message
                if event.llm_message and event.llm_message.content:
                    for content in event.llm_message.content:
                        if hasattr(content, "text"):
                            final_text = content.text
                            break
                break

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text),
            max_turns_reached=max_turns_reached,
            error=error_msg,
        )

    def _convert_events_to_trajectory(
        self,
        events: list[Any],
        message_event_cls: type,
        action_event_cls: type,
        observation_event_cls: type,
    ) -> AgentTrajectory:
        """Convert OpenHands events to AgentTrajectory.

        Args:
            events: List of OpenHands events from conversation.state.events.
            message_event_cls: The MessageEvent class.
            action_event_cls: The ActionEvent class.
            observation_event_cls: The ObservationEvent class.

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        for event in events:
            if isinstance(event, message_event_cls):
                # MessageEvent contains user or agent messages
                content = ""
                if event.llm_message and event.llm_message.content:
                    for c in event.llm_message.content:
                        if hasattr(c, "text"):
                            content = c.text
                            break

                if event.source == "user":
                    turns.append(AgentTurn.user(content=content))
                elif event.source == "agent":
                    turns.append(AgentTurn.assistant(content=content))

            elif isinstance(event, action_event_cls):
                # ActionEvent contains agent tool calls
                content = ""
                if event.thought:
                    # Extract thought text
                    for t in event.thought:
                        if hasattr(t, "text"):
                            content = t.text
                            break

                # Convert tool call
                tool_calls = []
                if event.tool_call:
                    # Get arguments from the action object
                    args = "{}"
                    if event.action:
                        # Serialize the action to get arguments
                        try:
                            args = json.dumps(event.action.model_dump())
                        except Exception:
                            args = "{}"

                    call = ToolCall.create(
                        call_id=event.tool_call_id or event.id,
                        name=event.tool_name,
                        arguments=args,
                    )
                    tool_calls.append(call)

                turns.append(AgentTurn.assistant(content=content, tool_calls=tool_calls))

            elif isinstance(event, observation_event_cls):
                # ObservationEvent contains tool results
                content = ""
                if event.observation:
                    # Serialize observation to get content
                    try:
                        content = json.dumps(event.observation.model_dump())
                    except Exception:
                        content = str(event.observation)

                tool_result = ToolResult(
                    tool_call_id=event.tool_call_id or event.action_id,
                    content=content,
                )
                turns.append(AgentTurn.tool([tool_result]))

        return AgentTrajectory(turns=tuple(turns))

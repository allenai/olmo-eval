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
            from openhands import (  # type: ignore[import-not-found]
                LLM,
                Agent,
                Conversation,
            )
            from openhands.tools import (  # type: ignore[import-not-found]
                FileEditorTool,
                TerminalTool,
            )
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
        tools = [TerminalTool(), FileEditorTool()]

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

        # Track events for trajectory conversion
        events: list[Any] = []

        def on_event(event: Any) -> None:
            events.append(event)

        # Subscribe to events if available
        has_stream = hasattr(conversation, "event_stream")
        has_subscribe = has_stream and hasattr(conversation.event_stream, "subscribe")
        if has_subscribe:
            from openhands.events import EventStreamSubscriber  # type: ignore[import-not-found]

            conversation.event_stream.subscribe(EventStreamSubscriber.MAIN, on_event)

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

        # Convert events to trajectory
        trajectory = self._convert_events_to_trajectory(events)

        # Get final output
        final_text = ""
        if hasattr(conversation, "get_last_response"):
            final_text = conversation.get_last_response() or ""
        elif events:
            # Try to extract final response from events
            for event in reversed(events):
                has_content = hasattr(event, "content") and hasattr(event, "role")
                if has_content and event.role == "assistant":
                    final_text = event.content
                    break

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text),
            max_turns_reached=max_turns_reached,
            error="Max turns exceeded" if max_turns_reached else None,
        )

    def _convert_events_to_trajectory(self, events: list[Any]) -> AgentTrajectory:
        """Convert OpenHands events to AgentTrajectory.

        Args:
            events: List of events from the conversation.

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        for event in events:
            event_type = type(event).__name__

            # Handle action events (agent decisions)
            if "Action" in event_type:
                tool_call = self._action_to_tool_call(event)
                if tool_call:
                    turns.append(AgentTurn.assistant(content="", tool_calls=[tool_call]))
                elif hasattr(event, "content") and event.content:
                    turns.append(AgentTurn.assistant(content=event.content))

            # Handle observation events (tool results)
            elif "Observation" in event_type:
                tool_result = self._observation_to_tool_result(event)
                if tool_result:
                    turns.append(AgentTurn.tool([tool_result]))

            # Handle message events
            elif hasattr(event, "role") and hasattr(event, "content"):
                if event.role == "assistant":
                    turns.append(AgentTurn.assistant(content=event.content or ""))
                elif event.role == "user":
                    turns.append(AgentTurn.user(content=event.content or ""))

        return AgentTrajectory(turns=tuple(turns))

    def _action_to_tool_call(self, action: Any) -> ToolCall | None:
        """Convert an OpenHands action event to a ToolCall.

        Args:
            action: The action event.

        Returns:
            ToolCall or None if not a tool action.
        """
        import json

        action_type = type(action).__name__

        # Map common action types to tool calls
        if action_type == "CmdRunAction":
            return ToolCall.create(
                call_id=getattr(action, "id", "") or "",
                name="execute_bash",
                arguments=json.dumps({"command": getattr(action, "command", "")}),
            )
        elif action_type == "FileEditAction":
            return ToolCall.create(
                call_id=getattr(action, "id", "") or "",
                name="file_edit",
                arguments=json.dumps(
                    {
                        "path": getattr(action, "path", ""),
                        "content": getattr(action, "content", ""),
                    }
                ),
            )
        elif action_type == "FileReadAction":
            return ToolCall.create(
                call_id=getattr(action, "id", "") or "",
                name="file_read",
                arguments=json.dumps({"path": getattr(action, "path", "")}),
            )
        elif hasattr(action, "action") and hasattr(action, "args"):
            # Generic action with action type and args
            args = action.args
            arguments = json.dumps(args) if isinstance(args, dict) else str(args)
            return ToolCall.create(
                call_id=getattr(action, "id", "") or "",
                name=action.action,
                arguments=arguments,
            )

        return None

    def _observation_to_tool_result(self, observation: Any) -> ToolResult | None:
        """Convert an OpenHands observation event to a ToolResult.

        Args:
            observation: The observation event.

        Returns:
            ToolResult or None if not a tool result.
        """
        # Extract content from observation
        content = ""
        if hasattr(observation, "content"):
            content = str(observation.content)
        elif hasattr(observation, "output"):
            content = str(observation.output)

        # Extract tool call ID (linked to the action that produced this observation)
        tool_call_id = ""
        if hasattr(observation, "cause_id"):
            tool_call_id = str(observation.cause_id)
        elif hasattr(observation, "action_id"):
            tool_call_id = str(observation.action_id)
        elif hasattr(observation, "id"):
            tool_call_id = str(observation.id)

        if content or tool_call_id:
            return ToolResult(
                tool_call_id=tool_call_id,
                content=content,
            )

        return None

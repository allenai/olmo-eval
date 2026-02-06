"""OpenAI Agents SDK backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.core.types.tools import ToolCall, ToolResult
from olmo_eval.core.types.trajectory import AgentTrajectory, AgentTurn

from ..result import HarnessResult
from . import Backend, register_backend

if TYPE_CHECKING:
    from ..harness import Harness


@register_backend("openai_agents")
class OpenAIAgentsBackend(Backend):
    """Backend that delegates execution to OpenAI Agents SDK.

    This backend converts Harness tools to the agents SDK format
    and uses the SDK's Runner for execution.
    """

    async def run(
        self,
        harness: Harness,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute using OpenAI Agents SDK.

        Args:
            harness: The Harness instance.
            request: The initial request.
            sampling_params: Optional sampling parameters.

        Returns:
            HarnessResult with trajectory from SDK execution.
        """
        try:
            from agents import (  # type: ignore[import-not-found]
                Agent,
                OpenAIChatCompletionsModel,
                Runner,
                function_tool,
            )
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            ) from e

        import os

        # Create OpenAI client pointing to harness provider
        base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=60.0,
        )

        model = OpenAIChatCompletionsModel(
            openai_client=client,
            model=harness.model_name,
        )

        # Convert harness tools to agents SDK format
        agent_tools = []
        for tool in harness.config.tools:
            # Use function_tool decorator to wrap the execute function
            wrapped = function_tool(strict_mode=False)(tool.execute)
            # Override name and description
            wrapped.name = tool.name
            if hasattr(wrapped, "description"):
                wrapped.description = tool.description
            agent_tools.append(wrapped)

        # Create agent
        agent = Agent(
            name=harness.config.name,
            instructions=harness.config.system_prompt or "",
            model=model,
            tools=agent_tools,
        )

        # Get the input message
        input_text = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break

        # Run agent
        max_turns = harness.config.max_turns or 10
        result = await Runner.run(
            starting_agent=agent,
            input=input_text,
            max_turns=max_turns,
        )

        # Convert result to HarnessResult
        trajectory = self._convert_trajectory(result)
        final_text = result.final_output if hasattr(result, "final_output") else ""

        return HarnessResult(
            trajectory=trajectory,
            final_output=LMOutput(text=final_text or ""),
        )

    def _convert_trajectory(self, result: Any) -> AgentTrajectory:
        """Convert agents SDK result to AgentTrajectory.

        Args:
            result: Result from Runner.run().

        Returns:
            AgentTrajectory with converted turns.
        """
        turns: list[AgentTurn] = []

        # The agents SDK provides run history in result.new_items or similar
        # This is a simplified conversion; actual implementation depends on SDK version
        if hasattr(result, "new_items"):
            for item in result.new_items:
                item_type = getattr(item, "type", None) or type(item).__name__

                if "message" in item_type.lower() or "output" in item_type.lower():
                    # Assistant message
                    content = ""
                    if hasattr(item, "output"):
                        content = str(item.output)
                    elif hasattr(item, "content"):
                        content = str(item.content)

                    tool_calls = []
                    if hasattr(item, "tool_calls") and item.tool_calls:
                        for tc in item.tool_calls:
                            tc_name = getattr(tc, "name", "") or getattr(
                                getattr(tc, "function", None), "name", ""
                            )
                            tool_calls.append(
                                ToolCall.create(
                                    call_id=getattr(tc, "id", ""),
                                    name=tc_name,
                                    arguments=getattr(tc, "arguments", "{}"),
                                )
                            )

                    turns.append(
                        AgentTurn.assistant(content=content, tool_calls=tool_calls or None)
                    )

                elif "tool" in item_type.lower():
                    # Tool result
                    results = []
                    if hasattr(item, "output"):
                        tool_call_id = getattr(item, "call_id", getattr(item, "tool_call_id", ""))
                        results.append(
                            ToolResult(
                                tool_call_id=tool_call_id,
                                content=str(item.output),
                            )
                        )
                    turns.append(AgentTurn.tool(results))

        return AgentTrajectory(turns=tuple(turns))

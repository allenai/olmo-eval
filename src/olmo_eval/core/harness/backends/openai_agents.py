"""OpenAI Agents SDK backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from olmo_eval.core.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.core.types.tools import ToolCall, ToolResult
from olmo_eval.core.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.inference.base import InferenceProvider

from ..config import HarnessConfig
from ..result import HarnessResult
from . import Backend, register_backend


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

    def _get_openai_client(self, provider: InferenceProvider) -> Any:
        """Get OpenAI client from provider or fallback to environment."""
        client = provider.get_openai_client()
        if client is None:
            # Fallback to environment variables for backward compatibility
            import os

            from openai import AsyncOpenAI  # type: ignore[import-not-found]

            client = AsyncOpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                timeout=60.0,
            )
        return client

    def _convert_tools(self, tools: Sequence[Any], function_tool: Any) -> list[Any]:
        """Convert harness tools to agents SDK format."""
        agent_tools = []
        for tool in tools:
            # Use function_tool decorator to wrap the execute function
            wrapped = function_tool(strict_mode=False)(tool.execute)
            # Override name and description
            wrapped.name = tool.name
            if hasattr(wrapped, "description"):
                wrapped.description = tool.description
            agent_tools.append(wrapped)
        return agent_tools

    def _get_or_create_agent(self, provider: InferenceProvider, config: HarnessConfig) -> Any:
        """Get cached agent or create a new one if config/provider changed."""
        from agents import (  # type: ignore[import-not-found]
            Agent,
            OpenAIChatCompletionsModel,
            function_tool,
        )

        # Return cached if same config (identity check - config is frozen)
        # and same provider (id check)
        if (
            self._cached_agent is not None
            and self._cached_config is config
            and self._cached_provider_id == id(provider)
        ):
            return self._cached_agent

        # Create model
        client = self._get_openai_client(provider)
        model = OpenAIChatCompletionsModel(
            openai_client=client,
            model=provider.model_name,
        )

        # Convert tools
        agent_tools = self._convert_tools(config.resolved_tools, function_tool)

        # Create agent
        agent = Agent(
            name=self.name,
            instructions=config.system_prompt or "",
            model=model,
            tools=agent_tools,
        )

        # Cache for reuse
        self._cached_agent = agent
        self._cached_config = config
        self._cached_provider_id = id(provider)

        return agent

    async def run(
        self,
        provider: InferenceProvider,
        config: HarnessConfig,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute using OpenAI Agents SDK.

        Args:
            provider: The inference provider for model calls.
            config: Harness configuration (tools, system prompt, etc.).
            request: The initial request.
            sampling_params: Optional sampling parameters.

        Returns:
            HarnessResult with trajectory from SDK execution.
        """
        try:
            from agents import Runner  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            ) from e

        # Get or create cached agent
        agent = self._get_or_create_agent(provider, config)

        # Get the input message
        input_text = ""
        if request.messages:
            for msg in reversed(request.messages):
                if msg.get("role") == "user":
                    input_text = msg.get("content", "")
                    break

        # Run agent
        max_turns = config.max_turns or 10
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

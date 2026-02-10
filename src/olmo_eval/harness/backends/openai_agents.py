"""OpenAI Agents SDK backend."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.common.types.tools import ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.harness.backends import Backend, register_backend
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.result import HarnessResult
from olmo_eval.inference.base import InferenceProvider

logger = logging.getLogger(__name__)


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

    def clear_cache(self) -> None:
        """Clear cached agent to allow recreation with new config/provider."""
        self._cached_agent = None
        self._cached_config = None
        self._cached_provider_id = None

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

        from olmo_eval.inference.utils import patch_openai_agents_for_vllm

        patch_openai_agents_for_vllm()

        if (
            self._cached_agent is not None
            and self._cached_config == config
            and self._cached_provider_id == id(provider)
        ):
            return self._cached_agent

        # Create model using provider's OpenAI client
        client = provider.get_openai_client()

        # Log client configuration for debugging
        base_url = getattr(client, "base_url", "unknown")
        timeout = getattr(client, "timeout", "unknown")
        logger.info(f"OpenAI client configured: base_url={base_url}, timeout={timeout}")

        model = OpenAIChatCompletionsModel(
            openai_client=client,
            model=provider.model_name,
        )

        agent_tools = self._convert_tools(config.resolved_tools, function_tool)

        agent = Agent(
            name=self.name,
            instructions=config.system_prompt or "",
            model=model,
            tools=agent_tools,
        )

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

        if not hasattr(result, "new_items"):
            return AgentTrajectory(turns=tuple(turns))

        for item in result.new_items:
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
                turns.append(AgentTurn.assistant(content=content))

            elif item_class == "ToolCallItem":
                raw = getattr(item, "raw_item", None)
                if raw is not None:
                    raw_dict = raw.model_dump() if hasattr(raw, "model_dump") else {}
                    tool_call = ToolCall.create(
                        call_id=raw.call_id,
                        name=raw.name,
                        arguments=raw.arguments or "{}",
                        metadata=raw_dict,
                    )
                    turns.append(AgentTurn.assistant(content="", tool_calls=[tool_call]))

            elif item_class == "ToolCallOutputItem":
                output = getattr(item, "output", None)
                raw = getattr(item, "raw_item", None)
                raw_dict = dict(raw) if isinstance(raw, dict) else {}
                tool_call_id = raw_dict.get("call_id", "")
                content = str(output) if output is not None else ""
                tool_result = ToolResult(
                    tool_call_id=tool_call_id,
                    content=content,
                    metadata=raw_dict,
                )
                turns.append(AgentTurn.tool([tool_result]))

        return AgentTrajectory(turns=tuple(turns))

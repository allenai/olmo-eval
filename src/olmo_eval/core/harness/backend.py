"""AgentBackend: Pluggable execution backends for multi-turn agent loops.

This module defines the abstract backend interface and provides implementations:
- InternalBackend: Built-in loop using Harness tools directly
- OpenAIAgentsBackend: Delegate to OpenAI Agents SDK

Backends are registered using the @register_backend decorator.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from olmo_eval.core.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.core.types.tools import ToolCall, ToolResult
from olmo_eval.core.types.trajectory import AgentTrajectory, AgentTurn

from .result import HarnessResult

if TYPE_CHECKING:
    from .harness import Harness

logger = logging.getLogger(__name__)


class AgentBackend(ABC):
    """Abstract base class for agent execution backends.

    Backends handle the multi-turn agent loop, including:
    - Sending requests to the model
    - Parsing tool calls from responses
    - Executing tools
    - Building the conversation history
    - Tracking the trajectory

    Different backends can use different execution strategies:
    - InternalBackend: Built-in loop using Harness tools
    - OpenAIAgentsBackend: Delegate to OpenAI Agents SDK
    """

    @abstractmethod
    async def run(
        self,
        harness: Harness,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute the agent loop and return the result.

        Args:
            harness: The Harness instance (provides provider and config).
            request: The initial request to start the conversation.
            sampling_params: Optional sampling parameters override.

        Returns:
            HarnessResult with trajectory and final output.
        """
        ...


# -----------------------------------------------------------------------------
# Backend Registry
# -----------------------------------------------------------------------------

BACKEND_REGISTRY: dict[str, type[AgentBackend]] = {}

T = TypeVar("T", bound=AgentBackend)


def register_backend(name: str):
    """Decorator to register an AgentBackend class.

    Usage:
        @register_backend("internal")
        class InternalBackend(AgentBackend):
            ...

    Args:
        name: Name to register the backend under.

    Returns:
        Decorator function that registers the class.
    """

    def decorator(cls: type[T]) -> type[T]:
        if name in BACKEND_REGISTRY:
            logger.warning(f"Overwriting existing backend: {name}")
        BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str) -> AgentBackend:
    """Get a backend instance by name.

    Args:
        name: Backend name (e.g., "internal", "openai_agents").

    Returns:
        Backend instance.

    Raises:
        ValueError: If backend name is unknown.
    """
    if name not in BACKEND_REGISTRY:
        available = ", ".join(sorted(BACKEND_REGISTRY.keys()))
        raise ValueError(f"Unknown backend: '{name}'. Available: {available}")
    return BACKEND_REGISTRY[name]()


def list_backends() -> list[str]:
    """List all registered backend names.

    Returns:
        Sorted list of backend names.
    """
    return sorted(BACKEND_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Backend Implementations
# -----------------------------------------------------------------------------


@register_backend("internal")
class InternalBackend(AgentBackend):
    """Built-in agent loop using Harness tools directly.

    This backend implements a simple turn-based loop:
    1. Send request to model
    2. If response has tool calls, execute them
    3. Append results to conversation and repeat
    4. Stop when no tool calls or max_turns reached
    """

    async def run(
        self,
        harness: Harness,
        request: LMRequest,
        sampling_params: SamplingParams | None = None,
    ) -> HarnessResult:
        """Execute the internal agent loop.

        Args:
            harness: The Harness instance.
            request: The initial request.
            sampling_params: Optional sampling parameters.

        Returns:
            HarnessResult with complete trajectory.
        """
        turns: list[AgentTurn] = []
        messages: list[dict[str, Any]] = list(request.messages)

        # Build tool lookup
        tools = {t.name: t for t in harness.config.tools}

        for _ in range(harness.config.max_turns):
            # Create request with current messages
            turn_request = harness._apply_config(
                LMRequest(
                    request_type=RequestType.CHAT,
                    messages=tuple(messages),
                )
            )

            # Generate response
            outputs = harness.provider.generate([turn_request], sampling_params)
            output = outputs[0][0]

            # Parse tool calls from output
            tool_calls = output.tool_calls or []

            # Record assistant turn
            turns.append(
                AgentTurn.assistant(
                    content=output.text,
                    tool_calls=tool_calls,
                )
            )

            # If no tool calls, we're done
            if not tool_calls:
                return HarnessResult(
                    trajectory=AgentTrajectory(turns=tuple(turns)),
                    final_output=output,
                )

            # Execute tools
            results = await self._execute_tools(tools, tool_calls)

            # Record tool turn
            turns.append(AgentTurn.tool(results))

            # Update messages for next turn
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": output.text or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [tc.to_openai() for tc in tool_calls]
            messages.append(assistant_msg)

            for result in results:
                messages.append(result.to_openai())

        # Max turns reached
        final_output = outputs[0][0] if outputs else LMOutput(text="")
        return HarnessResult(
            trajectory=AgentTrajectory(turns=tuple(turns)),
            final_output=final_output,
            max_turns_reached=True,
        )

    async def _execute_tools(
        self,
        tools: dict[str, Any],
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """Execute a batch of tool calls.

        Args:
            tools: Mapping of tool names to Tool instances.
            tool_calls: List of tool calls to execute.

        Returns:
            List of ToolResult instances.
        """
        results: list[ToolResult] = []

        for tc in tool_calls:
            tool_name = tc.function.name
            tool = tools.get(tool_name)

            if tool is None:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=f"Unknown tool: {tool_name}",
                        is_error=True,
                    )
                )
                continue

            try:
                args = json.loads(tc.function.arguments)
                result = await tool(**args)
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=str(result),
                    )
                )
            except json.JSONDecodeError as e:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=f"Invalid JSON arguments: {e}",
                        is_error=True,
                    )
                )
            except Exception as e:
                logger.exception(f"Tool {tool_name} failed")
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=f"Tool error: {e}",
                        is_error=True,
                    )
                )

        return results


@register_backend("openai_agents")
class OpenAIAgentsBackend(AgentBackend):
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
            from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAI Agents SDK not installed. Install with: pip install openai-agents"
            ) from e

        import os

        # Create OpenAI client pointing to harness provider
        base_url = harness.config.model_url or os.getenv(
            "OPENAI_BASE_URL", "http://localhost:8000/v1"
        )
        api_key = harness.config.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")

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
        result = await Runner.run(
            starting_agent=agent,
            input=input_text,
            max_turns=harness.config.max_turns,
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

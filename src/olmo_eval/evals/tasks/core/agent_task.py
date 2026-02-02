"""Base class for agent evaluation tasks.

This module provides the AgentTask base class that enables multi-turn agent
evaluations with tool use, similar to the reference AgentEval implementation
but using olmo-eval's type system.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from olmo_eval.core.agents import AgentExecutionResult
from olmo_eval.core.formatters import ChatFormatter, Formatter
from olmo_eval.core.types import (
    AgentTrajectory,
    AgentTurn,
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    ToolCall,
    ToolResult,
    ToolSchema,
)

from .base import Task, TaskConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agents import Agent  # type: ignore[import-not-found]


@dataclass
class AgentTaskConfig(TaskConfig):
    """Configuration for an agent task.

    Extends TaskConfig with agent-specific settings. Model and model_url
    are always provided via CLI, not in the config.

    Attributes:
        system_prompt: System prompt for the agent. Empty string means use task default.
        max_turns: Maximum number of agent turns before stopping.
        max_concurrency: Maximum number of concurrent agent executions.
        required_secrets: Environment variable names required for this task.
            Used by Beaker launcher to set up --env-secret flags.
        tools: Tool schemas available to the agent. Used for display and
            instance creation. The actual tool implementations are created
            in _get_agent().

    Note: temperature and max_tokens come from sampling_params.
    """

    system_prompt: str = ""
    max_turns: int = 10
    max_concurrency: int = 1
    required_secrets: tuple[str, ...] = ()
    tools: tuple[ToolSchema, ...] = ()
    # Override parent's None default with ChatFormatter for consistent serialization
    formatter: Formatter | None = field(default_factory=ChatFormatter)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config including agent-specific fields.

        Extends parent to_dict() with agent_settings. Temperature is already
        captured via sampling_params in the parent's to_dict().

        Note: required_secrets and tools are runtime/infrastructure details,
        not task configuration, so they are not included.
        """
        base = super().to_dict()
        base["agent_settings"] = {
            "system_prompt": self.system_prompt,
            "max_turns": self.max_turns,
            "max_concurrency": self.max_concurrency,
        }
        return base


class AgentTask(Task):
    """Base class for agent evaluation tasks.

    AgentTask extends the standard Task class to support multi-turn agent
    evaluations. Instead of using InferenceProvider.generate() for single-turn
    inference, AgentTask uses an async agent loop that allows the agent to
    make multiple tool calls before producing a final answer.

    Subclasses must implement:
        - instances: Yield evaluation instances from the dataset
        - _get_agent(): Async context manager returning Agent with tools
        - _compute_metrics(): Compute task-specific metrics from results

    Subclasses may optionally override:
        - _build_responses(): Customize how AgentExecutionResults become Responses
        - extract_answer(): Customize answer extraction from LMOutput
        - score_responses(): Customize scoring logic

    Example:
        class MyAgentTask(AgentTask):
            @asynccontextmanager
            async def _get_agent(self, model, model_url, system_prompt, temperature, **kwargs):
                from agents import Agent
                agent = Agent(name="MyAgent", instructions=system_prompt, ...)
                yield agent

            def _compute_metrics(self, results, **kwargs):
                return {"accuracy": sum(r.success for r in results) / len(results)}
    """

    config: AgentTaskConfig

    def __init__(self, config: AgentTaskConfig) -> None:
        super().__init__(config)

    # -------------------------------------------------------------------------
    # Abstract methods - subclasses must implement
    # -------------------------------------------------------------------------

    @abstractmethod
    @asynccontextmanager
    async def _get_agent(
        self,
        model: str,
        model_url: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncGenerator[Agent, None]:
        """Create agent with tools.

        Subclasses must implement this async context manager to create and
        configure the agent with appropriate tools (e.g., MCP servers).

        Args:
            model: The model identifier.
            model_url: The API endpoint URL for the model.
            system_prompt: Optional system prompt for the agent.
            temperature: Sampling temperature.
            **kwargs: Additional arguments for agent configuration.

        Yields:
            An Agent instance configured with tools.

        Example:
            @asynccontextmanager
            async def _get_agent(self, model, model_url, system_prompt, temperature, **kwargs):
                from openai import AsyncOpenAI
                from agents import Agent, OpenAIChatCompletionsModel
                from agents.mcp import MCPServerStdio

                client = AsyncOpenAI(base_url=model_url, api_key="EMPTY")
                llm = OpenAIChatCompletionsModel(openai_client=client, model=model)

                async with MCPServerStdio(
                    params={"command": "python", "args": ["-m", "search_mcp"]},
                ) as server:
                    agent = Agent(
                        name="SearchAgent",
                        instructions=system_prompt or "You are a helpful assistant.",
                        model=llm,
                        mcp_servers=[server],
                    )
                    yield agent
        """
        yield

    # -------------------------------------------------------------------------
    # Standard Task interface
    # -------------------------------------------------------------------------

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        For agent tasks, this creates a simple chat request with the question,
        along with tools and system_prompt for inspection purposes.
        The actual multi-turn interaction is handled by _run_agent_loop.

        If a formatter is configured, uses it for formatting. Otherwise falls
        back to basic chat-style formatting.
        """
        # Use formatter if set
        if self.config.formatter:
            request = self.config.formatter.format(instance, self.get_fewshot())
            # Add tools and system_prompt if not already set by formatter
            tools = instance.tools or self.config.tools or None
            system_prompt = self.config.system_prompt if self.config.system_prompt else None
            return LMRequest(
                request_type=request.request_type,
                messages=request.messages,
                prompt=request.prompt,
                continuations=request.continuations,
                tools=request.tools or (tools if tools else None),
                system_prompt=request.system_prompt or system_prompt,
            )

        # Fallback: basic chat-style formatting
        tools = instance.tools or self.config.tools or None
        system_prompt = self.config.system_prompt if self.config.system_prompt else None

        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            tools=tools if tools else None,
            system_prompt=system_prompt,
        )

    def extract_answer(self, output: LMOutput) -> Any:
        """Extract the answer from model output.

        For agent tasks, the extracted answer is typically set by the agent
        execution and stored in output.extracted_answer.
        """
        return output.extracted_answer or output.text.strip()

    # -------------------------------------------------------------------------
    # Agent execution
    # -------------------------------------------------------------------------

    async def _run_agent_loop(
        self,
        instances: list[Instance],
        model: str,
        model_url: str,
        system_prompt: str | None = None,
        max_turns: int = 10,
        max_concurrency: int = 1,
        temperature: float = 0.0,
        on_instance_complete: Any | None = None,
        **kwargs: Any,
    ) -> list[AgentExecutionResult]:
        """Run agent on all instances with concurrency control.

        This method manages the async execution of the agent across all
        instances, using a semaphore to control concurrency.

        Args:
            instances: List of evaluation instances.
            model: The model identifier.
            model_url: The API endpoint URL.
            system_prompt: Optional system prompt.
            max_turns: Maximum turns per instance.
            max_concurrency: Maximum concurrent executions.
            temperature: Sampling temperature.
            on_instance_complete: Optional callback called after each instance completes.
            **kwargs: Additional arguments passed to _get_agent.

        Returns:
            List of AgentExecutionResult, one per instance.
        """
        from agents import Runner  # type: ignore[import-not-found]

        results: list[AgentExecutionResult] = []
        semaphore = asyncio.Semaphore(max_concurrency)

        async with self._get_agent(
            model=model,
            model_url=model_url,
            system_prompt=system_prompt,
            temperature=temperature,
            **kwargs,
        ) as agent:

            async def process_instance(instance: Instance) -> AgentExecutionResult:
                async with semaphore:
                    try:
                        result = await Runner.run(
                            starting_agent=agent,
                            input=instance.question,
                            max_turns=max_turns,
                        )
                        trajectory = self._convert_to_trajectory(result)
                        exec_result = AgentExecutionResult(
                            trajectory=trajectory,
                            final_answer=result.final_output,
                            success=True,
                        )
                    except Exception as e:
                        logger.exception(f"Agent execution failed: {e}")
                        exec_result = AgentExecutionResult(
                            trajectory=AgentTrajectory(),
                            error=str(e),
                            success=False,
                        )
                    # Call progress callback if provided
                    if on_instance_complete is not None:
                        on_instance_complete()
                    return exec_result

            # Process all instances concurrently (up to max_concurrency)
            tasks = [process_instance(inst) for inst in instances]
            results = await asyncio.gather(*tasks)

        return list(results)

    def _convert_to_trajectory(self, result: Any) -> AgentTrajectory:
        """Convert OpenAI Agents SDK result to AgentTrajectory.

        Args:
            result: The Runner result from the agents SDK.

        Returns:
            An AgentTrajectory with the conversation turns.
        """
        import uuid

        turns: list[AgentTurn] = []
        # Map fake IDs to real IDs to maintain consistency between calls and results
        id_mapping: dict[str, str] = {}

        def get_real_id(fake_id: str) -> str:
            """Get or create a real ID for a potentially fake ID."""
            if not fake_id or fake_id.startswith("__"):
                if fake_id not in id_mapping:
                    id_mapping[fake_id] = f"call_{uuid.uuid4().hex[:24]}"
                return id_mapping[fake_id]
            return fake_id

        for item in result.new_items:
            item_type = getattr(item, "type", None)

            if item_type == "message_output_item":
                # Assistant message - extract from raw_item
                raw = getattr(item, "raw_item", None)
                content = ""
                if raw and hasattr(raw, "content"):
                    # Content can be a list of content parts or a string
                    if isinstance(raw.content, list):
                        content = "".join(
                            part.text if hasattr(part, "text") else str(part)
                            for part in raw.content
                        )
                    else:
                        content = raw.content or ""
                turns.append(
                    AgentTurn.assistant(
                        content=content, tool_calls=None, timestamp_ms=int(time.time() * 1000)
                    )
                )

            elif item_type == "tool_call_item":
                # Tool call - extract function name and arguments
                raw = getattr(item, "raw_item", None)
                if raw:
                    tool_calls: list[ToolCall] = []
                    raw_id = getattr(raw, "id", "") or getattr(raw, "call_id", "")
                    call_id = get_real_id(raw_id)
                    name = getattr(raw, "name", "")
                    arguments = getattr(raw, "arguments", "{}")

                    tool_calls.append(
                        ToolCall.create(
                            call_id=call_id,
                            name=name,
                            arguments=arguments,
                        )
                    )
                    turns.append(
                        AgentTurn.assistant(
                            content="", tool_calls=tool_calls, timestamp_ms=int(time.time() * 1000)
                        )
                    )

            elif item_type == "tool_call_output_item":
                # Tool result
                raw = getattr(item, "raw_item", None)
                output = getattr(item, "output", "")
                raw_id = ""
                if raw:
                    # Check tool_call_id first (OpenAI format), then id/call_id
                    raw_id = (
                        getattr(raw, "tool_call_id", "")
                        or getattr(raw, "id", "")
                        or getattr(raw, "call_id", "")
                    )
                call_id = get_real_id(raw_id)
                turns.append(
                    AgentTurn.tool(
                        [ToolResult(tool_call_id=call_id, content=str(output))],
                        timestamp_ms=int(time.time() * 1000),
                    )
                )

            # Fallback for legacy format with 'role' attribute
            elif hasattr(item, "role"):
                if item.role == "assistant":
                    legacy_tool_calls: list[ToolCall] = []
                    if hasattr(item, "tool_calls") and item.tool_calls:
                        legacy_tool_calls = [
                            ToolCall.from_openai(tc) if isinstance(tc, dict) else tc
                            for tc in item.tool_calls
                        ]
                    turns.append(
                        AgentTurn.assistant(
                            content=getattr(item, "content", "") or "",
                            tool_calls=legacy_tool_calls if legacy_tool_calls else None,
                            timestamp_ms=int(time.time() * 1000),
                        )
                    )
                elif item.role == "tool":
                    turns.append(
                        AgentTurn.tool(
                            [
                                ToolResult(
                                    tool_call_id=getattr(item, "tool_call_id", "")
                                    or getattr(item, "id", ""),
                                    content=getattr(item, "content", "") or "",
                                )
                            ],
                            timestamp_ms=int(time.time() * 1000),
                        )
                    )

        return AgentTrajectory(
            turns=tuple(turns),
            final_answer=result.final_output,
        )

    def _build_responses(
        self,
        instances: list[Instance],
        results: list[AgentExecutionResult],
    ) -> list[Response]:
        """Build olmo-eval Response objects from agent results.

        Args:
            instances: The evaluation instances.
            results: The agent execution results (parallel to instances).

        Returns:
            List of Response objects with trajectories attached.
        """
        responses = []
        for instance, result in zip(instances, results, strict=True):
            output = LMOutput(
                text=result.final_answer or "",
                extracted_answer=result.final_answer,
                metadata={
                    "success": result.success,
                    "error": result.error,
                    **result.metadata,
                },
            )
            responses.append(
                Response(
                    instance=instance,
                    request=LMRequest(
                        request_type=RequestType.CHAT,
                        messages=({"role": "user", "content": instance.question},),
                    ),
                    outputs=[output],
                    trajectory=result.trajectory,
                )
            )
        return responses

    def _validate_secrets(self) -> None:
        """Validate that required environment variables are set.

        Raises:
            ValueError: If a required secret is not set.
        """
        if not hasattr(self.config, "required_secrets"):
            return

        for secret in self.config.required_secrets:
            if not os.getenv(secret):
                raise ValueError(
                    f"Required environment variable {secret} not set. "
                    f"This task requires: {', '.join(self.config.required_secrets)}"
                )

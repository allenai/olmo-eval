"""Base class for agent evaluation tasks.

This module provides the AgentTask base class that enables multi-turn agent
evaluations with tool use, similar to the reference AgentEval implementation
but using olmo-eval's type system.
"""

from __future__ import annotations

import asyncio
import os
from abc import abstractmethod
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from olmo_eval.core.agents import AgentConfig, AgentExecutionResult
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
)

from .base import Task, TaskConfig

if TYPE_CHECKING:
    from agents import Agent  # type: ignore[import-not-found]


@dataclass
class AgentTaskConfig(TaskConfig):
    """Configuration for an agent task.

    Extends TaskConfig with agent-specific settings.

    Attributes:
        agent_config: Configuration for agent execution.
        required_secrets: Environment variable names required for this task.
            Used by Beaker launcher to set up --env-secret flags.
    """

    agent_config: AgentConfig | None = None
    required_secrets: tuple[str, ...] = ()


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

    @abstractmethod
    def _compute_metrics(
        self,
        results: list[AgentExecutionResult],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics from execution results.

        Subclasses must implement this to define task-specific metrics.

        Args:
            results: List of AgentExecutionResult from running the agent.
            **kwargs: Additional arguments (e.g., instances for pairing).

        Returns:
            Dictionary of metric name to value.

        Example:
            def _compute_metrics(self, results, **kwargs):
                correct = sum(1 for r in results if r.success and r.final_answer)
                return {
                    'accuracy': correct / len(results) if results else 0.0,
                    'success_rate': sum(r.success for r in results) / len(results),
                }
        """
        ...

    # -------------------------------------------------------------------------
    # Standard Task interface
    # -------------------------------------------------------------------------

    def format_request(self, instance: Instance) -> LMRequest:
        """Format an instance into an LM request.

        For agent tasks, this creates a simple chat request with the question.
        The actual multi-turn interaction is handled by _run_agent_loop.
        """
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
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
                        return AgentExecutionResult(
                            trajectory=trajectory,
                            final_answer=result.final_output,
                            success=True,
                        )
                    except Exception as e:
                        return AgentExecutionResult(
                            trajectory=AgentTrajectory(),
                            error=str(e),
                            success=False,
                        )

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
        turns: list[AgentTurn] = []

        for item in result.new_items:
            if hasattr(item, "role"):
                if item.role == "assistant":
                    tool_calls: list[ToolCall] = []
                    if hasattr(item, "tool_calls") and item.tool_calls:
                        tool_calls = [
                            ToolCall.from_openai(tc) if isinstance(tc, dict) else tc
                            for tc in item.tool_calls
                        ]
                    turns.append(
                        AgentTurn.assistant(
                            content=getattr(item, "content", "") or "",
                            tool_calls=tool_calls if tool_calls else None,
                        )
                    )
                elif item.role == "tool":
                    turns.append(
                        AgentTurn.tool(
                            [
                                ToolResult(
                                    tool_call_id=getattr(item, "tool_call_id", ""),
                                    content=getattr(item, "content", "") or "",
                                )
                            ]
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

    # -------------------------------------------------------------------------
    # Compute metrics with agent-specific handling
    # -------------------------------------------------------------------------

    def compute_metrics(self, responses: Sequence[Response]) -> dict[str, float]:
        """Compute metrics from scored responses.

        For agent tasks, this extracts the results from responses and calls
        _compute_metrics. Subclasses should implement _compute_metrics rather
        than overriding this method.
        """
        # For agent tasks, we need to convert responses back to results
        # This is a fallback - normally _run_agent_task_impl handles this
        results = []
        for resp in responses:
            if resp.trajectory is not None:
                results.append(
                    AgentExecutionResult(
                        trajectory=resp.trajectory,
                        final_answer=resp.trajectory.final_answer,
                        success=not resp.outputs[0].metadata.get("error") if resp.outputs else True,
                        error=resp.outputs[0].metadata.get("error") if resp.outputs else None,
                    )
                )
        return self._compute_metrics(results, instances=list(responses))

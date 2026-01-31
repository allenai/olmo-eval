"""SimpleQA agent evaluation task with search tools.

This module implements a SimpleQA evaluation where an agent can use search
tools to answer questions, following the pattern from the OpenAI SimpleQA
benchmark.
"""

import os
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from olmo_eval.core.metrics import AccuracyMetric
from olmo_eval.core.scorers import SimpleQAJudgeScorer
from olmo_eval.core.types import SEARCH_TOOL_NAMES, SEARCH_TOOLS, Instance, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import AgentTask, AgentTaskConfig, register

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that can search for information to answer questions accurately.

When answering questions:
1. If you're unsure about a fact, use the available search tools to find accurate information.
2. Provide concise, accurate answers based on the information you find.
3. If you cannot find reliable information, say so rather than guessing.

Always strive to give factually correct answers."""


class SimpleQAAgentTask(AgentTask):
    """SimpleQA evaluation with search tools.

    This task evaluates a model's ability to answer factual questions
    using search tools. The agent can use semantic scholar and web search
    to find relevant information before providing an answer.

    The task uses an LLM judge to evaluate whether the final answer is
    CORRECT, INCORRECT, or NOT_ATTEMPTED.
    """

    default_source: str = "allenai/simpleqa_full"
    fewshot_split: str = "test"  # SimpleQA only has test split

    def __init__(self, config: AgentTaskConfig) -> None:
        super().__init__(config)

    @property
    def instances(self) -> Iterator[Instance]:
        """Yield instances from the dataset."""
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            try:
                source = self.config.get_data_source()
            except ValueError:
                source = DataSource(path=self.default_source, split="test")

            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)

        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a dataset document to an Instance with tools."""
        # Handle different possible field names
        question = doc.get("question") or doc.get("problem") or ""
        gold_answer = doc.get("answer") or doc.get("ground_truth") or doc.get("gold_answer") or ""

        if not question:
            return None

        return Instance(
            question=question,
            gold_answer=gold_answer,
            tools=SEARCH_TOOLS,
            metadata={
                "id": doc.get("id", f"simpleqa_{index}"),
                "index": index,
                "dataset": "simpleqa",
            },
        )

    @asynccontextmanager
    async def _get_agent(
        self,
        model: str,
        model_url: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Create agent with search tools via MCP."""
        from agents import (  # type: ignore[import-not-found]
            Agent,
            ModelSettings,
            OpenAIChatCompletionsModel,
        )
        from agents.mcp import MCPServerStdio  # type: ignore[import-not-found]
        from agents.tool_filter import create_static_tool_filter  # type: ignore[import-not-found]
        from openai import AsyncOpenAI  # type: ignore[import-not-found]

        s2_api_key = os.getenv("S2_API_KEY")
        if not s2_api_key:
            raise ValueError("S2_API_KEY environment variable is required.")
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            raise ValueError("SERPER_API_KEY environment variable is required.")

        client = AsyncOpenAI(
            base_url=model_url or "http://localhost:8000/v1",
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )
        llm = OpenAIChatCompletionsModel(openai_client=client, model=model)
        model_settings = ModelSettings(temperature=temperature)

        tool_filter = create_static_tool_filter(allowed_tool_names=list(SEARCH_TOOL_NAMES))
        env = {
            "S2_API_KEY": s2_api_key,
            "SERPER_API_KEY": serper_api_key,
        }

        async with MCPServerStdio(
            cache_tools_list=True,
            tool_filter=tool_filter,
            client_session_timeout_seconds=60,
            params={
                "command": "python",
                "args": ["-m", "dr_agent.mcp_backend.main", "--transport", "stdio"],
                "env": env,
            },
        ) as server:
            agent = Agent(
                name="SearchAgent",
                instructions=system_prompt or DEFAULT_SYSTEM_PROMPT,
                model=llm,
                model_settings=model_settings,
                mcp_servers=[server],
            )
            yield agent


# =============================================================================
# Task Configuration
# =============================================================================


def _simpleqa_agent_config() -> AgentTaskConfig:
    """Create default configuration for SimpleQA agent task.

    This task REQUIRES a model to be specified via CLI. There is no default model.

    Usage examples:
        # With a HuggingFace model (starts vLLM server)
        olmo-eval run -m llama3.1-8b-instruct -t simpleqa_agent

        # With an API model preset
        olmo-eval run -m gpt-4o -t simpleqa_agent

        # With custom API endpoint
        olmo-eval run -m my-model::model_url=http://localhost:8000/v1 -t simpleqa_agent

    Optional environment variables:
    - OPENAI_API_KEY: Required for API models (gpt-4o, etc.) and LLM judge grading
    - S2_API_KEY: Required for Semantic Scholar search tool
    - SERPER_API_KEY: Required for web search tool
    """
    return AgentTaskConfig(
        name="simpleqa_agent",
        data_source=DataSource(path="allenai/simpleqa_full", split="test"),
        scorers=(SimpleQAJudgeScorer(),),
        metrics=(AccuracyMetric(),),
        sampling_params=SamplingParams(max_tokens=2048, temperature=0.0),
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        max_turns=10,
        max_concurrency=1,
        required_secrets=("S2_API_KEY", "SERPER_API_KEY"),
    )


# =============================================================================
# Task Registration
# =============================================================================


@register("simpleqa_agent", _simpleqa_agent_config)
class SimpleQAAgent(SimpleQAAgentTask):
    """SimpleQA evaluation with search tools."""

    pass

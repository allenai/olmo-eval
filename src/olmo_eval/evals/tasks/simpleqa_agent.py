"""SimpleQA agent evaluation task with search tools.

This module implements a SimpleQA evaluation where an agent can use search
tools to answer questions, following the pattern from the OpenAI SimpleQA
benchmark.
"""

from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from olmo_eval.core.agents import AgentConfig, AgentExecutionResult
from olmo_eval.core.metrics import AccuracyMetric
from olmo_eval.core.scorers import JudgeFn, SimpleQAJudgeScorer
from olmo_eval.core.types import (
    Instance,
    LMOutput,
    SamplingParams,
    ToolSchema,
)
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import AgentTask, AgentTaskConfig, register

# Default system prompt for the search agent
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful assistant that can search for information to answer questions accurately.

When answering questions:
1. If you're unsure about a fact, use the available search tools to find accurate information.
2. Provide concise, accurate answers based on the information you find.
3. If you cannot find reliable information, say so rather than guessing.

Always strive to give factually correct answers."""

# Tool schemas for search (these match the MCP tools that will be available)
SEARCH_TOOLS = (
    ToolSchema(
        name="semantic_scholar_snippet_search",
        description="Search Semantic Scholar for academic papers and snippets matching a query.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for academic papers and snippets.",
                },
            },
            "required": ["query"],
        },
    ),
    ToolSchema(
        name="web_search",
        description="Search the web for information using a search engine.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant web pages.",
                },
            },
            "required": ["query"],
        },
    ),
)


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
        """Create agent with search tools via MCP.

        This sets up an agent with MCP-based search tools (Semantic Scholar
        and web search) for answering questions.
        """
        from agents import Agent, OpenAIChatCompletionsModel  # type: ignore[import-not-found]
        from agents.mcp import MCPServerStdio  # type: ignore[import-not-found]
        from openai import AsyncOpenAI  # type: ignore[import-not-found]

        # Validate required secrets
        self._validate_secrets()

        # Create OpenAI-compatible client
        client = AsyncOpenAI(
            base_url=model_url or "http://localhost:8000/v1",
            api_key="EMPTY",  # For vLLM or similar
        )
        llm = OpenAIChatCompletionsModel(openai_client=client, model=model)

        # Start MCP server for search tools
        # Note: The actual MCP server implementation would be provided separately
        async with MCPServerStdio(
            params={
                "command": "python",
                "args": ["-m", "search_mcp"],
                "env": {
                    "S2_API_KEY": kwargs.get("s2_api_key", ""),
                    "SERPER_API_KEY": kwargs.get("serper_api_key", ""),
                },
            },
        ) as server:
            agent = Agent(
                name="SearchAgent",
                instructions=system_prompt or DEFAULT_SYSTEM_PROMPT,
                model=llm,
                mcp_servers=[server],
            )
            yield agent

    def _compute_metrics(
        self,
        results: list[AgentExecutionResult],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute metrics from execution results using LLM judge."""
        instances = kwargs.get("instances", [])

        # Build judge function if we have the required API key
        import os

        judge_fn: JudgeFn | None = None
        if os.getenv("OPENAI_API_KEY"):
            judge_fn = self._build_judge_fn()

        scorer = SimpleQAJudgeScorer(judge_fn=judge_fn)

        # Track grades
        grades = {"CORRECT": 0, "INCORRECT": 0, "NOT_ATTEMPTED": 0}
        scores: list[float] = []

        for i, result in enumerate(results):
            if not result.success or not result.final_answer:
                grades["NOT_ATTEMPTED"] += 1
                scores.append(0.0)
                continue

            # Get corresponding instance
            instance = instances[i] if i < len(instances) else Instance(question="", gold_answer="")

            # Create LMOutput for scoring
            output = LMOutput(
                text=result.final_answer,
                extracted_answer=result.final_answer,
            )

            if judge_fn:
                # Use judge to determine grade
                prompt = scorer.format_judge_prompt(instance, output)
                response = judge_fn(prompt)
                grade = scorer.get_grade(response)
                grades[grade] += 1
                scores.append(scorer.parse_judge_response(response))
            else:
                # Without judge, mark as correct if answer exists
                scores.append(1.0 if result.final_answer else 0.0)
                grades["CORRECT" if result.final_answer else "NOT_ATTEMPTED"] += 1

        total = len(results)
        return {
            "accuracy": sum(scores) / total if total else 0.0,
            "correct_rate": grades["CORRECT"] / total if total else 0.0,
            "incorrect_rate": grades["INCORRECT"] / total if total else 0.0,
            "not_attempted_rate": grades["NOT_ATTEMPTED"] / total if total else 0.0,
            "success_rate": sum(1 for r in results if r.success) / total if total else 0.0,
            "num_instances": float(total),
        }

    def _build_judge_fn(self) -> JudgeFn:
        """Build a judge function using OpenAI API."""
        import os

        from openai import OpenAI  # type: ignore[import-not-found]

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        def judge(prompt: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            return response.choices[0].message.content or ""

        return judge


# =============================================================================
# Task Configuration
# =============================================================================


def _simpleqa_agent_config() -> AgentTaskConfig:
    """Create default configuration for SimpleQA agent task."""
    return AgentTaskConfig(
        name="simpleqa_agent",
        data_source=DataSource(path="allenai/simpleqa_full", split="test"),
        scorers=(),  # Scoring handled by _compute_metrics
        metrics=(AccuracyMetric(),),
        sampling_params=SamplingParams(max_tokens=2048, temperature=0.0),
        agent_config=AgentConfig(
            model="gpt-4o-mini",
            model_url="https://api.openai.com/v1",
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=5,
            temperature=0.0,
            max_tokens=2048,
        ),
        required_secrets=(
            "OPENAI_API_KEY",
            "S2_API_KEY",
            "SERPER_API_KEY",
        ),
    )


# =============================================================================
# Task Registration
# =============================================================================


@register("simpleqa_agent", _simpleqa_agent_config)
class SimpleQAAgent(SimpleQAAgentTask):
    """SimpleQA evaluation with search tools."""

    pass

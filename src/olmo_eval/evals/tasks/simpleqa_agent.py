"""SimpleQA agent evaluation task with search tools.

This module implements a SimpleQA evaluation where an agent can use search
tools to answer questions, following the pattern from the OpenAI SimpleQA
benchmark.
"""

import logging
import os
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from olmo_eval.core.debug import create_debug_http_client
from olmo_eval.core.formatters import ChatFormatter
from olmo_eval.core.metrics import AccuracyMetric
from olmo_eval.core.scorers import SimpleQAJudgeScorer
from olmo_eval.core.types import SEARCH_TOOLS, Instance, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.tasks.core import AgentTask, AgentTaskConfig, register
from olmo_eval.inference.utils import patch_openai_agents_for_vllm

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Implementations
# =============================================================================


async def semantic_scholar_snippet_search(query: str) -> str:
    """Search Semantic Scholar for academic papers and snippets matching a query.

    Args:
        query: Search query for academic papers and snippets.

    Returns:
        Formatted search results with paper titles, abstracts, and URLs.
    """
    api_key = os.getenv("S2_API_KEY")
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={"query": query, "limit": 5, "fields": "title,abstract,url,year,authors"},
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Error searching Semantic Scholar: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error searching Semantic Scholar: {e}"

    papers = data.get("data", [])
    if not papers:
        return "No papers found for query."

    results = []
    for paper in papers:
        title = paper.get("title", "Unknown")
        abstract = paper.get("abstract", "No abstract available")
        url = paper.get("url", "")
        year = paper.get("year", "")
        authors = paper.get("authors", [])
        author_names = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_names += " et al."

        result = f"**{title}**"
        if year:
            result += f" ({year})"
        if author_names:
            result += f"\nAuthors: {author_names}"
        if abstract:
            # Truncate long abstracts
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            result += f"\nAbstract: {abstract}"
        if url:
            result += f"\nURL: {url}"
        results.append(result)

    return "\n\n---\n\n".join(results)


async def serper_google_webpage_search(query: str) -> str:
    """Search the web for information using Google via Serper.

    Args:
        query: The search query to find relevant web pages.

    Returns:
        Formatted search results with titles, snippets, and URLs.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not configured."

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": 5},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Error searching web: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error searching web: {e}"

    results = []

    # Process organic results
    organic = data.get("organic", [])
    for item in organic[:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        link = item.get("link", "")
        result = f"**{title}**\n{snippet}\nURL: {link}"
        results.append(result)

    # Include knowledge graph if available
    kg = data.get("knowledgeGraph")
    if kg:
        kg_title = kg.get("title", "")
        kg_desc = kg.get("description", "")
        if kg_title and kg_desc:
            results.insert(0, f"**Knowledge Graph: {kg_title}**\n{kg_desc}")

    # Include answer box if available
    answer_box = data.get("answerBox")
    if answer_box:
        answer = answer_box.get("answer") or answer_box.get("snippet", "")
        if answer:
            results.insert(0, f"**Direct Answer:**\n{answer}")

    if not results:
        return "No search results found."

    return "\n\n---\n\n".join(results)


async def serper_fetch_webpage_content(url: str) -> str:
    """Fetch and extract content from a webpage URL.

    Args:
        url: The URL of the webpage to fetch.

    Returns:
        Extracted text content from the webpage.
    """
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not configured."

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(
                "https://scrape.serper.dev",
                json={"url": url},
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            return f"Error fetching webpage: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Error fetching webpage: {e}"

    # Extract text content
    text = data.get("text", "")
    if not text:
        return "No content extracted from webpage."

    # Truncate if too long
    if len(text) > 4000:
        text = text[:4000] + "\n\n[Content truncated...]"

    return text


def _create_function_tools() -> list[Any]:
    """Create function tools for the agent.

    Returns:
        List of function tools decorated with @function_tool.
    """
    from agents import function_tool  # type: ignore[import-not-found]

    # Use strict_mode=False for vLLM compatibility.
    # vLLM doesn't support the 'strict' field in tool schemas.
    # See: https://github.com/vllm-project/vllm/issues/27746
    return [
        function_tool(strict_mode=False)(semantic_scholar_snippet_search),
        function_tool(strict_mode=False)(serper_google_webpage_search),
        function_tool(strict_mode=False)(serper_fetch_webpage_content),
    ]


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
        # The dataset may have question directly, or in messages format
        question = doc.get("question") or doc.get("problem") or ""

        # Handle messages format: [{"role": "user", "content": "..."}]
        if not question and "messages" in doc:
            messages = doc["messages"]
            if messages and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict) and first_msg.get("role") == "user":
                    question = first_msg.get("content", "")

        gold_answer = doc.get("answer") or doc.get("ground_truth") or doc.get("gold_answer") or ""

        if not question:
            return None

        return Instance(
            question=question,
            gold_answer=gold_answer,
            tools=self.config.tools or None,
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
        """Create agent with search tools as direct function tools."""
        from agents import (  # type: ignore[import-not-found]
            Agent,
            ModelSettings,
            OpenAIChatCompletionsModel,
            set_tracing_disabled,
        )
        from openai import AsyncOpenAI  # type: ignore[import-not-found]

        # Enable/disable tracing based on debug mode
        # When debugging, keep tracing on; otherwise disable (we're using vLLM, not OpenAI)
        debug_mode = os.getenv("OLMO_EVAL_DEBUG_REQUESTS")
        set_tracing_disabled(not debug_mode)

        # Enable debug logging for agents SDK when debug mode is on
        if debug_mode:
            import logging as _logging

            agents_logger = _logging.getLogger("openai.agents")
            agents_logger.setLevel(_logging.DEBUG)
            if not agents_logger.handlers:
                agents_logger.addHandler(_logging.StreamHandler())

            # Also enable httpx debug logging
            httpx_logger = _logging.getLogger("httpx")
            httpx_logger.setLevel(_logging.DEBUG)
            if not httpx_logger.handlers:
                httpx_logger.addHandler(_logging.StreamHandler())

        # Patch SDK to omit 'strict' field for vLLM compatibility
        patch_openai_agents_for_vllm()

        s2_api_key = os.getenv("S2_API_KEY")
        if not s2_api_key:
            raise ValueError("S2_API_KEY environment variable is required.")
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            raise ValueError("SERPER_API_KEY environment variable is required.")

        # Create httpx client with request/response logging for debugging
        # Enable with OLMO_EVAL_DEBUG_REQUESTS=1
        http_client = create_debug_http_client()

        # Disable retries in debug mode to see actual errors
        max_retries = 0 if debug_mode else 2

        client = AsyncOpenAI(
            base_url=model_url or "http://localhost:8000/v1",
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            http_client=http_client,
            max_retries=max_retries,
            timeout=60.0,  # 60 second timeout
        )
        llm = OpenAIChatCompletionsModel(openai_client=client, model=model)
        model_settings = ModelSettings(temperature=temperature)

        # Create function tools directly instead of using MCP
        tools = _create_function_tools()

        # Use provided system_prompt, or fall back to task's formatter, or default
        effective_prompt = system_prompt or self.system_prompt or DEFAULT_SYSTEM_PROMPT
        agent = Agent(
            name="SearchAgent",
            instructions=effective_prompt,
            model=llm,
            model_settings=model_settings,
            tools=tools,
        )
        yield agent


# =============================================================================
# Task Configuration
# =============================================================================


def _simpleqa_agent_config() -> AgentTaskConfig:
    """Create default configuration for SimpleQA agent task."""
    return AgentTaskConfig(
        name="simpleqa_agent",
        data_source=DataSource(path="allenai/simpleqa_full", split="test"),
        formatter=ChatFormatter(system_prompt=DEFAULT_SYSTEM_PROMPT),
        metrics=(AccuracyMetric(scorer=SimpleQAJudgeScorer),),
        sampling_params=SamplingParams(max_tokens=2048, temperature=0.0),
        max_turns=10,
        max_concurrency=1,
        required_secrets=("OPENAI_API_KEY", "S2_API_KEY", "SERPER_API_KEY"),
        tools=SEARCH_TOOLS,
    )


# =============================================================================
# Task Registration
# =============================================================================


@register("simpleqa_agent", _simpleqa_agent_config)
class SimpleQAAgent(SimpleQAAgentTask):
    """SimpleQA evaluation with search tools."""

    pass

"""Pre-built harness configurations.

Presets are accessed via `HarnessPresets.name` or `get_harness_preset("name")`.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from olmo_eval.common.constants import BEAKER_RESULT_DIR, LOCAL_RESULT_DIR
from olmo_eval.common.types import ProviderKind
from olmo_eval.harness.sandbox import Capability
from olmo_eval.inference.metrics import MetricsConfig
from olmo_eval.runners.asynq.batching import BatchConfig

from .config import HarnessConfig, ProviderConfig
from .constants import (
    CODE_COMPLETION_SYSTEM_PROMPT,
    CODING_AGENT_SYSTEM_PROMPT,
    DR_TULU_SYSTEM_PROMPT,
)

WEB_SEARCH_SYSTEM_PROMPT = """\
You are a web-search assistant for open-domain attributed question answering.

Only these tool names exist; use them verbatim:
- serper_google_webpage_search
- serper_fetch_webpage_content

Search first with serper_google_webpage_search, then fetch promising pages with
serper_fetch_webpage_content before answering. Quote only text copied from
fetched page content, and do not invent citations."""


# The delimiter contract for deepscholar_bench's single arm. The benchmark's
# user prompt is run verbatim and cannot be edited, and the 9B run showed why
# something had to give: 45 of 63 answers reached their Related Works section
# only after a median 9.8k characters of planning, all of which the scorer read
# as the report. Banning deliberation would cost quality, so this buys the
# split instead -- think above the line, deliver below it. Kept in sync with
# deepscholar_citations.split_final_report by a test that reads the marker back
# out of this string.
#
# The wording is emphatic about where the prose goes because the first version
# was not, and Qwen3.5-9B read it the other way: it wrote the whole Related
# Works section while deliberating and put only the numbered list below the
# line, so the split threw its report away. Saying 'your deliverable is what
# follows' was not enough -- it has to say the section itself must not appear
# above the marker.
ARXIV_PAPER_SEARCH_SYSTEM_PROMPT = """\
You may plan and deliberate freely in your reply. When you are ready to deliver,
write a line containing exactly:

=== FINAL REPORT ===

Your entire report goes after that line, in this order: the complete Related
Works section first, then the numbered reference list. Do not write any part of
the Related Works section before the marker, and do not put the reference list
there on its own. Everything above the marker is discarded and is never
evaluated, so a report written above it does not count at all."""


# TODO(undfined): Remove reference to beaker
def _get_logs_dir() -> str:
    """Get the logs directory based on environment."""
    result_dir = BEAKER_RESULT_DIR if os.environ.get("BEAKER_JOB_ID") else LOCAL_RESULT_DIR
    return os.path.join(result_dir, "logs")


# ─────────────────────────────────────────────────────────
# Lazy Descriptor
# ─────────────────────────────────────────────────────────


class Lazy:
    """Descriptor for lazily-loaded presets with auto-injected name."""

    def __init__(self, factory: Callable[[str], HarnessConfig]):
        self._factory = factory
        self._cached: HarnessConfig | None = None
        self._name: str = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> HarnessConfig:
        if self._cached is None:
            self._cached = self._factory(self._name)
        return self._cached


def lazy(fn: Callable[[str], HarnessConfig]) -> Lazy:
    """Mark a preset factory for lazy loading. Factory receives preset name."""
    return Lazy(fn)


# ─────────────────────────────────────────────────────────
# Preset Harness Configurations
# ─────────────────────────────────────────────────────────


class HarnessPresets:
    """Harness presets. Access as HarnessPresets.name or get_harness_preset("name")."""

    @lazy
    def default(name: str) -> HarnessConfig:
        """Default preset with vllm_server and batched processing."""

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(kind=ProviderKind.VLLM_SERVER),
            metrics=MetricsConfig(),
            batching=BatchConfig.batched(),
        )

    @lazy
    def simple_agent(name: str) -> HarnessConfig:
        """Simple agent preset."""
        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 60},
            ),
            metrics=MetricsConfig(),
            scaffold="openai_agents",
            max_concurrency=4,
            batching=BatchConfig.streaming(),
        )

    @lazy
    def dr_tulu(name: str) -> HarnessConfig:
        """Dr. Tulu preset with web and academic search tools."""
        from .tools.search import semantic_scholar_search, serper_fetch_page, serper_web_search

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(semantic_scholar_search, serper_web_search, serper_fetch_page),
            system_prompt=DR_TULU_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=4,
            scaffold="openai_agents",
            required_secrets=("S2_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY"),
            batching=BatchConfig.streaming(),
        )

    @lazy
    def dr_tulu_crawl4ai(name: str) -> HarnessConfig:
        """Dr. Tulu-style S2 plus web-search harness that browses with in-process
        crawl4ai.

        Install with ``pip install 'olmo-eval[crawl4ai]'``, then run
        ``crawl4ai-setup`` once to provision the headless browser. ``S2_API_KEY``
        is not required but is strongly recommended for throughput because
        keyless S2 is rate-limited to about 1 request/second process-wide.
        """
        from .tools.search import crawl4ai_browse, semantic_scholar_search, serper_web_search

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(semantic_scholar_search, serper_web_search, crawl4ai_browse),
            system_prompt=DR_TULU_SYSTEM_PROMPT,
            # Agentic paper search needs more turns than the dr_tulu default of 10.
            max_turns=20,
            max_concurrency=4,
            scaffold="openai_agents",
            required_secrets=("SERPER_API_KEY",),
            batching=BatchConfig.streaming(),
        )

    @lazy
    def paper_search_agent(name: str) -> HarnessConfig:
        """Agentic harness exposing only Semantic Scholar paper search.

        For literature-search tasks (e.g. litsearch). Exposes a single paper-search
        tool and declares no required secrets, so it runs against the public
        Semantic Scholar API keyless (rate-limited). For higher rate limits, mount
        a key with `--secret-env <user>_S2_API_KEY:S2_API_KEY`. Additional keys
        mount as `S2_API_KEY_2`, `S2_API_KEY_3`, ... (or a comma-separated list
        in any one of them); the tool spreads requests across all of them and
        raises its request rate proportionally.
        """
        from .tools.search import semantic_scholar_search

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(semantic_scholar_search,),
            max_turns=10,
            max_concurrency=4,
            scaffold="openai_agents",
            batching=BatchConfig.streaming(),
        )

    @lazy
    def arxiv_paper_search_agent(name: str) -> HarnessConfig:
        """Agentic harness exposing only arXiv-filtered paper search.

        The system prompt carries a delimiter contract: the model may deliberate
        in its reply, but its deliverable starts at an exact marker line and
        everything above that line is discarded before scoring. See
        ARXIV_PAPER_SEARCH_SYSTEM_PROMPT and
        deepscholar_citations.split_final_report, which reads the marker back
        out.

        For tasks whose scorer credits arXiv sources alone (e.g.
        deepscholar_bench). Kept separate from `paper_search_agent` rather than
        added to it: which search tools an agent holds changes what it retrieves,
        so sharing a preset would move litsearch and SAGE numbers that were
        measured against Semantic Scholar search.

        No secret is declared, so this runs keyless against the public Semantic
        Scholar API at its shared ~1 request/second. `required_secrets` is a
        hard launch gate with no optional form, and declaring one here would
        make a keyless run impossible; mount a key per run instead with
        `--secret-env <user>_S2_API_KEY:S2_API_KEY`.

        Reasoning effort is deliberately not pinned here, because backbones that
        reason well on this preset should keep doing so. Models that refuse
        function tools alongside a reasoning effort -- gpt-5.6-sol on
        /v1/chat/completions rejects the combination outright -- need it turned
        off for the run:
        `-o scaffold_kwargs.model_settings.reasoning_effort=none`.
        """
        from .tools.search import arxiv_paper_search

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(arxiv_paper_search,),
            system_prompt=ARXIV_PAPER_SEARCH_SYSTEM_PROMPT,
            # Writing a related-works section takes more searching than a
            # single-answer lookup, so this doubles paper_search_agent's turns.
            max_turns=20,
            max_concurrency=4,
            scaffold="openai_agents",
            batching=BatchConfig.streaming(),
        )

    @lazy
    def web_search_agent(name: str) -> HarnessConfig:
        """Agentic harness exposing only web search/fetch tools.

        For open-domain attributed-QA tasks (e.g. expertqa). Web-only search
        keeps non-science domains searchable and prevents models from
        substituting paper search for general evidence.
        """
        from .tools.search import serper_fetch_page, serper_web_search

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(serper_web_search, serper_fetch_page),
            system_prompt=WEB_SEARCH_SYSTEM_PROMPT,
            max_turns=30,
            max_concurrency=4,
            scaffold="openai_agents",
            required_secrets=("SERPER_API_KEY",),
            batching=BatchConfig.streaming(),
        )

    @lazy
    def web_search_agent_crawl4ai(name: str) -> HarnessConfig:
        """Web search via Serper + in-process crawl4ai browsing (no hosted scrape service).

        Install with ``pip install 'olmo-eval[crawl4ai]'``, then run
        ``crawl4ai-setup`` once to provision the headless browser.
        """
        from .tools.search import crawl4ai_browse, serper_web_search

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(serper_web_search, crawl4ai_browse),
            system_prompt="""\
You are a helpful assistant that can search and browse webpages to answer questions accurately.

You have access to these tools:
- serper_google_webpage_search: Search the web for relevant webpages.
- browse_webpage: Fetch and extract a webpage's content as clean markdown.

When answering questions:
1. Use serper_google_webpage_search to find relevant sources when needed.
2. Use browse_webpage to inspect promising URLs.
3. Provide concise, accurate answers based on the information you find.
4. If you cannot find reliable information, say so rather than guessing.

Always strive to give factually correct answers.""",
            max_turns=30,
            max_concurrency=4,
            scaffold="openai_agents",
            required_secrets=("SERPER_API_KEY",),
            batching=BatchConfig.streaming(),
        )

    @lazy
    def codex_universal(name: str) -> HarnessConfig:
        """Universal code execution preset with multiple capabilities."""
        from .sandbox import SandboxConfig, SandboxMode

        return HarnessConfig(
            name=name,
            metrics=MetricsConfig(),
            sandboxes=(
                SandboxConfig(
                    image="volcengine/sandbox-fusion:base-20250609",
                    mode=SandboxMode.DOCKER,
                    startup_timeout=300.0,
                    log_dir=_get_logs_dir(),
                    inject_swerex=True,
                    dockerfile_extra=(
                        "RUN mkdir -p /runtime/java",
                        "RUN curl -L -o /runtime/java/javatuples-1.2.jar https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar",
                    ),
                ),
                SandboxConfig(
                    image="bigcodebench/bigcodebench-gradio:latest",
                    mode=SandboxMode.DOCKER,
                    capabilities=frozenset({"sandbox:bigcodebench"}),
                    startup_timeout=300.0,
                    log_dir=_get_logs_dir(),
                    inject_swerex=True,
                ),
            ),
        )

    @lazy
    def codex_python(name: str) -> HarnessConfig:
        """Python only code execution preset."""
        from .sandbox import SandboxConfig, SandboxMode

        return HarnessConfig(
            name=name,
            metrics=MetricsConfig(),
            scoring_concurrency=4,
            sandboxes=(
                SandboxConfig(
                    instances=4,
                    image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
                    mode=SandboxMode.DOCKER,
                    startup_timeout=60.0,
                    log_dir=_get_logs_dir(),
                ),
            ),
        )

    @lazy
    def codex_agent(name: str) -> HarnessConfig:
        """Coding agent preset with sandboxed shell execution."""
        from .sandbox import SandboxConfig, SandboxMode
        from .tools.search import serper_fetch_page, serper_web_search
        from .tools.shell import execute_bash

        return HarnessConfig(
            name=name,
            metrics=MetricsConfig(),
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                # Higher timeout for multi-turn agent runs (each turn can take time)
                kwargs={"timeout": 300},
            ),
            tools=(execute_bash, serper_fetch_page, serper_web_search),
            system_prompt=CODING_AGENT_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=4,
            scaffold="openai_agents",
            required_secrets=("OPENAI_API_KEY",),
            sandboxes=(
                SandboxConfig(
                    capabilities=frozenset(Capability.BASH),
                    instances=4,  # Match max_concurrency for parallel execution
                    image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
                    mode=SandboxMode.DOCKER,
                    startup_timeout=120.0,
                    log_dir=_get_logs_dir(),
                ),
            ),
            batching=BatchConfig.streaming(),
        )

    @lazy
    def codex_completion(name: str) -> HarnessConfig:
        """Code completion agent with sandbox for testing and web search."""
        from .sandbox import SandboxConfig, SandboxMode
        from .tools.search import serper_fetch_page, serper_web_search
        from .tools.shell import execute_bash

        return HarnessConfig(
            name=name,
            metrics=MetricsConfig(),
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 300},
            ),
            tools=(execute_bash, serper_fetch_page, serper_web_search),
            system_prompt=CODE_COMPLETION_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=16,
            scaffold="openai_agents",
            required_secrets=("OPENAI_API_KEY",),
            sandboxes=(
                SandboxConfig(
                    capabilities=frozenset(Capability.BASH),
                    instances=1,
                    image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
                    mode=SandboxMode.DOCKER,
                    startup_timeout=120.0,
                    log_dir=_get_logs_dir(),
                ),
            ),
            batching=BatchConfig.streaming(),
        )


# ─────────────────────────────────────────────────────────
# API Functions
# ─────────────────────────────────────────────────────────


def _is_preset(name: str) -> bool:
    """Check if a name is a valid preset (not private, is HarnessConfig or Lazy)."""
    if name.startswith("_"):
        return False
    attr = getattr(HarnessPresets, name, None)
    return isinstance(attr, (HarnessConfig, Lazy))


def get_harness_preset(name: str) -> HarnessConfig:
    """Get a harness preset by name."""
    if not hasattr(HarnessPresets, name) or not _is_preset(name):
        available = ", ".join(list_harness_presets())
        raise ValueError(f"Unknown harness preset: '{name}'. Available: {available}")
    return getattr(HarnessPresets, name)


def list_harness_presets() -> list[str]:
    """List all available harness preset names."""
    return sorted(name for name in dir(HarnessPresets) if _is_preset(name))


def register_harness_preset(name: str, config: HarnessConfig) -> None:
    """Register a harness preset directly."""
    setattr(HarnessPresets, name, config)

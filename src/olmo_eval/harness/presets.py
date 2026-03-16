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


# TODO(undfined): Remove reference to beaker
def _get_logs_dir() -> str:
    """Get the logs directory based on environment."""
    result_dir = BEAKER_RESULT_DIR if os.environ.get("BEAKER_JOB_ID") else LOCAL_RESULT_DIR
    return os.path.join(result_dir, "logs")


# ─────────────────────────────────────────────────────────
# Lazy Descriptor
# ─────────────────────────────────────────────────────────


class _Lazy:
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


def lazy(fn: Callable[[str], HarnessConfig]) -> _Lazy:
    """Mark a preset factory for lazy loading. Factory receives preset name."""
    return _Lazy(fn)


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
            backend="openai_agents",
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
            backend="openai_agents",
            required_secrets=("S2_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY"),
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
                    instances=16,
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
            backend="openai_agents",
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
            backend="openai_agents",
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
    """Check if a name is a valid preset (not private, is HarnessConfig or _Lazy)."""
    if name.startswith("_"):
        return False
    attr = getattr(HarnessPresets, name, None)
    return isinstance(attr, (HarnessConfig, _Lazy))


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

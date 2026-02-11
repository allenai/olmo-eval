"""Pre-built harness configurations.

Presets are accessed via `Presets.name` or `get_harness_preset("name")`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from olmo_eval.common.types import ProviderKind

from .config import HarnessConfig, ProviderConfig
from .constants import DR_TULU_SYSTEM_PROMPT

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
# System Prompts
# ─────────────────────────────────────────────────────────

CODING_AGENT_SYSTEM_PROMPT = """\
You are a helpful coding assistant with access to a sandboxed bash shell.

You can execute bash commands to:
- Run code and tests
- Install packages
- Manipulate files
- Explore the filesystem

Use the execute_bash tool to run commands. The environment is isolated,
so you can safely experiment.

When solving coding problems:
1. First understand the problem by reading any provided files
2. Write and test your solution incrementally
3. Verify your solution works before providing the final answer
"""


# ─────────────────────────────────────────────────────────
# Preset Harness Configurations
# ─────────────────────────────────────────────────────────


class HarnessPresets:
    """Harness presets. Access as HarnessPresets.name or get_harness_preset("name")."""

    default = HarnessConfig(name="default")

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
            max_concurrency=8,
            backend="openai_agents",
            required_secrets=("S2_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY"),
        )

    @lazy
    def codex_python(name: str) -> HarnessConfig:
        """Python only code execution preset."""
        from .sandbox import SandboxConfig, SandboxMode

        return HarnessConfig(
            name=name,
            sandbox=SandboxConfig(
                image="python:3.12",
                mode=SandboxMode.DOCKER,
                startup_timeout=60.0,
                docker_args=("--log-driver=json-file", "--log-opt", "path=/results/sandbox.log"),
            ),
        )

    @lazy
    def codex_universal(name: str) -> HarnessConfig:
        """Universal code execution preset."""
        from .sandbox import SandboxConfig, SandboxMode

        return HarnessConfig(
            name=name,
            sandbox=SandboxConfig(
                image="volcengine/sandbox-fusion:server-20250609",
                mode=SandboxMode.DOCKER,
                startup_timeout=60.0,
                docker_args=("--log-driver=json-file", "--log-opt", "path=/results/sandbox.log"),
            ),
        )

    @lazy
    def codex_agent(name: str) -> HarnessConfig:
        """Coding agent preset with sandboxed shell execution."""
        from .sandbox import SandboxConfig, SandboxMode
        from .tools.shell import execute_bash

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                kwargs={"timeout": 120},
            ),
            tools=(execute_bash,),
            system_prompt=CODING_AGENT_SYSTEM_PROMPT,
            max_turns=20,
            backend="openai_agents",
            sandbox=SandboxConfig(
                image="volcengine/sandbox-fusion:server-20250609",
                mode=SandboxMode.DOCKER,
                startup_timeout=60.0,
                docker_args=("--log-driver=json-file", "--log-opt", "path=/results/sandbox.log"),
            ),
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

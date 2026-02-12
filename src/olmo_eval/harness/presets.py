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

from .config import HarnessConfig, ProviderConfig
from .constants import DR_TULU_SYSTEM_PROMPT


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
# System Prompts
# ─────────────────────────────────────────────────────────

CODING_AGENT_SYSTEM_PROMPT = """\
You are a helpful coding assistant with access to a sandboxed bash shell.

You can execute bash commands to:
- Run code and tests
- Install packages
- Manipulate files
- Explore the filesystem

You can also search the web for documentation and examples using the provided tools.

Use the execute_bash tool to run commands. The environment is isolated,
so you can safely experiment.

IMPORTANT: When writing code with special characters (quotes, backslashes, newlines),
use a heredoc to write the code to a file, then run the file:

cat << 'EOF' > solution.py
# Your code here - special characters are preserved exactly
def example():
    return "hello\\nworld"
EOF
python solution.py

Do NOT use python -c or inline code with lots of special characters.

When solving coding problems:
1. First understand the problem by reading any provided files
2. Plan your approach and write code in a file using heredoc syntax
3. Run and test your solution
4. Verify it works before providing the final answer
"""

CODE_COMPLETION_SYSTEM_PROMPT = """\
You are a Python coding assistant that completes function implementations.

When given a function signature and docstring, write the implementation code that \
goes inside the function body. Output only valid Python code.

You have access to tools to help you:
- execute_bash: Run Python code in a sandbox to test your solution
- Web search: Look up documentation or examples if needed

Workflow:
1. Read the function signature and docstring carefully
2. Write the implementation code
3. Test your code using execute_bash to verify it works
4. Provide the final implementation

When testing code, write it to a file using heredoc syntax:

cat << 'EOF' > solution.py
# Your implementation here
EOF
python solution.py

Output only the function body code in your final answer - no explanations, \
markdown formatting, or the function signature itself.
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
            max_concurrency=4,
            backend="openai_agents",
            required_secrets=("S2_API_KEY", "SERPER_API_KEY", "OPENAI_API_KEY"),
        )

    @lazy
    def codex_python(name: str) -> HarnessConfig:
        """Python only code execution preset."""
        from .sandbox import SandboxConfig, SandboxMode

        return HarnessConfig(
            name=name,
            sandboxes=(
                SandboxConfig(
                    instances=3,
                    image="python:3.12",
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
                    image="python:3.12",
                    mode=SandboxMode.DOCKER,
                    startup_timeout=120.0,
                    log_dir=_get_logs_dir(),
                ),
            ),
        )

    @lazy
    def codex_completion(name: str) -> HarnessConfig:
        """Code completion agent with sandbox for testing and web search."""
        from .sandbox import SandboxConfig, SandboxMode
        from .tools.search import serper_fetch_page, serper_web_search
        from .tools.shell import execute_bash

        return HarnessConfig(
            name=name,
            provider=ProviderConfig(
                kind=ProviderKind.VLLM_SERVER,
                # Higher timeout for multi-turn agent runs (each turn can take time)
                kwargs={"timeout": 300},
            ),
            tools=(execute_bash, serper_fetch_page, serper_web_search),
            system_prompt=CODE_COMPLETION_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=4,
            backend="openai_agents",
            required_secrets=("OPENAI_API_KEY",),
            sandboxes=(
                SandboxConfig(
                    capabilities=frozenset(Capability.BASH),
                    image="python:3.12",
                    mode=SandboxMode.DOCKER,
                    startup_timeout=120.0,
                    log_dir=_get_logs_dir(),
                ),
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

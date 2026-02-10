"""Pre-built harness configurations.

Presets are registered using the @harness_preset decorator.
"""

from __future__ import annotations

from collections.abc import Callable

from olmo_eval.common.types import ProviderKind

from .config import HarnessConfig, ProviderConfig
from .constants import DR_TULU_SYSTEM_PROMPT

# ─────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────

# Maps preset name -> either a HarnessConfig or a factory that produces one
_PRESET_REGISTRY: dict[str, HarnessConfig | Callable[[], HarnessConfig]] = {}


def harness_preset(
    name: str,
) -> Callable[[Callable[[], HarnessConfig]], Callable[[], HarnessConfig]]:
    """Decorator to register a harness preset factory.

    The decorated function is called lazily on first access, allowing
    presets to defer heavy imports (like tool modules).

    Args:
        name: Name to register the preset under.

    Returns:
        Decorator that registers the factory function.
    """

    def decorator(fn: Callable[[], HarnessConfig]) -> Callable[[], HarnessConfig]:
        _PRESET_REGISTRY[name] = fn
        return fn

    return decorator


def register_harness_preset(name: str, config: HarnessConfig) -> None:
    """Register a harness preset directly (non-lazy).

    Args:
        name: Name to register the preset under.
        config: HarnessConfig to register.
    """
    _PRESET_REGISTRY[name] = config


def get_harness_preset(name: str) -> HarnessConfig:
    """Get a harness preset by name.

    Args:
        name: Name of the preset to retrieve.

    Returns:
        HarnessConfig for the requested preset.

    Raises:
        ValueError: If the preset name is unknown.
    """
    if name not in _PRESET_REGISTRY:
        available = ", ".join(sorted(_PRESET_REGISTRY.keys()))
        raise ValueError(f"Unknown harness preset: '{name}'. Available: {available}")

    entry = _PRESET_REGISTRY[name]
    if callable(entry):
        # Lazy initialization: call factory and cache result
        config = entry()
        _PRESET_REGISTRY[name] = config
        return config
    return entry


def list_harness_presets() -> list[str]:
    """List all available harness preset names.

    Returns:
        Sorted list of preset names.
    """
    return sorted(_PRESET_REGISTRY.keys())


# ─────────────────────────────────────────────────────────
# Preset Harness Configurations
# ─────────────────────────────────────────────────────────

register_harness_preset(
    "default",
    HarnessConfig(
        name="default",
    ),
)


@harness_preset("dr_tulu")
def _dr_tulu() -> HarnessConfig:
    """Dr. Tulu preset with web and academic search tools."""
    from .tools.search import semantic_scholar_search, serper_fetch_page, serper_web_search

    return HarnessConfig(
        name="dr_tulu",
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


# System prompt for coding agent
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


@harness_preset("code_execution")
def _code_execution() -> HarnessConfig:
    """Code execution preset for pass@k evaluation tasks.

    Enables sandboxed code execution for scorers without agent tools.
    Use with tasks like humaneval:pass_at_1, mbpp:pass_at_10, etc.
    """
    from .sandbox import SandboxConfig

    return HarnessConfig(
        name="code_execution",
        sandbox=SandboxConfig(
            image="volcengine/sandbox-fusion:server-20250609",
            deployment_mode="docker",
        ),
    )


@harness_preset("coding_agent")
def _coding_agent() -> HarnessConfig:
    """Coding agent preset with sandboxed shell execution."""
    from .sandbox import SandboxConfig
    from .tools.shell import execute_bash

    return HarnessConfig(
        name="coding_agent",
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
            deployment_mode="docker",
        ),
    )

"""Pre-built harness configurations for common use cases.

This module provides ready-to-use HarnessConfig presets:
- default: No tools, standard model behavior
- search: Web and academic search tools for factual QA

Presets are registered using the @harness_preset decorator.
"""

from __future__ import annotations

from collections.abc import Callable

from .config import HarnessBackend, HarnessConfig

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

    Available presets:
    - "default": No tools, standard model behavior
    - "search": Web and academic search tools for factual QA

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
# System Prompts
# ─────────────────────────────────────────────────────────

SEARCH_SYSTEM_PROMPT = """\
You are a helpful assistant that can search for information to answer questions accurately.

When answering questions:
1. If you're unsure about a fact, use the available search tools to find accurate information.
2. Provide concise, accurate answers based on the information you find.
3. If you cannot find reliable information, say so rather than guessing.

Always strive to give factually correct answers."""

# ─────────────────────────────────────────────────────────
# Preset Definitions
# ─────────────────────────────────────────────────────────

# Default preset: no tools, single turn
register_harness_preset(
    "default",
    HarnessConfig(
        name="default",
        tool_names=(),
        system_prompt=None,
        max_turns=1,
    ),
)


@harness_preset("search")
def _search_preset() -> HarnessConfig:
    """Search preset with web and academic search tools.

    Lazily imports search tools to avoid loading httpx etc unless needed.
    """
    # Import triggers @registered_tool decorators
    from .tools import search as _  # noqa: F401

    return HarnessConfig(
        name="search",
        tool_names=(
            "semantic_scholar_snippet_search",
            "serper_google_webpage_search",
            "serper_fetch_webpage_content",
        ),
        system_prompt=SEARCH_SYSTEM_PROMPT,
        max_turns=10,
        max_concurrency=8,
        backend=HarnessBackend.OPENAI_AGENTS,
        required_secrets=("S2_API_KEY", "SERPER_API_KEY"),
    )


# ─────────────────────────────────────────────────────────
# Backwards Compatibility
# ─────────────────────────────────────────────────────────

# Expose registry for tests that directly access HARNESS_PRESETS
HARNESS_PRESETS = _PRESET_REGISTRY

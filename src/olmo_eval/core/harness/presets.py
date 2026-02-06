"""Pre-built harness configurations for common use cases.

This module provides ready-to-use HarnessConfig presets:
- default: No tools, standard model behavior
- search: Web and academic search tools for factual QA
"""

from __future__ import annotations

from .config import HarnessConfig

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
# Harness Presets
# ─────────────────────────────────────────────────────────


def _ensure_search_tools_registered() -> tuple[str, ...]:
    """Ensure search tools are registered and return their names.

    This lazily imports and registers the search tools.
    """
    # Import triggers @registered_tool decorators
    from .tools.search import SEARCH_TOOL_NAMES

    return SEARCH_TOOL_NAMES


HARNESS_PRESETS: dict[str, HarnessConfig] = {
    "default": HarnessConfig(
        name="default",
        tool_names=(),
        system_prompt=None,
        max_turns=1,
    ),
}


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
    # Handle lazy initialization for presets with tools
    if name == "search" and "search" not in HARNESS_PRESETS:
        tool_names = _ensure_search_tools_registered()
        HARNESS_PRESETS["search"] = HarnessConfig(
            name="search",
            tool_names=tool_names,
            system_prompt=SEARCH_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=8,
            backend="openai_agents",
            required_secrets=("S2_API_KEY", "SERPER_API_KEY"),
        )

    if name not in HARNESS_PRESETS:
        available = ", ".join(sorted(HARNESS_PRESETS.keys()))
        raise ValueError(f"Unknown harness preset: '{name}'. Available: {available}")

    return HARNESS_PRESETS[name]


def list_harness_presets() -> list[str]:
    """List all available harness preset names.

    Returns:
        Sorted list of preset names.
    """
    # Ensure search is available
    if "search" not in HARNESS_PRESETS:
        _ensure_search_tools_registered()
        HARNESS_PRESETS["search"] = HarnessConfig(
            name="search",
            tool_names=_ensure_search_tools_registered(),
            system_prompt=SEARCH_SYSTEM_PROMPT,
            max_turns=10,
            max_concurrency=8,
            backend="openai_agents",
            required_secrets=("S2_API_KEY", "SERPER_API_KEY"),
        )

    return sorted(HARNESS_PRESETS.keys())


def register_harness_preset(name: str, config: HarnessConfig) -> None:
    """Register a custom harness preset.

    Args:
        name: Name to register the preset under.
        config: HarnessConfig to register.
    """
    HARNESS_PRESETS[name] = config

"""Tool registry and pre-built tools.

This package provides:
- Global tool registry for cross-process lookup
- Pre-built tools for common use cases (search, etc.)
"""

from .registry import (
    TOOL_REGISTRY,
    clear_registry,
    ensure_tools_registered,
    get_tool,
    get_tools,
    list_tools,
    register_tool,
    registered_tool,
)

__all__ = [
    # Registry
    "TOOL_REGISTRY",
    "clear_registry",
    "ensure_tools_registered",
    "get_tool",
    "get_tools",
    "list_tools",
    "register_tool",
    "registered_tool",
]

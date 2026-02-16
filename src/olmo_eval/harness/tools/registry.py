"""Global tool registry for cross-process lookup.

Tools contain callable functions which cannot be serialized across processes.
The registry allows tools to be registered by name, so configurations can
serialize just the tool names and workers can look them up at runtime.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import Tool

# Global registry mapping tool names to Tool instances
TOOL_REGISTRY: dict[str, Tool] = {}


def register_tool(tool: Tool) -> Tool:
    """Register a tool globally for cross-process lookup.

    Args:
        tool: The Tool instance to register.

    Returns:
        The same Tool instance (for chaining).

    Raises:
        ValueError: If a tool with the same name is already registered.
    """
    if tool.name in TOOL_REGISTRY:
        # Allow re-registration of the same tool (idempotent)
        if TOOL_REGISTRY[tool.name] is tool:
            return tool
        # Otherwise, warn but allow override (useful for testing)
        import warnings

        warnings.warn(
            f"Tool '{tool.name}' is being re-registered. This may cause unexpected behavior.",
            stacklevel=2,
        )

    TOOL_REGISTRY[tool.name] = tool
    return tool


_tools_loaded = False


def load_tools() -> None:
    """Import all built-in tool modules to register their tools.

    This is called automatically when accessing tools via get_tool(),
    but can be called explicitly to ensure all tools are registered.
    """
    global _tools_loaded
    if _tools_loaded:
        return

    import importlib
    import pkgutil
    import sys

    import olmo_eval.harness.tools as tools_pkg

    for module_info in pkgutil.iter_modules(tools_pkg.__path__):
        if module_info.name not in ("registry", "tool", "__init__"):
            full_name = f"{tools_pkg.__name__}.{module_info.name}"
            if full_name in sys.modules:
                # Reload to re-register tools after clear_registry()
                importlib.reload(sys.modules[full_name])
            else:
                importlib.import_module(f".{module_info.name}", tools_pkg.__name__)

    _tools_loaded = True


def get_tool(name: str) -> Tool:
    """Get a registered tool by name. Auto-loads built-in tools if needed.

    Args:
        name: The name of the tool to retrieve.

    Returns:
        The registered Tool instance.

    Raises:
        ValueError: If no tool with the given name is registered.
    """
    if name not in TOOL_REGISTRY:
        load_tools()

    if name not in TOOL_REGISTRY:
        available = ", ".join(sorted(TOOL_REGISTRY.keys())) or "(none)"
        raise ValueError(f"Unknown tool: '{name}'. Available: {available}")

    return TOOL_REGISTRY[name]


def get_tools(names: list[str] | tuple[str, ...]) -> tuple[Tool, ...]:
    """Get multiple registered tools by name.

    Args:
        names: Sequence of tool names to retrieve.

    Returns:
        Tuple of Tool instances in the same order as names.

    Raises:
        ValueError: If any tool name is not registered.
    """
    return tuple(get_tool(name) for name in names)


def list_tools() -> list[str]:
    """List all registered tool names.

    Returns:
        Sorted list of registered tool names.
    """
    return sorted(TOOL_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered tools and reset loading state.

    Primarily useful for testing.
    """
    global _tools_loaded
    TOOL_REGISTRY.clear()
    _tools_loaded = False


def registered_tool(
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
    sandbox: set[str] | frozenset[str] | None = None,
    session: bool = False,
) -> Callable[[Callable[..., Awaitable[str] | str]], Tool]:
    """Decorator to create and register a Tool from a function.

    Combines @tool and register_tool() in one step.

    Example:
        @registered_tool(description="Search the web for information")
        async def web_search(query: str) -> str:
            ...

    Args:
        name: Optional name override for the tool.
        description: Optional description override.
        strict: Whether to use OpenAI strict mode.
        sandbox: Required sandbox capabilities for execution.
        session: Whether tool requires persistent shell session.

    Returns:
        Decorator function that converts a function to a registered Tool.
    """
    from .tool import Tool

    def decorator(fn: Callable[..., Awaitable[str] | str]) -> Tool:
        t = Tool.from_function(
            fn,
            name=name,
            description=description,
            strict=strict,
            sandbox=sandbox,
            session=session,
        )
        return register_tool(t)

    # Handle @registered_tool without parentheses
    if callable(name):
        fn = name
        name = None
        from .tool import Tool

        t = Tool.from_function(fn)  # type: ignore[arg-type]
        return register_tool(t)  # type: ignore[return-value]

    return decorator


def ensure_tools_registered(tools: tuple[Tool, ...] | list[Tool]) -> tuple[str, ...]:
    """Ensure all tools are registered and return their names.

    This is a convenience function for creating HarnessConfig with Tool objects.
    It registers any unregistered tools and returns the list of names.

    Args:
        tools: Sequence of Tool instances to ensure are registered.

    Returns:
        Tuple of tool names corresponding to the input tools.
    """
    names = []
    for tool in tools:
        if tool.name not in TOOL_REGISTRY:
            register_tool(tool)
        names.append(tool.name)
    return tuple(names)

"""Tool class with unified schema and implementation.

This module provides the Tool class which combines both the schema (for LLM)
and implementation (for execution) in a single source of truth.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, get_type_hints

from olmo_eval.common.types import ToolSchema

F = TypeVar("F", bound=Callable[..., Any])


def _python_type_to_json(python_type: type) -> str:
    """Convert a Python type annotation to a JSON schema type string.

    Args:
        python_type: Python type annotation to convert.

    Returns:
        JSON schema type string.
    """
    # Handle Optional types (Union with None)
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        # Handle Union types (including Optional)
        import typing

        if origin is typing.Union:
            args = getattr(python_type, "__args__", ())
            # Filter out NoneType for Optional
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return _python_type_to_json(non_none_args[0])
            # For complex unions, default to string
            return "string"
        # Handle List/list
        if origin is list:
            return "array"
        # Handle Dict/dict
        if origin is dict:
            return "object"

    # Basic type mappings
    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }
    return type_map.get(python_type, "string")


def _build_json_schema_from_function(fn: Callable) -> dict[str, Any]:
    """Build a JSON schema for function parameters from type hints.

    Args:
        fn: Function to analyze.

    Returns:
        JSON schema dictionary for the function's parameters.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self' and 'cls' for methods
        if param_name in ("self", "cls"):
            continue

        # Build property definition
        prop: dict[str, Any] = {}

        if param_name in hints:
            prop["type"] = _python_type_to_json(hints[param_name])
        else:
            prop["type"] = "string"  # Default to string if no type hint

        # Add description from docstring if available
        # (Could be enhanced to parse docstring for param descriptions)

        properties[param_name] = prop

        # Parameters without defaults are required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


@dataclass
class Tool:
    """A tool with both schema (for LLM) and implementation (for execution).

    Single source of truth - no separate schema and executor definitions.
    Tools can be registered globally for cross-process serialization.

    Attributes:
        name: The tool's name (used in tool calls).
        description: Human-readable description for the LLM.
        execute: The async or sync function that executes the tool.
        parameters: JSON schema for the tool's parameters.
        strict: Whether to use OpenAI strict mode for this tool.
        sandbox: Required sandbox capabilities for execution.
    """

    name: str
    description: str
    execute: Callable[..., Awaitable[str] | str]
    parameters: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    sandbox: frozenset[str] = field(default_factory=frozenset)
    session: bool = False

    @property
    def schema(self) -> ToolSchema:
        """Get the schema to send to the LLM.

        Returns:
            ToolSchema with name, description, parameters, and strict flag.
        """
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            strict=self.strict,
        )

    async def __call__(self, **kwargs: Any) -> str:
        """Execute the tool with the given arguments.

        Handles both sync and async execute functions.

        Args:
            **kwargs: Arguments to pass to the tool's execute function.

        Returns:
            String result from the tool execution.
        """
        result = self.execute(**kwargs)
        if asyncio.iscoroutine(result):
            return await result  # type: ignore[return-value]
        return result  # type: ignore[return-value]

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Awaitable[str] | str],
        name: str | None = None,
        description: str | None = None,
        strict: bool = False,
        sandbox: set[str] | frozenset[str] | None = None,
        session: bool = False,
    ) -> Tool:
        """Create a Tool from a function, deriving schema from type hints.

        Args:
            fn: The function to wrap as a tool.
            name: Optional name override (defaults to function name).
            description: Optional description override (defaults to docstring).
            strict: Whether to use OpenAI strict mode.
            sandbox: Required sandbox capabilities for execution.
            session: Whether tool requires persistent shell session.

        Returns:
            A new Tool instance.
        """
        tool_name = name or fn.__name__  # type: ignore[union-attr]
        tool_description = description or fn.__doc__ or ""

        # Clean up docstring - remove excess whitespace
        if tool_description:
            lines = tool_description.strip().split("\n")
            # Take first non-empty paragraph
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("Args:") or stripped.startswith("Returns:"):
                    break
                cleaned_lines.append(stripped)
            tool_description = " ".join(cleaned_lines).strip()

        parameters = _build_json_schema_from_function(fn)

        return cls(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            execute=fn,
            strict=strict,
            sandbox=frozenset(sandbox or ()),
            session=session,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (schema only, no execute function).

        Used for serialization. The execute function cannot be serialized,
        so only the schema is included.

        Returns:
            Dictionary with tool schema data.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "strict": self.strict,
            "sandbox": sorted(self.sandbox),
            "session": self.session,
        }


def tool(
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
    sandbox: set[str] | None = None,
    session: bool = False,
) -> Callable[[F], Tool]:
    """Decorator to create a Tool from a function.

    Can be used with or without arguments:
        @tool
        async def my_tool(query: str) -> str:
            ...

        @tool(name="search", description="Search the web")
        async def web_search(query: str) -> str:
            ...

    Args:
        name: Optional name override for the tool.
        description: Optional description override.
        strict: Whether to use OpenAI strict mode.
        sandbox: Required sandbox capabilities for execution.
        session: Whether tool requires persistent shell session.

    Returns:
        Decorator function that converts a function to a Tool.
    """

    def decorator(fn: F) -> Tool:
        return Tool.from_function(
            fn,
            name=name,
            description=description,
            strict=strict,
            sandbox=sandbox,
            session=session,
        )

    # Handle @tool without parentheses
    if callable(name):
        fn = name
        name = None
        return Tool.from_function(fn)  # type: ignore[arg-type, return-value]

    return decorator

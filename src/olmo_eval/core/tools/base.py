"""Base classes for executable tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import ToolCategory, ToolResult, ToolSchema


@dataclass
class BaseTool(ABC):
    """Abstract base class for executable tools.

    Subclasses must implement the execute method to provide
    tool functionality.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    category: ToolCategory = ToolCategory.CUSTOM

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given arguments.

        Args:
            **kwargs: Tool-specific arguments.

        Returns:
            ToolResult with the execution output.
        """
        ...

    def to_schema(self) -> ToolSchema:
        """Convert to ToolSchema for LLM tool definition.

        Returns:
            ToolSchema representation of this tool.
        """
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters_schema,
        )

    def validate_args(self, kwargs: dict[str, Any]) -> bool:
        """Validate arguments against schema.

        Args:
            kwargs: Arguments to validate.

        Returns:
            True if valid, False otherwise.
        """
        required = self.parameters_schema.get("required", [])
        return all(req in kwargs for req in required)


class ToolRegistry:
    """Registry for managing available tools.

    Provides registration, lookup, and iteration over tools.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: The tool to register.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The tool, or None if not found.
        """
        return self._tools.get(name)

    def __getitem__(self, name: str) -> BaseTool:
        """Get a tool by name, raising KeyError if not found.

        Args:
            name: The tool name.

        Returns:
            The tool.

        Raises:
            KeyError: If tool not found.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The tool name.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._tools

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools)

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def items(self):
        """Iterate over (name, tool) pairs."""
        return self._tools.items()

    def values(self):
        """Iterate over tools."""
        return self._tools.values()

    def names(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            List of tool names.
        """
        return list(self._tools.keys())

    def to_schemas(self) -> list[ToolSchema]:
        """Get schemas for all registered tools.

        Returns:
            List of ToolSchema objects.
        """
        return [tool.to_schema() for tool in self._tools.values()]

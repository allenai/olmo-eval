"""Tool calling types matching OpenAI's ChatCompletionMessageToolCall schema.

This module provides the canonical types for tool calling that are compatible
with LiteLLM and OpenAI's API format.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Literal


class ToolCategory(str, Enum):
    """Categories of tools for classification and filtering."""

    SEARCH = "search"
    CODE_EXECUTION = "code_execution"
    FILE_SYSTEM = "file_system"
    API_CALL = "api_call"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class Function:
    """Function call details within a tool call.

    Matches OpenAI's ChatCompletionMessageToolCallFunction format.
    """

    name: str
    arguments: str  # JSON-encoded string


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call from a language model.

    Matches OpenAI's ChatCompletionMessageToolCall format.
    """

    id: str
    type: Literal["function"] = "function"
    function: Function = field(default_factory=lambda: Function(name="", arguments="{}"))

    @classmethod
    def create(cls, call_id: str, name: str, arguments: dict[str, Any] | str) -> "ToolCall":
        """Create a ToolCall with the given parameters.

        Args:
            call_id: Unique identifier for the tool call.
            name: Name of the function to call.
            arguments: Arguments as dict (will be JSON-encoded) or JSON string.

        Returns:
            A new ToolCall instance.
        """
        import json

        args_str = json.dumps(arguments) if isinstance(arguments, dict) else arguments
        return cls(id=call_id, function=Function(name=name, arguments=args_str))

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from OpenAI API response format.

        Args:
            data: Dictionary from OpenAI's tool_calls response.

        Returns:
            A new ToolCall instance.
        """
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=Function(
                name=func.get("name", ""),
                arguments=func.get("arguments", "{}"),
            ),
        )

    @classmethod
    def from_anthropic(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from Anthropic API response format.

        Anthropic uses a different format with 'tool_use' blocks.

        Args:
            data: Dictionary from Anthropic's tool_use content block.

        Returns:
            A new ToolCall instance.
        """
        import json

        return cls(
            id=data.get("id", ""),
            type="function",
            function=Function(
                name=data.get("name", ""),
                arguments=json.dumps(data.get("input", {})),
            ),
        )

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI API format.

        Returns:
            Dictionary in OpenAI's tool_calls format.
        """
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }

    def get_arguments(self) -> dict[str, Any]:
        """Parse and return arguments as a dictionary.

        Returns:
            Parsed arguments dictionary.

        Raises:
            json.JSONDecodeError: If arguments are not valid JSON.
        """
        import json

        return json.loads(self.function.arguments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the ToolCall.
        """
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolCall":
        """Create from dictionary.

        Args:
            data: Dictionary with ToolCall data.

        Returns:
            A new ToolCall instance.
        """
        func = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function=Function(
                name=func.get("name", ""),
                arguments=func.get("arguments", "{}"),
            ),
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Result from executing a tool call."""

    tool_call_id: str
    content: str
    is_error: bool = False

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI tool message format.

        Returns:
            Dictionary suitable for use as a tool message.
        """
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the ToolResult.
        """
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        """Create from dictionary.

        Args:
            data: Dictionary with ToolResult data.

        Returns:
            A new ToolResult instance.
        """
        return cls(
            tool_call_id=data.get("tool_call_id", ""),
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """Schema describing a tool for the language model.

    This follows OpenAI's function schema format used in tools parameter.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    strict: bool = False

    def to_openai(self) -> dict[str, Any]:
        """Convert to OpenAI tools format.

        Returns:
            Dictionary suitable for use in OpenAI's tools parameter.
        """
        result: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        if self.strict:
            result["function"]["strict"] = True
        return result

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "ToolSchema":
        """Create from OpenAI tools format.

        Args:
            data: Dictionary from OpenAI's tools parameter.

        Returns:
            A new ToolSchema instance.
        """
        func = data.get("function", data)  # Handle both wrapped and unwrapped
        return cls(
            name=func.get("name", ""),
            description=func.get("description", ""),
            parameters=func.get("parameters", {}),
            strict=func.get("strict", False),
        )

    @classmethod
    def from_tool_type(cls, tool_type: "ToolType") -> "ToolSchema":
        """Create a ToolSchema from a ToolType instance.

        Args:
            tool_type: A ToolType instance with schema information.

        Returns:
            A new ToolSchema instance.
        """
        return cls(
            name=tool_type.name,
            description=tool_type.description,
            parameters=tool_type.parameters_schema,
        )


class ToolTypeRegistry:
    """Registry for ToolType classes with decorator-based registration."""

    _registry: dict[str, type["ToolType"]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type["ToolType"]], type["ToolType"]]:
        """Decorator to register a ToolType class.

        Args:
            name: Optional name override. Uses class.name if not provided.

        Returns:
            Decorator function.

        Example:
            @ToolTypeRegistry.register()
            class SearchToolType(ToolType):
                name = "search"
                ...
        """

        def decorator(tool_cls: type["ToolType"]) -> type["ToolType"]:
            registry_name = name or tool_cls.name
            cls._registry[registry_name] = tool_cls
            return tool_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type["ToolType"] | None:
        """Get a tool type class by name."""
        return cls._registry.get(name)

    @classmethod
    def list_names(cls) -> list[str]:
        """List all registered tool type names."""
        return list(cls._registry.keys())

    @classmethod
    def list_by_category(cls, category: ToolCategory) -> list[type["ToolType"]]:
        """Get all tool types in a given category.

        Args:
            category: The category to filter by.

        Returns:
            List of tool type classes matching the category.
        """
        return [t for t in cls._registry.values() if t.category == category]

    @classmethod
    def all(cls) -> dict[str, type["ToolType"]]:
        """Get all registered tool types."""
        return dict(cls._registry)


class ToolType(ABC):
    """Abstract base class for tool type definitions.

    ToolTypes define the expected schema and validation for different
    categories of tools. They are used for categorizing and validating
    tool calls without providing execution logic.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    category: ClassVar[ToolCategory]
    parameters_schema: ClassVar[dict[str, Any]]

    @abstractmethod
    def validate_arguments(self, arguments: dict[str, Any]) -> bool:
        """Validate that arguments match the expected schema.

        Args:
            arguments: The arguments to validate.

        Returns:
            True if arguments are valid, False otherwise.
        """
        ...


@ToolTypeRegistry.register()
class SearchToolType(ToolType):
    """Tool type for search operations."""

    name: ClassVar[str] = "search"
    description: ClassVar[str] = "Search for information"
    category: ClassVar[ToolCategory] = ToolCategory.SEARCH
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
        },
        "required": ["query"],
    }

    def validate_arguments(self, arguments: dict[str, Any]) -> bool:
        """Validate search arguments."""
        return "query" in arguments and isinstance(arguments["query"], str)


@ToolTypeRegistry.register()
class CodeExecutionToolType(ToolType):
    """Tool type for code execution."""

    name: ClassVar[str] = "execute_code"
    description: ClassVar[str] = "Execute code in a sandboxed environment"
    category: ClassVar[ToolCategory] = ToolCategory.CODE_EXECUTION
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "The code to execute"},
            "language": {"type": "string", "description": "Programming language"},
        },
        "required": ["code"],
    }

    def validate_arguments(self, arguments: dict[str, Any]) -> bool:
        """Validate code execution arguments."""
        return "code" in arguments and isinstance(arguments["code"], str)


@ToolTypeRegistry.register()
class APICallToolType(ToolType):
    """Tool type for API calls."""

    name: ClassVar[str] = "api_call"
    description: ClassVar[str] = "Make an HTTP API call"
    category: ClassVar[ToolCategory] = ToolCategory.API_CALL
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to call"},
            "method": {"type": "string", "description": "HTTP method"},
            "body": {"type": "object", "description": "Request body"},
        },
        "required": ["url"],
    }

    def validate_arguments(self, arguments: dict[str, Any]) -> bool:
        """Validate API call arguments."""
        return "url" in arguments and isinstance(arguments["url"], str)


@ToolTypeRegistry.register()
class DatabaseToolType(ToolType):
    """Tool type for database operations."""

    name: ClassVar[str] = "database_query"
    description: ClassVar[str] = "Execute a database query"
    category: ClassVar[ToolCategory] = ToolCategory.DATABASE
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The SQL query"},
            "database": {"type": "string", "description": "Database name"},
        },
        "required": ["query"],
    }

    def validate_arguments(self, arguments: dict[str, Any]) -> bool:
        """Validate database query arguments."""
        return "query" in arguments and isinstance(arguments["query"], str)


@ToolTypeRegistry.register()
class FileSystemToolType(ToolType):
    """Tool type for file system operations."""

    name: ClassVar[str] = "file_operation"
    description: ClassVar[str] = "Perform file system operations"
    category: ClassVar[ToolCategory] = ToolCategory.FILE_SYSTEM
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "description": "Operation type (read/write/list)"},
            "path": {"type": "string", "description": "File or directory path"},
            "content": {"type": "string", "description": "Content for write operations"},
        },
        "required": ["operation", "path"],
    }

    def validate_arguments(self, arguments: dict[str, Any]) -> bool:
        """Validate file system arguments."""
        return (
            "operation" in arguments
            and "path" in arguments
            and isinstance(arguments["operation"], str)
            and isinstance(arguments["path"], str)
        )


# Convenience functions wrapping ToolTypeRegistry for backwards compatibility


def get_tool_type(name: str) -> type[ToolType] | None:
    """Get a tool type class by name.

    Args:
        name: The name of the tool type.

    Returns:
        The tool type class, or None if not found.
    """
    return ToolTypeRegistry.get(name)


def register_tool_type(name: str, tool_type: type[ToolType]) -> None:
    """Register a new tool type.

    Args:
        name: The name to register the tool type under.
        tool_type: The tool type class to register.
    """
    ToolTypeRegistry._registry[name] = tool_type


def list_tool_types() -> list[str]:
    """List all registered tool type names.

    Returns:
        List of registered tool type names.
    """
    return ToolTypeRegistry.list_names()


def get_tool_types_by_category(category: ToolCategory) -> list[type[ToolType]]:
    """Get all tool types in a given category.

    Args:
        category: The category to filter by.

    Returns:
        List of tool type classes matching the category.
    """
    return ToolTypeRegistry.list_by_category(category)


# =============================================================================
# MCP Tool Schemas - Reusable tool definitions for agent tasks
# =============================================================================

SEMANTIC_SCHOLAR_SEARCH = ToolSchema(
    name="semantic_scholar_snippet_search",
    description="Search Semantic Scholar for academic papers and snippets matching a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for academic papers and snippets.",
            },
        },
        "required": ["query"],
    },
)

SERPER_WEB_SEARCH = ToolSchema(
    name="serper_google_webpage_search",
    description="Search the web for information using Google via Serper.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant web pages.",
            },
        },
        "required": ["query"],
    },
)

SERPER_FETCH_WEBPAGE = ToolSchema(
    name="serper_fetch_webpage_content",
    description="Fetch and extract content from a webpage URL.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to fetch.",
            },
        },
        "required": ["url"],
    },
)

# Pre-built tool collections for common use cases
SEARCH_TOOLS = (SEMANTIC_SCHOLAR_SEARCH, SERPER_WEB_SEARCH, SERPER_FETCH_WEBPAGE)
SEARCH_TOOL_NAMES = tuple(t.name for t in SEARCH_TOOLS)

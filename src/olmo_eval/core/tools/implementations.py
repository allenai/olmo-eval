"""Tool implementations for agent evaluation.

This module provides concrete tool implementations including search,
code execution, and mock tools for testing.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..types import ToolCategory, ToolResult
from .base import BaseTool


@dataclass
class SearchTool(BaseTool):
    """Tool for performing searches.

    Uses an injected search function for flexibility in testing
    and different search backends.
    """

    name: str = "search"
    description: str = "Search for information"
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        }
    )
    category: ToolCategory = ToolCategory.SEARCH
    search_fn: Callable[[str], list[str]] | None = None
    max_results: int = 10

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute search with given query.

        Args:
            **kwargs: Must include 'query' string.

        Returns:
            ToolResult with search results or error.
        """
        call_id = kwargs.pop("_tool_call_id", "search_call")
        query = kwargs.get("query", "")

        if not query:
            return ToolResult(
                tool_call_id=call_id,
                content="Error: No query provided",
                is_error=True,
            )

        if self.search_fn is None:
            return ToolResult(
                tool_call_id=call_id,
                content="Error: No search function configured",
                is_error=True,
            )

        try:
            results = self.search_fn(query)
            limited = results[: self.max_results]
            return ToolResult(
                tool_call_id=call_id,
                content="\n".join(limited) if limited else "No results found",
                is_error=False,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call_id,
                content=f"Error: {e!s}",
                is_error=True,
            )


@dataclass
class CodeExecutionTool(BaseTool):
    """Tool for executing code.

    Uses an injected executor function for sandboxed execution.
    """

    name: str = "execute_code"
    description: str = "Execute code in a sandboxed environment"
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The code to execute"},
                "language": {
                    "type": "string",
                    "description": "Programming language",
                    "default": "python",
                },
            },
            "required": ["code"],
        }
    )
    category: ToolCategory = ToolCategory.CODE_EXECUTION
    executor: Callable[[str, str], tuple[bool, str]] | None = None
    language: str = "python"
    timeout: float = 5.0

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute code.

        Args:
            **kwargs: Must include 'code' string, optionally 'language'.

        Returns:
            ToolResult with execution output or error.
        """
        call_id = kwargs.pop("_tool_call_id", "code_call")
        code = kwargs.get("code", "")
        language = kwargs.get("language", self.language)

        if not code:
            return ToolResult(
                tool_call_id=call_id,
                content="Error: No code provided",
                is_error=True,
            )

        if self.executor is None:
            return ToolResult(
                tool_call_id=call_id,
                content="Error: No executor configured",
                is_error=True,
            )

        try:
            success, output = self.executor(code, language)
            return ToolResult(
                tool_call_id=call_id,
                content=output,
                is_error=not success,
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=call_id,
                content=f"Error: {e!s}",
                is_error=True,
            )


@dataclass
class MockTool(BaseTool):
    """Mock tool for testing purposes.

    Returns predefined responses for given inputs.
    """

    name: str = "mock_tool"
    description: str = "A mock tool for testing"
    parameters_schema: dict[str, Any] = field(default_factory=dict)
    category: ToolCategory = ToolCategory.CUSTOM
    responses: dict[str, str] = field(default_factory=dict)
    default_response: str = "Mock response"

    def execute(self, **kwargs: Any) -> ToolResult:
        """Return mock response.

        Args:
            **kwargs: Arguments (used to look up response).

        Returns:
            ToolResult with mock response.
        """
        call_id = kwargs.pop("_tool_call_id", "mock_call")

        # Try to match a response based on arguments
        import json

        key = json.dumps(kwargs, sort_keys=True)
        content = self.responses.get(key, self.default_response)

        return ToolResult(
            tool_call_id=call_id,
            content=content,
            is_error=False,
        )

    def add_response(self, response: str, **kwargs: Any) -> None:
        """Add a response for specific arguments.

        Args:
            response: The response to return.
            **kwargs: The arguments to match.
        """
        import json

        key = json.dumps(kwargs, sort_keys=True)
        self.responses[key] = response


def create_mock_tool(
    name: str,
    responses: dict[str, str] | None = None,
    default_response: str = "Mock response",
) -> MockTool:
    """Create a mock tool with given responses.

    Args:
        name: Tool name.
        responses: Optional dict mapping serialized args to responses.
        default_response: Default response when no match found.

    Returns:
        Configured MockTool instance.
    """
    return MockTool(
        name=name,
        description=f"Mock {name} tool",
        responses=responses or {},
        default_response=default_response,
    )

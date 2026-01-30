"""Tool implementations for agent evaluation.

This module provides concrete tool implementations including search,
code execution, calculator, and mock tools for testing.
"""

import ast
import operator
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


# Safe operators for calculator
_SAFE_OPERATORS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Safely evaluate an AST node for math expressions.

    Args:
        node: AST node to evaluate.

    Returns:
        Numeric result.

    Raises:
        ValueError: If the expression contains unsafe operations.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    else:
        raise ValueError(f"Unsupported expression type: {type(node)}")


@dataclass
class CalculatorTool(BaseTool):
    """Tool for safe mathematical expression evaluation.

    Only supports basic arithmetic operations for safety.
    """

    name: str = "calculator"
    description: str = "Evaluate mathematical expressions safely"
    parameters_schema: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2 * 3')",
                },
            },
            "required": ["expression"],
        }
    )
    category: ToolCategory = ToolCategory.CALCULATOR

    def execute(self, **kwargs: Any) -> ToolResult:
        """Evaluate a mathematical expression.

        Args:
            **kwargs: Must include 'expression' string.

        Returns:
            ToolResult with the computed result or error.
        """
        call_id = kwargs.pop("_tool_call_id", "calc_call")
        expression = kwargs.get("expression", "")

        if not expression:
            return ToolResult(
                tool_call_id=call_id,
                content="Error: No expression provided",
                is_error=True,
            )

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return ToolResult(
                tool_call_id=call_id,
                content=str(result),
                is_error=False,
            )
        except (SyntaxError, ValueError) as e:
            return ToolResult(
                tool_call_id=call_id,
                content=f"Error: {e!s}",
                is_error=True,
            )
        except ZeroDivisionError:
            return ToolResult(
                tool_call_id=call_id,
                content="Error: Division by zero",
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

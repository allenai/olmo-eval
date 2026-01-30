"""Tests for olmo_eval.core.tools module."""

import pytest

from olmo_eval.core.tools import (
    CalculatorTool,
    CodeExecutionTool,
    MockTool,
    SearchTool,
    ToolRegistry,
    create_mock_tool,
)
from olmo_eval.core.types import ToolCategory, ToolSchema


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        """Test registering and getting tools."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry.get("calculator") is tool

    def test_get_nonexistent(self):
        """Test getting nonexistent tool."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_getitem(self):
        """Test dictionary-style access."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert registry["calculator"] is tool

    def test_getitem_raises(self):
        """Test KeyError on missing tool."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            _ = registry["nonexistent"]

    def test_contains(self):
        """Test 'in' operator."""
        registry = ToolRegistry()
        tool = CalculatorTool()
        registry.register(tool)
        assert "calculator" in registry
        assert "nonexistent" not in registry

    def test_iteration(self):
        """Test iterating over tool names."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(SearchTool(search_fn=lambda q: []))
        names = list(registry)
        assert "calculator" in names
        assert "search" in names

    def test_len(self):
        """Test length."""
        registry = ToolRegistry()
        assert len(registry) == 0
        registry.register(CalculatorTool())
        assert len(registry) == 1

    def test_names(self):
        """Test getting list of names."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert registry.names() == ["calculator"]

    def test_to_schemas(self):
        """Test converting all tools to schemas."""
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        schemas = registry.to_schemas()
        assert len(schemas) == 1
        assert isinstance(schemas[0], ToolSchema)
        assert schemas[0].name == "calculator"


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    def test_basic_addition(self):
        """Test basic addition."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 + 2")
        assert result.content == "4"
        assert result.is_error is False

    def test_complex_expression(self):
        """Test complex expression."""
        tool = CalculatorTool()
        result = tool.execute(expression="(10 + 5) * 2 - 3")
        assert result.content == "27"

    def test_division(self):
        """Test division."""
        tool = CalculatorTool()
        result = tool.execute(expression="10 / 4")
        assert result.content == "2.5"

    def test_floor_division(self):
        """Test floor division."""
        tool = CalculatorTool()
        result = tool.execute(expression="10 // 3")
        assert result.content == "3"

    def test_power(self):
        """Test exponentiation."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 ** 10")
        assert result.content == "1024"

    def test_negative_number(self):
        """Test negative numbers."""
        tool = CalculatorTool()
        result = tool.execute(expression="-5 + 10")
        assert result.content == "5"

    def test_division_by_zero(self):
        """Test division by zero error."""
        tool = CalculatorTool()
        result = tool.execute(expression="1 / 0")
        assert result.is_error is True
        assert "Division by zero" in result.content

    def test_invalid_expression(self):
        """Test invalid expression."""
        tool = CalculatorTool()
        result = tool.execute(expression="2 +* 2")
        assert result.is_error is True

    def test_no_expression(self):
        """Test with no expression."""
        tool = CalculatorTool()
        result = tool.execute()
        assert result.is_error is True

    def test_unsafe_expression_rejected(self):
        """Test that unsafe expressions are rejected."""
        tool = CalculatorTool()
        result = tool.execute(expression="__import__('os').system('ls')")
        assert result.is_error is True

    def test_to_schema(self):
        """Test converting to schema."""
        tool = CalculatorTool()
        schema = tool.to_schema()
        assert schema.name == "calculator"
        assert "expression" in schema.parameters["properties"]


class TestSearchTool:
    """Tests for SearchTool."""

    def test_with_search_function(self):
        """Test with injected search function."""

        def mock_search(query: str) -> list[str]:
            return [f"Result 1 for {query}", f"Result 2 for {query}"]

        tool = SearchTool(search_fn=mock_search)
        result = tool.execute(query="test query")
        assert "Result 1 for test query" in result.content
        assert result.is_error is False

    def test_no_search_function(self):
        """Test error when no search function configured."""
        tool = SearchTool()
        result = tool.execute(query="test")
        assert result.is_error is True
        assert "No search function" in result.content

    def test_no_query(self):
        """Test error when no query provided."""
        tool = SearchTool(search_fn=lambda q: [])
        result = tool.execute()
        assert result.is_error is True
        assert "No query" in result.content

    def test_max_results(self):
        """Test max_results limiting."""

        def many_results(query: str) -> list[str]:
            return [f"Result {i}" for i in range(100)]

        tool = SearchTool(search_fn=many_results, max_results=5)
        result = tool.execute(query="test")
        lines = result.content.split("\n")
        assert len(lines) == 5

    def test_no_results(self):
        """Test when no results found."""
        tool = SearchTool(search_fn=lambda q: [])
        result = tool.execute(query="test")
        assert "No results found" in result.content

    def test_search_error(self):
        """Test handling search errors."""

        def failing_search(query: str) -> list[str]:
            raise ValueError("Search failed")

        tool = SearchTool(search_fn=failing_search)
        result = tool.execute(query="test")
        assert result.is_error is True
        assert "Search failed" in result.content


class TestCodeExecutionTool:
    """Tests for CodeExecutionTool."""

    def test_with_executor(self):
        """Test with injected executor."""

        def mock_executor(code: str, lang: str) -> tuple[bool, str]:
            return True, "Execution successful"

        tool = CodeExecutionTool(executor=mock_executor)
        result = tool.execute(code="print('hello')")
        assert result.content == "Execution successful"
        assert result.is_error is False

    def test_execution_failure(self):
        """Test failed execution."""

        def failing_executor(code: str, lang: str) -> tuple[bool, str]:
            return False, "SyntaxError"

        tool = CodeExecutionTool(executor=failing_executor)
        result = tool.execute(code="invalid code")
        assert result.is_error is True

    def test_no_executor(self):
        """Test error when no executor configured."""
        tool = CodeExecutionTool()
        result = tool.execute(code="print('hello')")
        assert result.is_error is True
        assert "No executor" in result.content

    def test_no_code(self):
        """Test error when no code provided."""
        tool = CodeExecutionTool(executor=lambda code, lang: (True, ""))
        result = tool.execute()
        assert result.is_error is True
        assert "No code" in result.content


class TestMockTool:
    """Tests for MockTool."""

    def test_default_response(self):
        """Test default response."""
        tool = MockTool(default_response="Default")
        result = tool.execute(any_arg="value")
        assert result.content == "Default"
        assert result.is_error is False

    def test_specific_response(self):
        """Test response for specific arguments."""
        tool = MockTool()
        tool.add_response("Specific result", query="test")
        result = tool.execute(query="test")
        assert result.content == "Specific result"

    def test_create_mock_tool(self):
        """Test create_mock_tool factory."""
        tool = create_mock_tool("test_tool", default_response="Mock result")
        assert tool.name == "test_tool"
        result = tool.execute()
        assert result.content == "Mock result"


class TestBaseTool:
    """Tests for BaseTool interface."""

    def test_validate_args(self):
        """Test argument validation."""
        tool = CalculatorTool()
        assert tool.validate_args({"expression": "2+2"})
        assert not tool.validate_args({})

    def test_category(self):
        """Test tool category."""
        assert CalculatorTool().category == ToolCategory.CALCULATOR
        assert SearchTool().category == ToolCategory.SEARCH
        assert CodeExecutionTool().category == ToolCategory.CODE_EXECUTION

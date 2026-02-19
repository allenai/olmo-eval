"""Tests for Tool class and decorators."""

from __future__ import annotations

import pytest

from olmo_eval.harness import (
    TOOL_REGISTRY,
    clear_registry,
    get_tool,
    list_tools,
    register_tool,
    registered_tool,
)
from olmo_eval.harness.tools import Tool, tool


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the tool registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestTool:
    """Tests for the Tool class."""

    def test_tool_creation(self):
        """Test creating a Tool directly."""

        async def my_func(query: str) -> str:
            return f"Result: {query}"

        t = Tool(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            execute=my_func,
        )

        assert t.name == "test_tool"
        assert t.description == "A test tool"
        assert "query" in t.parameters["properties"]

    def test_tool_schema_property(self):
        """Test that schema property returns a valid ToolSchema."""

        async def my_func(query: str) -> str:
            return query

        t = Tool(
            name="schema_test",
            description="Test description",
            parameters={"type": "object", "properties": {}},
            execute=my_func,
        )

        schema = t.schema
        assert schema.name == "schema_test"
        assert schema.description == "Test description"

    def test_tool_from_function(self):
        """Test creating a Tool from a function."""

        async def search(query: str, limit: int = 10) -> str:
            """Search for something."""
            return f"Searching: {query}"

        t = Tool.from_function(search)

        assert t.name == "search"
        assert "Search for something" in t.description
        assert "query" in t.parameters["properties"]
        assert t.parameters["properties"]["query"]["type"] == "string"
        assert t.parameters["properties"]["limit"]["type"] == "integer"
        assert "query" in t.parameters["required"]
        assert "limit" not in t.parameters["required"]

    def test_tool_from_function_with_overrides(self):
        """Test creating a Tool with name/description overrides."""

        def my_func(x: int) -> str:
            """Original docstring."""
            return str(x)

        t = Tool.from_function(
            my_func,
            name="custom_name",
            description="Custom description",
        )

        assert t.name == "custom_name"
        assert t.description == "Custom description"

    @pytest.mark.anyio
    async def test_tool_call(self):
        """Test calling a Tool."""

        async def add(a: int, b: int) -> str:
            return str(a + b)

        t = Tool(
            name="add",
            description="Add numbers",
            parameters={},
            execute=add,
        )

        result = await t(a=5, b=3)
        assert result == "8"

    @pytest.mark.anyio
    async def test_tool_call_sync_function(self):
        """Test calling a Tool with a sync execute function."""

        def multiply(a: int, b: int) -> str:
            return str(a * b)

        t = Tool(
            name="multiply",
            description="Multiply numbers",
            parameters={},
            execute=multiply,
        )

        result = await t(a=4, b=7)
        assert result == "28"

    def test_tool_to_dict(self):
        """Test Tool serialization."""

        async def my_func(query: str) -> str:
            return query

        t = Tool(
            name="serialize_test",
            description="Test serialization",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            execute=my_func,
            strict=True,
        )

        d = t.to_dict()
        assert d["name"] == "serialize_test"
        assert d["description"] == "Test serialization"
        assert d["strict"] is True
        assert "execute" not in d  # Functions can't be serialized


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_tool_decorator_with_args(self):
        """Test @tool decorator with arguments."""

        @tool(name="decorated", description="A decorated tool")
        async def my_tool(query: str) -> str:
            return query

        assert isinstance(my_tool, Tool)
        assert my_tool.name == "decorated"
        assert my_tool.description == "A decorated tool"

    def test_tool_decorator_without_args(self):
        """Test @tool decorator without parentheses."""

        @tool
        async def simple_tool(x: str) -> str:
            """Simple tool docstring."""
            return x

        assert isinstance(simple_tool, Tool)
        assert simple_tool.name == "simple_tool"
        assert "Simple tool docstring" in simple_tool.description


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_register_tool(self):
        """Test registering a tool."""

        @tool(name="reg_test")
        async def reg_func(x: str) -> str:
            return x

        register_tool(reg_func)

        assert "reg_test" in TOOL_REGISTRY
        assert get_tool("reg_test") is reg_func

    def test_get_tool_unknown(self):
        """Test getting an unknown tool raises error."""
        with pytest.raises(ValueError, match="Unknown tool"):
            get_tool("nonexistent_tool")

    def test_list_tools(self):
        """Test listing registered tools."""

        @tool(name="tool_a")
        async def func_a(x: str) -> str:
            return x

        @tool(name="tool_b")
        async def func_b(x: str) -> str:
            return x

        register_tool(func_a)
        register_tool(func_b)

        tools = list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools
        assert tools == sorted(tools)  # Should be sorted

    def test_registered_tool_decorator(self):
        """Test @registered_tool decorator."""

        @registered_tool(name="auto_registered", description="Auto-registered tool")
        async def auto_func(query: str) -> str:
            return query

        assert isinstance(auto_func, Tool)
        assert "auto_registered" in TOOL_REGISTRY
        assert get_tool("auto_registered") is auto_func

    def test_idempotent_registration(self):
        """Test that re-registering the same tool is idempotent."""

        @tool(name="idem_test")
        async def idem_func(x: str) -> str:
            return x

        register_tool(idem_func)
        register_tool(idem_func)  # Should not raise

        assert get_tool("idem_test") is idem_func


class TestToolTypeInference:
    """Tests for type inference in Tool.from_function."""

    def test_basic_types(self):
        """Test inference of basic Python types."""

        def func(s: str, i: int, f: float, b: bool) -> str:
            return ""

        t = Tool.from_function(func)

        assert t.parameters["properties"]["s"]["type"] == "string"
        assert t.parameters["properties"]["i"]["type"] == "integer"
        assert t.parameters["properties"]["f"]["type"] == "number"
        assert t.parameters["properties"]["b"]["type"] == "boolean"

    def test_optional_parameters(self):
        """Test that optional parameters are not required."""

        def func(required: str, optional: int = 10) -> str:
            return ""

        t = Tool.from_function(func)

        assert "required" in t.parameters["required"]
        assert "optional" not in t.parameters["required"]

    def test_no_type_hints(self):
        """Test function with no type hints defaults to string."""

        def func(x, y):
            return ""

        t = Tool.from_function(func)

        # Without type hints, defaults to string
        assert t.parameters["properties"]["x"]["type"] == "string"
        assert t.parameters["properties"]["y"]["type"] == "string"

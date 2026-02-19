"""Tests for olmo_eval.core.tool_types module."""

import json

from olmo_eval.common.types import (
    APICallToolType,
    CodeExecutionToolType,
    DatabaseToolType,
    FileSystemToolType,
    Function,
    SearchToolType,
    ToolCall,
    ToolCategory,
    ToolResult,
    ToolSchema,
    ToolTypeRegistry,
    get_tool_type,
    get_tool_types_by_category,
    list_tool_types,
    register_tool_type,
)


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_category_values(self):
        """Test category string values."""
        assert ToolCategory.SEARCH.value == "search"
        assert ToolCategory.CODE_EXECUTION.value == "code_execution"
        assert ToolCategory.FILE_SYSTEM.value == "file_system"
        assert ToolCategory.API_CALL.value == "api_call"
        assert ToolCategory.DATABASE.value == "database"
        assert ToolCategory.CUSTOM.value == "custom"


class TestFunction:
    """Tests for Function dataclass."""

    def test_create_function(self):
        """Test creating a Function."""
        func = Function(name="test", arguments='{"key": "value"}')
        assert func.name == "test"
        assert func.arguments == '{"key": "value"}'


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_create_factory(self):
        """Test ToolCall.create factory method."""
        call = ToolCall.create("123", "get_weather", {"location": "NYC"})
        assert call.id == "123"
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"location": "NYC"}

    def test_create_with_string_arguments(self):
        """Test ToolCall.create with string arguments."""
        call = ToolCall.create("123", "test", '{"key": "value"}')
        assert call.function.arguments == '{"key": "value"}'

    def test_from_openai(self):
        """Test creating ToolCall from OpenAI format."""
        data = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
        }
        call = ToolCall.from_openai(data)
        assert call.id == "call_123"
        assert call.type == "function"
        assert call.function.name == "get_weather"
        assert call.function.arguments == '{"location": "NYC"}'

    def test_from_anthropic(self):
        """Test creating ToolCall from Anthropic format."""
        data = {
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"location": "NYC"},
        }
        call = ToolCall.from_anthropic(data)
        assert call.id == "toolu_123"
        assert call.type == "function"
        assert call.function.name == "get_weather"
        assert json.loads(call.function.arguments) == {"location": "NYC"}

    def test_to_openai(self):
        """Test converting ToolCall to OpenAI format."""
        call = ToolCall.create("123", "test", {"key": "value"})
        result = call.to_openai()
        assert result["id"] == "123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "test"
        assert json.loads(result["function"]["arguments"]) == {"key": "value"}

    def test_get_arguments(self):
        """Test parsing arguments."""
        call = ToolCall.create("123", "test", {"num": 42, "text": "hello"})
        args = call.get_arguments()
        assert args == {"num": 42, "text": "hello"}


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_to_openai(self):
        """Test converting to OpenAI tool message format."""
        result = ToolResult(tool_call_id="123", content="Result")
        msg = result.to_openai()
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "123"
        assert msg["content"] == "Result"


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_to_openai(self):
        """Test converting to OpenAI tools format."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters={"type": "object"},
        )
        result = schema.to_openai()
        assert result["type"] == "function"
        assert result["function"]["name"] == "test"
        assert result["function"]["description"] == "Test tool"

    def test_to_openai_strict(self):
        """Test strict mode in OpenAI format."""
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters={},
            strict=True,
        )
        result = schema.to_openai()
        assert result["function"]["strict"] is True

    def test_from_openai(self):
        """Test creating from OpenAI format."""
        data = {
            "type": "function",
            "function": {
                "name": "test",
                "description": "Test tool",
                "parameters": {"type": "object"},
            },
        }
        schema = ToolSchema.from_openai(data)
        assert schema.name == "test"
        assert schema.description == "Test tool"

    def test_from_tool_type(self):
        """Test creating from ToolType."""
        schema = ToolSchema.from_tool_type(SearchToolType())
        assert schema.name == "search"
        assert "query" in schema.parameters["properties"]


class TestToolCallSerialization:
    """Tests for ToolCall serialization."""

    def test_to_dict(self):
        """Test converting ToolCall to dict."""
        call = ToolCall.create("123", "get_weather", {"location": "NYC"})
        result = call.to_dict()
        assert result["id"] == "123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert json.loads(result["function"]["arguments"]) == {"location": "NYC"}

    def test_from_dict(self):
        """Test creating ToolCall from dict."""
        data = {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}',
            },
        }
        call = ToolCall.from_dict(data)
        assert call.id == "call_456"
        assert call.type == "function"
        assert call.function.name == "search"
        assert call.function.arguments == '{"query": "test"}'

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = ToolCall.create("789", "execute_code", {"code": "print(2+2)"})
        restored = ToolCall.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.function.name == original.function.name
        assert restored.function.arguments == original.function.arguments


class TestToolResultSerialization:
    """Tests for ToolResult serialization."""

    def test_to_dict(self):
        """Test converting ToolResult to dict."""
        result = ToolResult(tool_call_id="123", content="Result text", is_error=False)
        data = result.to_dict()
        assert data["tool_call_id"] == "123"
        assert data["content"] == "Result text"
        assert data["is_error"] is False

    def test_to_dict_error(self):
        """Test converting error ToolResult to dict."""
        result = ToolResult(tool_call_id="456", content="Error message", is_error=True)
        data = result.to_dict()
        assert data["is_error"] is True

    def test_from_dict(self):
        """Test creating ToolResult from dict."""
        data = {
            "tool_call_id": "789",
            "content": "Search results",
            "is_error": False,
        }
        result = ToolResult.from_dict(data)
        assert result.tool_call_id == "789"
        assert result.content == "Search results"
        assert result.is_error is False

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = ToolResult(tool_call_id="test", content="Content here", is_error=True)
        restored = ToolResult.from_dict(original.to_dict())
        assert restored.tool_call_id == original.tool_call_id
        assert restored.content == original.content
        assert restored.is_error == original.is_error


class TestToolTypes:
    """Tests for ToolType implementations."""

    def test_search_tool_type(self):
        """Test SearchToolType."""
        tool = SearchToolType()
        assert tool.name == "search"
        assert tool.category == ToolCategory.SEARCH
        assert tool.validate_arguments({"query": "test"})
        assert not tool.validate_arguments({})
        assert not tool.validate_arguments({"query": 123})

    def test_code_execution_tool_type(self):
        """Test CodeExecutionToolType."""
        tool = CodeExecutionToolType()
        assert tool.name == "execute_code"
        assert tool.category == ToolCategory.CODE_EXECUTION
        assert tool.validate_arguments({"code": "print('hello')"})
        assert not tool.validate_arguments({})

    def test_api_call_tool_type(self):
        """Test APICallToolType."""
        tool = APICallToolType()
        assert tool.name == "api_call"
        assert tool.category == ToolCategory.API_CALL
        assert tool.validate_arguments({"url": "https://api.example.com"})
        assert not tool.validate_arguments({})

    def test_database_tool_type(self):
        """Test DatabaseToolType."""
        tool = DatabaseToolType()
        assert tool.name == "database_query"
        assert tool.category == ToolCategory.DATABASE
        assert tool.validate_arguments({"query": "SELECT * FROM users"})
        assert not tool.validate_arguments({})

    def test_file_system_tool_type(self):
        """Test FileSystemToolType."""
        tool = FileSystemToolType()
        assert tool.name == "file_operation"
        assert tool.category == ToolCategory.FILE_SYSTEM
        assert tool.validate_arguments({"operation": "read", "path": "/tmp/test"})
        assert not tool.validate_arguments({"operation": "read"})


class TestRegistry:
    """Tests for tool type registry functions."""

    def test_get_tool_type(self):
        """Test getting registered tool types."""
        assert get_tool_type("search") is SearchToolType
        assert get_tool_type("execute_code") is CodeExecutionToolType
        assert get_tool_type("nonexistent") is None

    def test_list_tool_types(self):
        """Test listing all tool types."""
        types = list_tool_types()
        assert "search" in types
        assert "execute_code" in types
        assert "api_call" in types

    def test_register_tool_type(self):
        """Test registering a custom tool type."""

        class CustomToolType(SearchToolType):
            name = "custom"

        register_tool_type("custom", CustomToolType)
        assert get_tool_type("custom") is CustomToolType

    def test_get_tool_types_by_category(self):
        """Test getting tool types by category."""
        search_types = get_tool_types_by_category(ToolCategory.SEARCH)
        assert SearchToolType in search_types

        code_types = get_tool_types_by_category(ToolCategory.CODE_EXECUTION)
        assert CodeExecutionToolType in code_types

    def test_registry_decorator(self):
        """Test the decorator-based registration."""
        # All built-in types should be auto-registered
        assert ToolTypeRegistry.get("search") is SearchToolType
        assert ToolTypeRegistry.get("execute_code") is CodeExecutionToolType
        assert ToolTypeRegistry.get("api_call") is APICallToolType
        assert ToolTypeRegistry.get("database_query") is DatabaseToolType
        assert ToolTypeRegistry.get("file_operation") is FileSystemToolType

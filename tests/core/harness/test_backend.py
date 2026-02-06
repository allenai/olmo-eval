"""Tests for Backend implementations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from olmo_eval.core.harness import clear_registry, register_tool
from olmo_eval.core.harness.backend import (
    BACKEND_REGISTRY,
    InternalBackend,
    get_backend,
    list_backends,
    register_backend,
)
from olmo_eval.core.harness.config import HarnessConfig
from olmo_eval.core.harness.harness import Harness
from olmo_eval.core.harness.tool import tool
from olmo_eval.core.types import LMOutput, LMRequest, RequestType
from olmo_eval.core.types.tools import ToolCall


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the tool registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def mock_provider():
    """Create a mock inference provider."""
    provider = MagicMock()
    provider.model_name = "test-model"
    return provider


@pytest.fixture
def simple_tool():
    """Create and register a simple tool."""

    @tool(name="echo", description="Echo the input")
    async def echo(message: str) -> str:
        return f"Echo: {message}"

    register_tool(echo)
    return echo


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_internal_backend(self):
        """Test getting the internal backend."""
        backend = get_backend("internal")
        assert isinstance(backend, InternalBackend)

    def test_get_unknown_backend(self):
        """Test getting an unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_register_custom_backend(self):
        """Test registering a custom backend using the decorator."""
        from olmo_eval.core.harness.backend import Backend

        @register_backend("custom")
        class CustomBackend(Backend):
            async def run(self, harness, request, sampling_params=None):
                pass

        assert "custom" in BACKEND_REGISTRY
        assert "custom" in list_backends()

        # Clean up
        del BACKEND_REGISTRY["custom"]


class TestInternalBackend:
    """Tests for InternalBackend."""

    @pytest.mark.asyncio
    async def test_run_no_tool_calls(self, mock_provider):
        """Test run completes when response has no tool calls."""
        mock_provider.generate.return_value = [[LMOutput(text="Done", tool_calls=None)]]

        config = HarnessConfig(name="test", backend="internal")
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        backend = InternalBackend()
        result = await backend.run(harness, request)

        assert result.final_text == "Done"
        assert result.num_turns == 1
        assert result.max_turns_reached is False
        assert result.trajectory.total_tool_calls == 0

    @pytest.mark.asyncio
    async def test_run_with_tool_call(self, mock_provider, simple_tool):
        """Test run executes tool and continues."""
        # First call: model requests tool
        tool_call = ToolCall.create("call_1", "echo", {"message": "test"})
        first_output = LMOutput(text="", tool_calls=[tool_call])

        # Second call: model returns final answer
        second_output = LMOutput(text="Final answer", tool_calls=None)

        mock_provider.generate.side_effect = [[[first_output]], [[second_output]]]

        config = HarnessConfig(
            name="tool_test",
            tool_names=("echo",),
            backend="internal",
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Test"},),
        )

        backend = InternalBackend()
        result = await backend.run(harness, request)

        assert result.final_text == "Final answer"
        assert result.trajectory.total_tool_calls == 1
        assert result.max_turns_reached is False

        # Check tool was executed
        tool_results = result.trajectory.tool_result_sequence
        assert len(tool_results) == 1
        assert "Echo: test" in tool_results[0].content

    @pytest.mark.asyncio
    async def test_run_max_turns_reached(self, mock_provider, simple_tool):
        """Test run stops when max_turns is reached."""
        # Model always requests a tool call
        tool_call = ToolCall.create("call_1", "echo", {"message": "loop"})
        output = LMOutput(text="", tool_calls=[tool_call])
        mock_provider.generate.return_value = [[output]]

        config = HarnessConfig(
            name="max_turns_test",
            tool_names=("echo",),
            max_turns=3,
            backend="internal",
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Test"},),
        )

        backend = InternalBackend()
        result = await backend.run(harness, request)

        assert result.max_turns_reached is True
        assert result.num_turns <= 3 * 2  # assistant + tool turns

    @pytest.mark.asyncio
    async def test_run_unknown_tool(self, mock_provider):
        """Test run handles unknown tool call gracefully."""
        tool_call = ToolCall.create("call_1", "unknown_tool", {})
        first_output = LMOutput(text="", tool_calls=[tool_call])
        second_output = LMOutput(text="Error handled", tool_calls=None)

        mock_provider.generate.side_effect = [[[first_output]], [[second_output]]]

        config = HarnessConfig(
            name="unknown_tool_test",
            backend="internal",
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Test"},),
        )

        backend = InternalBackend()
        result = await backend.run(harness, request)

        # Should complete without exception
        tool_results = result.trajectory.tool_result_sequence
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "Unknown tool" in tool_results[0].content

    @pytest.mark.asyncio
    async def test_run_tool_error(self, mock_provider):
        """Test run handles tool execution errors."""

        @tool(name="failing_tool")
        async def failing_tool(x: str) -> str:
            raise ValueError("Tool failed!")

        register_tool(failing_tool)

        tool_call = ToolCall.create("call_1", "failing_tool", {"x": "test"})
        first_output = LMOutput(text="", tool_calls=[tool_call])
        second_output = LMOutput(text="Recovered", tool_calls=None)

        mock_provider.generate.side_effect = [[[first_output]], [[second_output]]]

        config = HarnessConfig(
            name="error_test",
            tool_names=("failing_tool",),
            backend="internal",
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Test"},),
        )

        backend = InternalBackend()
        result = await backend.run(harness, request)

        tool_results = result.trajectory.tool_result_sequence
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "Tool error" in tool_results[0].content

    @pytest.mark.asyncio
    async def test_run_invalid_json_arguments(self, mock_provider, simple_tool):
        """Test run handles invalid JSON in tool arguments."""
        from olmo_eval.core.types.tools import Function

        tool_call = ToolCall(
            id="call_1",
            function=Function(name="echo", arguments="not valid json"),
        )
        first_output = LMOutput(text="", tool_calls=[tool_call])
        second_output = LMOutput(text="Recovered", tool_calls=None)

        mock_provider.generate.side_effect = [[[first_output]], [[second_output]]]

        config = HarnessConfig(
            name="json_error_test",
            tool_names=("echo",),
            backend="internal",
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Test"},),
        )

        backend = InternalBackend()
        result = await backend.run(harness, request)

        tool_results = result.trajectory.tool_result_sequence
        assert len(tool_results) == 1
        assert tool_results[0].is_error is True
        assert "Invalid JSON" in tool_results[0].content

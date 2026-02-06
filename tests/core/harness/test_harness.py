"""Tests for Harness class."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from olmo_eval.core.harness import clear_registry, register_tool
from olmo_eval.core.harness.config import HarnessConfig
from olmo_eval.core.harness.harness import Harness, create_harness
from olmo_eval.core.harness.tools import tool
from olmo_eval.core.types import LMOutput, LMRequest, RequestType


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
    provider.generate.return_value = [[LMOutput(text="Generated text")]]
    provider.logprobs.return_value = [[LMOutput(text="", logprobs=[])]]
    return provider


@pytest.fixture
def sample_tool():
    """Create and register a sample tool."""

    @tool(name="test_search", description="Search for information")
    async def test_search(query: str) -> str:
        return f"Results for: {query}"

    register_tool(test_search)
    return test_search


class TestHarness:
    """Tests for the Harness class."""

    def test_harness_creation(self, mock_provider):
        """Test creating a Harness."""
        config = HarnessConfig(name="test")
        harness = Harness(mock_provider, config)

        assert harness.provider is mock_provider
        assert harness.config is config
        assert harness.model_name == "test-model"

    def test_harness_generate(self, mock_provider):
        """Test single-turn generate method."""
        config = HarnessConfig(name="test")
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        outputs = harness.generate([request])

        assert len(outputs) == 1
        assert len(outputs[0]) == 1
        assert outputs[0][0].text == "Generated text"
        mock_provider.generate.assert_called_once()

    def test_harness_logprobs(self, mock_provider):
        """Test logprobs method."""
        config = HarnessConfig(name="test")
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.LOGLIKELIHOOD,
            prompt="Test prompt",
            continuations=("continuation",),
        )

        harness.logprobs([request])
        mock_provider.logprobs.assert_called_once()

    def test_harness_apply_config_with_tools(self, mock_provider, sample_tool):
        """Test that _apply_config injects tool schemas."""
        config = HarnessConfig(
            name="with_tools",
            tool_names=("test_search",),
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Search something"},),
        )

        transformed = harness._apply_config(request)

        assert transformed.tools is not None
        assert len(transformed.tools) == 1
        assert transformed.tools[0].name == "test_search"

    def test_harness_apply_config_with_system_prompt(self, mock_provider):
        """Test that _apply_config injects system prompt."""
        config = HarnessConfig(
            name="with_prompt",
            system_prompt="You are a helpful assistant.",
        )
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        transformed = harness._apply_config(request)

        assert transformed.system_prompt == "You are a helpful assistant."

    def test_harness_inject_system_prompt(self, mock_provider):
        """Test system prompt injection into messages."""
        config = HarnessConfig(
            name="inject_test",
            system_prompt="System message",
        )
        harness = Harness(mock_provider, config)

        messages = ({"role": "user", "content": "Hello"},)
        injected = harness._inject_system_prompt(messages)

        assert len(injected) == 2
        assert injected[0]["role"] == "system"
        assert injected[0]["content"] == "System message"
        assert injected[1]["role"] == "user"

    def test_harness_no_inject_if_system_exists(self, mock_provider):
        """Test that system prompt isn't injected if one already exists."""
        config = HarnessConfig(
            name="no_inject",
            system_prompt="New system",
        )
        harness = Harness(mock_provider, config)

        messages = (
            {"role": "system", "content": "Existing system"},
            {"role": "user", "content": "Hello"},
        )
        injected = harness._inject_system_prompt(messages)

        assert len(injected) == 2  # No new system message added
        assert injected[0]["content"] == "Existing system"

    def test_harness_no_inject_if_no_prompt(self, mock_provider):
        """Test that nothing is injected if no system prompt configured."""
        config = HarnessConfig(name="no_prompt")
        harness = Harness(mock_provider, config)

        messages = ({"role": "user", "content": "Hello"},)
        injected = harness._inject_system_prompt(messages)

        assert injected == messages


class TestCreateHarness:
    """Tests for create_harness factory function."""

    def test_create_harness_default(self, mock_provider):
        """Test create_harness with default config."""
        harness = create_harness(mock_provider)

        assert harness.config.name == "default"
        assert harness.provider is mock_provider

    def test_create_harness_with_config(self, mock_provider):
        """Test create_harness with explicit config."""
        config = HarnessConfig(name="explicit")
        harness = create_harness(mock_provider, config)

        assert harness.config.name == "explicit"

    def test_create_harness_with_dict(self, mock_provider):
        """Test create_harness with dict config."""
        config_dict = {
            "name": "from_dict",
            "system_prompt": "Test prompt",
            "max_turns": 5,
        }
        harness = create_harness(mock_provider, config_dict)

        assert harness.config.name == "from_dict"
        assert harness.config.system_prompt == "Test prompt"
        assert harness.config.max_turns == 5


class TestHarnessRun:
    """Tests for Harness.run() multi-turn execution."""

    @pytest.mark.anyio
    async def test_harness_run_no_tools(self, mock_provider):
        """Test run completes immediately when no tools used."""
        # Configure mock to return response without tool calls
        mock_provider.generate.return_value = [[LMOutput(text="Final answer", tool_calls=None)]]

        config = HarnessConfig(name="no_tools", backend="default")
        harness = Harness(mock_provider, config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        result = await harness.run(request)

        assert result.final_text == "Final answer"
        assert result.num_turns == 1
        assert result.max_turns_reached is False

    @pytest.mark.anyio
    async def test_harness_run_batch(self, mock_provider):
        """Test run_batch processes multiple requests."""
        mock_provider.generate.return_value = [[LMOutput(text="Answer", tool_calls=None)]]

        config = HarnessConfig(name="batch_test", backend="default")
        harness = Harness(mock_provider, config)

        requests = [
            LMRequest(
                request_type=RequestType.CHAT,
                messages=({"role": "user", "content": f"Question {i}"},),
            )
            for i in range(3)
        ]

        results = await harness.run_batch(requests)

        assert len(results) == 3
        for result in results:
            assert result.final_text == "Answer"

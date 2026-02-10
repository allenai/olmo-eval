"""Tests for HarnessConfig."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from olmo_eval.harness import clear_registry, register_tool
from olmo_eval.harness.config import HarnessConfig, harness_config
from olmo_eval.harness.tools import tool


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the tool registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


@pytest.fixture
def sample_tool():
    """Create a sample tool for testing."""

    @tool(name="sample_tool", description="A sample tool")
    async def sample(query: str) -> str:
        return query

    register_tool(sample)
    return sample


class TestHarnessConfig:
    """Tests for HarnessConfig dataclass."""

    def test_default_config(self):
        """Test creating a default HarnessConfig."""
        config = HarnessConfig(name="default")

        assert config.name == "default"
        assert config.tool_names == ()
        assert config.system_prompt is None
        assert config.tool_choice == "auto"
        assert config.max_turns is None
        assert config.max_concurrency is None
        assert config.backend is None

    def test_config_with_tools(self, sample_tool):
        """Test HarnessConfig with tool names."""
        config = HarnessConfig(
            name="with_tools",
            tools=("sample_tool",),
        )

        assert config.has_tools is True
        tools = config.resolved_tools
        assert len(tools) == 1
        assert tools[0].name == "sample_tool"

    def test_config_tool_schemas(self, sample_tool):
        """Test getting tool schemas from config."""
        config = HarnessConfig(
            name="schemas",
            tools=("sample_tool",),
        )

        schemas = config.tool_schemas
        assert len(schemas) == 1
        assert schemas[0].name == "sample_tool"

    def test_config_has_tools_false(self):
        """Test has_tools is False when no tools configured."""
        config = HarnessConfig(name="no_tools")
        assert config.has_tools is False

    def test_config_serialization(self):
        """Test to_dict / from_dict round-trip."""
        config = HarnessConfig(
            name="serialize",
            tools=("tool_a", "tool_b"),
            system_prompt="Test prompt",
            tool_choice="required",
            max_turns=5,
            max_concurrency=4,
            backend="openai_agents",
            required_secrets=("API_KEY",),
        )

        d = config.to_dict()
        restored = HarnessConfig.from_dict(d)

        assert restored.name == config.name
        assert restored.tool_names == config.tool_names
        assert restored.system_prompt == config.system_prompt
        assert restored.tool_choice == config.tool_choice
        assert restored.max_turns == config.max_turns
        assert restored.max_concurrency == config.max_concurrency
        assert restored.backend == config.backend
        assert restored.required_secrets == config.required_secrets

    def test_config_immutable(self):
        """Test that HarnessConfig is frozen (immutable)."""
        from dataclasses import FrozenInstanceError

        config = HarnessConfig(name="frozen")

        with pytest.raises(FrozenInstanceError):
            config.name = "changed"

    def test_validate_secrets_all_present(self):
        """Test validate_secrets when all secrets are present."""
        with patch.dict(os.environ, {"TEST_KEY": "value", "OTHER_KEY": "value2"}):
            config = HarnessConfig(
                name="test",
                required_secrets=("TEST_KEY", "OTHER_KEY"),
            )
            missing = config.validate_secrets()
            assert missing == []

    def test_validate_secrets_missing(self):
        """Test validate_secrets when some secrets are missing."""
        with patch.dict(os.environ, {"PRESENT_KEY": "value"}, clear=True):
            config = HarnessConfig(
                name="test",
                required_secrets=("PRESENT_KEY", "MISSING_KEY"),
            )
            missing = config.validate_secrets()
            assert "MISSING_KEY" in missing
            assert "PRESENT_KEY" not in missing

    def test_with_tools_method(self, sample_tool):
        """Test with_tools creates a new config with additional tools."""
        config = HarnessConfig(name="original")
        new_config = config.with_tools("sample_tool")

        assert config.tool_names == ()  # Original unchanged
        assert new_config.tool_names == ("sample_tool",)

    def test_with_system_prompt_method(self):
        """Test with_system_prompt creates a new config."""
        config = HarnessConfig(name="original", system_prompt="old")
        new_config = config.with_system_prompt("new prompt")

        assert config.system_prompt == "old"  # Original unchanged
        assert new_config.system_prompt == "new prompt"


class TestHarnessConfigFactory:
    """Tests for harness_config factory function."""

    def test_harness_config_with_tool_objects(self):
        """Test harness_config accepts Tool objects and registers them."""

        @tool(name="factory_tool")
        async def factory_func(x: str) -> str:
            return x

        register_tool(factory_func)

        config = harness_config(
            name="factory_test",
            tools=[factory_func],  # factory_func is now a Tool object
        )

        assert "factory_tool" in config.tool_names
        # Tool should be registered
        from olmo_eval.harness import get_tool

        assert get_tool("factory_tool") is factory_func

    def test_harness_config_with_tool_names(self, sample_tool):
        """Test harness_config accepts tool names."""
        config = harness_config(
            name="names_test",
            tools=["sample_tool"],
        )

        assert config.tool_names == ("sample_tool",)

    def test_harness_config_mixed_tools(self, sample_tool):
        """Test harness_config with mix of Tool objects and names."""

        @tool(name="another_tool")
        async def another_func(x: str) -> str:
            return x

        config = harness_config(
            name="mixed_test",
            tools=["sample_tool", another_func],  # another_func is now a Tool object
        )

        assert "sample_tool" in config.tool_names
        assert "another_tool" in config.tool_names

    def test_harness_config_all_params(self):
        """Test harness_config with all parameters."""
        config = harness_config(
            name="full_test",
            tools=[],
            system_prompt="Test prompt",
            tool_choice="none",
            max_turns=15,
            max_concurrency=16,
            backend="openai_agents",
            required_secrets=["SECRET"],
        )

        assert config.name == "full_test"
        assert config.system_prompt == "Test prompt"
        assert config.tool_choice == "none"
        assert config.max_turns == 15
        assert config.max_concurrency == 16
        assert config.backend == "openai_agents"
        assert config.required_secrets == ("SECRET",)

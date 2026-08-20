"""Tests for harness presets."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import ProviderKind
from olmo_eval.harness import clear_registry
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.presets import (
    HarnessPresets,
    get_harness_preset,
    list_harness_presets,
    register_harness_preset,
)


@pytest.fixture(autouse=True)
def clean_registry(request):
    """Clear the tool registry before and after each test.

    Tests in TestSearchTools skip this fixture to preserve tool registration.
    """
    # Skip for TestSearchTools class
    if request.node.cls and request.node.cls.__name__ == "TestSearchTools":
        yield
        return

    clear_registry()
    yield
    clear_registry()


class TestHarnessPresets:
    """Tests for harness preset functions."""

    def test_get_default_preset(self):
        """Test getting the default preset."""
        config = get_harness_preset("default")

        assert isinstance(config, HarnessConfig)
        assert config.name == "default"
        assert config.tool_names == ()
        assert config.max_turns is None

    def test_get_dr_tulu_preset(self):
        """Test getting the dr_tulu preset."""
        config = get_harness_preset("dr_tulu")

        assert isinstance(config, HarnessConfig)
        assert config.name == "dr_tulu"
        assert len(config.tool_names) > 0
        assert "semantic_scholar_snippet_search" in config.tool_names
        assert "serper_google_webpage_search" in config.tool_names
        assert config.system_prompt is not None
        assert config.scaffold == "openai_agents"

    def test_get_unknown_preset(self):
        """Test getting an unknown preset raises error."""
        with pytest.raises(ValueError, match="Unknown harness preset"):
            get_harness_preset("nonexistent")

    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_harness_presets()

        assert "default" in presets
        assert "dr_tulu" in presets
        assert presets == sorted(presets)  # Should be sorted

    def test_register_custom_preset(self):
        """Test registering a custom preset."""
        custom = HarnessConfig(
            name="custom",
            system_prompt="Custom prompt",
            max_turns=5,
        )

        register_harness_preset("custom", custom)

        assert hasattr(HarnessPresets, "custom")
        retrieved = get_harness_preset("custom")
        assert retrieved.system_prompt == "Custom prompt"

        # Clean up
        delattr(HarnessPresets, "custom")

    def test_dr_tulu_preset_required_secrets(self):
        """Test that dr_tulu preset has required secrets."""
        config = get_harness_preset("dr_tulu")

        assert "S2_API_KEY" in config.required_secrets
        assert "SERPER_API_KEY" in config.required_secrets

    def test_get_dr_tulu_crawl4ai_preset(self):
        """Test the Dr. Tulu S2 + Serper search + crawl4ai browse preset."""
        config = get_harness_preset("dr_tulu_crawl4ai")

        assert isinstance(config, HarnessConfig)
        assert config.name == "dr_tulu_crawl4ai"
        assert config.tool_names == (
            "semantic_scholar_snippet_search",
            "serper_google_webpage_search",
            "browse_webpage",
        )
        assert config.max_turns == 20
        assert config.required_secrets == ("SERPER_API_KEY",)

    def test_get_web_search_agent_crawl4ai_preset(self):
        """Test the Serper search + crawl4ai browse preset."""
        config = get_harness_preset("web_search_agent_crawl4ai")

        assert isinstance(config, HarnessConfig)
        assert config.name == "web_search_agent_crawl4ai"
        assert config.tool_names == ("serper_google_webpage_search", "browse_webpage")
        assert config.required_secrets == ("SERPER_API_KEY",)
        assert config.provider.kind == ProviderKind.VLLM_SERVER
        assert config.provider.kwargs["timeout"] == 120
        assert config.max_turns == 30
        assert config.max_concurrency == 4
        assert config.scaffold == "openai_agents"
        assert config.batching is not None
        assert config.batching.strategy == "streaming"
        assert config.system_prompt is not None
        assert "serper_google_webpage_search" in config.system_prompt
        assert "browse_webpage" in config.system_prompt

    def test_direct_preset_access(self):
        """Test accessing presets directly via HarnessPresets class."""
        config = HarnessPresets.default
        assert isinstance(config, HarnessConfig)
        assert config.name == "default"

        config = HarnessPresets.dr_tulu
        assert isinstance(config, HarnessConfig)
        assert config.name == "dr_tulu"

    def test_codex_universal_bigcodebench_uses_public_upstream_image(self):
        """Test BigCodeBench sandbox reuses the public upstream execution image."""
        config = get_harness_preset("codex_universal")

        bigcodebench_sandbox = next(
            sandbox
            for sandbox in config.sandboxes
            if sandbox.capabilities == frozenset({"sandbox:bigcodebench"})
        )

        assert bigcodebench_sandbox.image == "bigcodebench/bigcodebench-gradio:latest"
        assert bigcodebench_sandbox.dockerfile_extra == ()
        assert bigcodebench_sandbox.instances is None


class TestSearchTools:
    """Tests for search tools in the search preset."""

    @pytest.fixture(autouse=True)
    def register_search_tools(self):
        """Ensure search tools are registered for these tests."""
        # Force import to trigger @registered_tool decorators
        from olmo_eval.harness import register_tool
        from olmo_eval.harness.tools import search  # noqa: F401

        # Re-register tools since registry may have been cleared
        register_tool(search.semantic_scholar_search)
        register_tool(search.serper_web_search)
        register_tool(search.serper_fetch_page)
        register_tool(search.crawl4ai_browse)

    def test_search_tools_registered(self):
        """Test that search tools are registered when preset is loaded."""
        from olmo_eval.harness import list_tools

        tools = list_tools()
        assert "semantic_scholar_snippet_search" in tools
        assert "serper_google_webpage_search" in tools
        assert "serper_fetch_webpage_content" in tools
        assert "browse_webpage" in tools

    def test_search_tools_have_schemas(self):
        """Test that search tools have valid schemas."""
        config = get_harness_preset("dr_tulu")
        schemas = config.tool_schemas

        assert len(schemas) == 3

        # Check each schema has required fields
        for schema in schemas:
            assert schema.name
            assert schema.description
            assert "properties" in schema.parameters

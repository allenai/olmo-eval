"""Tests for Scaffold implementations."""

from __future__ import annotations

import pytest

from olmo_eval.harness.scaffolds import (
    SCAFFOLD_REGISTRY,
    Scaffold,
    get_scaffold,
    list_scaffolds,
    register_scaffold,
)


class TestGetScaffold:
    """Tests for get_scaffold function."""

    def test_get_unknown_scaffold(self):
        """Test getting an unknown scaffold raises error."""
        with pytest.raises(ValueError, match="Unknown scaffold"):
            get_scaffold("nonexistent")

    def test_register_custom_scaffold(self):
        """Test registering a custom scaffold using the decorator."""

        @register_scaffold("custom")
        class CustomScaffold(Scaffold):
            async def run(
                self,
                provider,
                config,
                request,
                sampling_params=None,
                trace_metadata=None,
                **kwargs,
            ):
                pass

        assert "custom" in SCAFFOLD_REGISTRY
        assert "custom" in list_scaffolds()

        # Clean up
        del SCAFFOLD_REGISTRY["custom"]

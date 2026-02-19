"""Tests for Backend implementations."""

from __future__ import annotations

import pytest

from olmo_eval.harness.backends import (
    BACKEND_REGISTRY,
    Backend,
    get_backend,
    list_backends,
    register_backend,
)


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_unknown_backend(self):
        """Test getting an unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_register_custom_backend(self):
        """Test registering a custom backend using the decorator."""

        @register_backend("custom")
        class CustomBackend(Backend):
            async def run(self, provider, config, request, sampling_params=None):
                pass

        assert "custom" in BACKEND_REGISTRY
        assert "custom" in list_backends()

        # Clean up
        del BACKEND_REGISTRY["custom"]

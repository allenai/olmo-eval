"""Tests for Backend implementations."""

from __future__ import annotations

import pytest

from olmo_eval.core.harness.backends import (
    BACKEND_REGISTRY,
    DefaultBackend,
    get_backend,
    list_backends,
    register_backend,
)
from olmo_eval.core.harness.config import HarnessConfig
from olmo_eval.core.harness.harness import Harness
from olmo_eval.core.types import LMRequest, RequestType


@pytest.fixture
def mock_provider_config():
    """Create a mock provider config."""
    from olmo_eval.core.harness.config import ProviderConfig
    from olmo_eval.core.types import ProviderKind

    return ProviderConfig(kind=ProviderKind.MOCK, model="test-model")


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_default_backend(self):
        """Test getting the default backend."""
        backend = get_backend("default")
        assert isinstance(backend, DefaultBackend)

    def test_get_unknown_backend(self):
        """Test getting an unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_register_custom_backend(self):
        """Test registering a custom backend using the decorator."""
        from olmo_eval.core.harness.backends import Backend

        @register_backend("custom")
        class CustomBackend(Backend):
            async def run(self, provider, config, request, sampling_params=None):
                pass

        assert "custom" in BACKEND_REGISTRY
        assert "custom" in list_backends()

        # Clean up
        del BACKEND_REGISTRY["custom"]


class TestDefaultBackend:
    """Tests for DefaultBackend."""

    @pytest.mark.anyio
    async def test_run_single_generation(self, mock_provider_config):
        """Test run performs single generation."""
        config = HarnessConfig(name="test", provider=mock_provider_config, backend="default")
        harness = Harness(config)

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "Hello"},),
        )

        backend = DefaultBackend()
        result = await backend.run(harness.provider, harness.config, request)

        assert result.final_text is not None
        assert result.num_turns == 1
        assert result.max_turns_reached is False

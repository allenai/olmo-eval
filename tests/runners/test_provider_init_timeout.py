"""Tests for how long the async runner waits for inference workers to initialize."""

from __future__ import annotations

import pytest

from olmo_eval.harness.config import ProviderConfig
from olmo_eval.runners.asynq.monitoring import (
    DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS,
    provider_init_timeout,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS),
        ({"startup_timeout": 300}, DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS),
        ({"startup_timeout": 1800}, 1860.0),
        ({"startup_timeout": "3600"}, 3660.0),
    ],
)
def test_provider_startup_timeout_extends_init_wait(
    kwargs: dict[str, object], expected: float
) -> None:
    provider = ProviderConfig(model="test-model", kind="vllm_server", kwargs=kwargs)

    assert provider_init_timeout(provider) == expected

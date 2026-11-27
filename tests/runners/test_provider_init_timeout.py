"""Tests for how long the async runner waits for inference workers to initialize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import olmo_eval.evals.tasks  # noqa: F401  (registers the tasks)
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.inference.provider_manager import ProviderManager
from olmo_eval.runners.asynq import runner as runner_module
from olmo_eval.runners.asynq.monitoring import (
    DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS,
    provider_init_timeout,
)
from olmo_eval.runners.asynq.runner import AsyncEvalRunner


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS),
        ({"startup_timeout": 300}, DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS + 300),
        ({"startup_timeout": 1800}, DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS + 1800),
    ],
)
def test_provider_startup_timeout_extends_init_wait(
    kwargs: dict[str, object], expected: float
) -> None:
    provider = ProviderConfig(model="test-model", kind="vllm_server", kwargs=kwargs)

    assert provider_init_timeout(provider) == expected


class _StopAtInitWait(Exception):
    """Ends a run at the init wait, before any work is dispatched."""


def test_the_runner_waits_for_the_provider_startup_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[float] = []

    def prepare(spec: str, model_name: str, overrides: Any, sampling_overrides: Any = None):
        return get_task(spec, overrides), []

    def wait(*args: Any, timeout: float, **kwargs: Any) -> dict[str, float]:
        waits.append(timeout)
        raise _StopAtInitWait

    # No worker starts, so the mock provider never receives startup_timeout.
    monkeypatch.setattr(ProviderManager, "start", lambda self, *args: [])
    monkeypatch.setattr(runner_module, "prepare_task_items", prepare)
    monkeypatch.setattr(runner_module, "wait_for_init_times", wait)
    provider = ProviderConfig(model="mock", kind="mock", kwargs={"startup_timeout": 1800})
    runner = AsyncEvalRunner(
        harness_config=HarnessConfig(name="test", provider=provider),
        task_specs=["gsm8k"],
        output_dir=str(tmp_path),
    )

    with pytest.raises(_StopAtInitWait):
        runner.run()

    assert waits == [DEFAULT_PROVIDER_INIT_TIMEOUT_SECONDS + 1800]

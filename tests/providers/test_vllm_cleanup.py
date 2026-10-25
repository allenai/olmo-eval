"""Release the inline vLLM engine through the evaluator's worker lifecycle."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.inference.providers.vllm import VLLMProvider
from olmo_eval.runners.asynq.workers import inference_worker


@pytest.fixture
def engine(monkeypatch):
    core = SimpleNamespace(shutdown=Mock())
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=core))
    constructor = Mock(return_value=llm)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=constructor))
    return core, constructor


def test_close_releases_engine_once(engine):
    core, constructor = engine
    provider = VLLMProvider("model", seed=1234)
    assert constructor.call_args.kwargs["model"] == "model"
    assert constructor.call_args.kwargs["seed"] == 1234
    core.shutdown.assert_not_called()

    provider.close()
    provider.close()

    core.shutdown.assert_called_once_with()


def test_close_propagates_failure_and_can_retry(engine):
    core, _ = engine
    core.shutdown.side_effect = [RuntimeError("shutdown failed"), None]
    provider = VLLMProvider("model")

    with pytest.raises(RuntimeError, match="shutdown failed"):
        provider.close()
    provider.close()
    provider.close()

    assert core.shutdown.call_count == 2


@pytest.mark.parametrize("generation_fails", [False, True])
def test_worker_releases_vllm_after_processing(engine, monkeypatch, generation_fails):
    core, _ = engine
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("olmo_eval.inference._print_model_config", Mock())
    monkeypatch.setattr("olmo_eval.common.logging.configure_logging", Mock())
    monkeypatch.setattr(
        "olmo_eval.common.logging.configure_worker_logging", Mock(return_value=Mock())
    )
    strategy = SimpleNamespace(
        run=AsyncMock(side_effect=RuntimeError("generation failed") if generation_fails else None)
    )
    monkeypatch.setattr(
        "olmo_eval.runners.asynq.batching.get_strategy", Mock(return_value=strategy)
    )
    cleanup = AsyncMock()

    def make_harness(config):
        return SimpleNamespace(
            provider=config.provider.create_provider(),
            initialize_reporters=Mock(),
            cleanup=cleanup,
        )

    monkeypatch.setattr("olmo_eval.harness.Harness", make_harness)
    config = HarnessConfig(name="test", provider=ProviderConfig(kind="vllm", model="model"))
    results = Mock()

    with pytest.raises(SystemExit) if generation_fails else nullcontext() as exit_info:
        inference_worker("test", [0], Mock(), results, config.to_dict(), 0)

    strategy.run.assert_awaited_once()
    cleanup.assert_awaited_once()
    core.shutdown.assert_called_once_with()
    if generation_fails:
        assert exit_info.value.code == 1
        results.put.assert_called_once()
        assert "generation failed" in results.put.call_args.args[0].error
    else:
        results.put.assert_not_called()

"""Generation diagnostics retained from API responses."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
)
from olmo_eval.inference.providers.litellm import LiteLLMProvider
from olmo_eval.runners.processing.generation_counts import count_generations


@pytest.mark.parametrize("finish_reason", ["length", "stop", None])
@pytest.mark.parametrize("with_logprobs", [False, True])
def test_finish_reason_survives_logprob_metadata(finish_reason, with_logprobs) -> None:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider.model_name = "mock"
    provider.api_base = None
    provider.api_kwargs = {}
    provider._request_overrides = {}
    provider._openai_params = None
    logprobs = SimpleNamespace(content=[SimpleNamespace(token="answer", logprob=-0.1)])
    choice = SimpleNamespace(
        message=SimpleNamespace(content="answer"),
        logprobs=logprobs if with_logprobs else None,
        finish_reason=finish_reason,
    )
    provider._litellm = SimpleNamespace(
        acompletion=AsyncMock(return_value=SimpleNamespace(choices=[choice]))
    )
    request = LMRequest(request_type=RequestType.CHAT, prompt="question")

    outputs = asyncio.run(provider._generate_single_impl(request, SamplingParams()))
    assert outputs[0].metadata["finish_reason"] == finish_reason
    counts = count_generations(
        [Response(instance=Instance(question="q"), request=request, outputs=outputs)]
    )
    assert counts is not None
    assert counts["cap_hit"] == int(finish_reason == "length")
    assert counts["finish_reason_unknown"] == int(finish_reason is None)


@pytest.mark.parametrize(("count", "expected_level"), [(1, "debug"), (3, "info")])
def test_only_multi_request_calls_log_at_info(count, expected_level, monkeypatch) -> None:
    from olmo_eval.inference.providers import litellm as litellm_module

    logged: list[tuple[str, str]] = []

    class Recorder:
        def __getattr__(self, level: str):
            return lambda message, *args, **kwargs: logged.append((level, message))

    monkeypatch.setattr(litellm_module, "logger", Recorder())
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider.model_name = "mock"
    provider.max_concurrency = 4
    provider.max_retries = 0
    provider._default_sampling_params = lambda params: SamplingParams()  # type: ignore[method-assign]
    provider._generate_single_async = AsyncMock(  # type: ignore[method-assign]
        return_value=[LMOutput(text="x")]
    )
    requests = [LMRequest(request_type=RequestType.CHAT, prompt="q")] * count

    asyncio.run(provider.agenerate(requests))

    assert {level for level, message in logged if "Sending" in message} == {expected_level}
    assert any("Generating" in message for _, message in logged) == (count > 1)

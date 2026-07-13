"""Unit tests for LiteLLMProvider request construction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.litellm import LiteLLMProvider


def _make_provider(*, generation_logprobs: bool) -> LiteLLMProvider:
    provider = LiteLLMProvider.__new__(LiteLLMProvider)
    provider.model_name = "test-model"
    provider.generation_logprobs = generation_logprobs
    provider.api_base = None
    provider.api_kwargs = {}

    message = SimpleNamespace(content="ok")
    choice = SimpleNamespace(message=message, logprobs=None)
    provider._litellm = SimpleNamespace(
        acompletion=AsyncMock(return_value=SimpleNamespace(choices=[choice]))
    )
    return provider


@pytest.mark.anyio
async def test_generate_omits_logprobs_when_disabled() -> None:
    provider = _make_provider(generation_logprobs=False)
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hi"},),
    )

    await provider._generate_single_impl(request, SamplingParams(max_tokens=1))

    call_kwargs = provider._litellm.acompletion.call_args.kwargs
    assert "logprobs" not in call_kwargs
    assert "top_logprobs" not in call_kwargs


@pytest.mark.anyio
async def test_generate_explicit_logprobs_overrides_disabled_provider() -> None:
    provider = _make_provider(generation_logprobs=False)
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hi"},),
    )

    await provider._generate_single_impl(request, SamplingParams(max_tokens=1, logprobs=4))

    call_kwargs = provider._litellm.acompletion.call_args.kwargs
    assert call_kwargs["logprobs"] is True
    assert call_kwargs["top_logprobs"] == 4

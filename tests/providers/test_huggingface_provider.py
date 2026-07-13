"""Unit tests for HuggingFaceProvider generate-kwargs construction."""

from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.huggingface import HuggingFaceProvider


@pytest.fixture
def provider() -> HuggingFaceProvider:
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
    instance.tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2, 3])
    instance.generation_logprobs = True
    return instance


def test_finite_max_tokens_passes_through(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=512), prompt_len=100)
    assert kwargs["max_new_tokens"] == 512


def test_uncapped_reserves_room_after_prompt(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=2000)
    assert kwargs["max_new_tokens"] == 2048 - 2000


def test_uncapped_with_no_prompt_uses_full_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None))
    assert kwargs["max_new_tokens"] == 2048


def test_uncapped_floors_at_one_when_prompt_exceeds_context(provider: HuggingFaceProvider) -> None:
    kwargs = provider._build_generate_kwargs(SamplingParams(max_tokens=None), prompt_len=5000)
    assert kwargs["max_new_tokens"] == 1


def test_describe_request_omits_logprobs_when_disabled(provider: HuggingFaceProvider) -> None:
    provider.generation_logprobs = False

    trace = provider.describe_request(
        LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
        SamplingParams(max_tokens=1),
    )

    assert trace is not None
    assert "logprobs" not in trace["generation_kwargs"]


def test_describe_request_explicit_logprobs_overrides_disabled_provider(
    provider: HuggingFaceProvider,
) -> None:
    provider.generation_logprobs = False

    trace = provider.describe_request(
        LMRequest(request_type=RequestType.COMPLETION, prompt="Prompt"),
        SamplingParams(max_tokens=1, logprobs=2),
    )

    assert trace is not None
    assert trace["generation_kwargs"]["logprobs"] == 2

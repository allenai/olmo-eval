"""Unit tests for HuggingFaceProvider generate-kwargs construction."""

from types import SimpleNamespace

import pytest

from olmo_eval.common.types import SamplingParams
from olmo_eval.inference.providers.huggingface import HuggingFaceProvider


@pytest.fixture
def provider() -> HuggingFaceProvider:
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=2048))
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


class _TemplateBosTokenizer:
    """Puts BOS (1) first when encoding with special tokens, like Llama 3 and Gemma."""

    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [ord(c) for c in text]
        return [self.bos_token_id, *ids] if add_special_tokens else ids


def test_generation_and_likelihood_start_the_same() -> None:
    """Generation prompts and scored contexts start with the tokenizer's BOS, once."""
    from olmo_eval.inference.tokenizer_utils import (
        default_prefix_ids,
        encode_context_and_continuation,
    )

    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.tokenizer = _TemplateBosTokenizer()
    instance._prefix_ids = default_prefix_ids(instance.tokenizer)

    ctx, _ = encode_context_and_continuation(
        instance.tokenizer, "ab", "cd", prefix_ids=instance._prefix_ids
    )
    assert ctx == [1, ord("a"), ord("b")]
    assert instance._generation_prompt_ids("ab") == ctx
    # A chat template that wrote BOS into the text does not get a second one
    assert instance._generation_prompt_ids("\x01ab") == ctx


def test_add_bos_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="add_bos_token was removed"):
        HuggingFaceProvider("test-model", add_bos_token=True)

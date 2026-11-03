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


@pytest.mark.parametrize(("add_bos_token", "first"), [(None, [1]), (True, [1]), (False, [])])
def test_generation_and_likelihood_start_the_same(add_bos_token, first) -> None:
    """Generation prompts and scored contexts start with the same resolved prefix."""
    from olmo_eval.inference.tokenizer_utils import encode_context_and_continuation

    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.tokenizer = _TemplateBosTokenizer()
    instance.device = "cpu"
    instance._add_bos_token = add_bos_token

    ctx, _ = encode_context_and_continuation(
        instance.tokenizer, "ab", "cd", prefix_ids=instance._prefix_ids()
    )
    assert ctx == [*first, ord("a"), ord("b")]
    if add_bos_token is not None:
        pytest.importorskip("torch")
        encoded = instance._encode_generation_prompt("ab")
        assert encoded["input_ids"].tolist() == [ctx]

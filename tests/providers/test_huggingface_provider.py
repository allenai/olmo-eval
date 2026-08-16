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


def test_stop_decoding_preserves_tokenizer_context(provider: HuggingFaceProvider) -> None:
    class ContextAwareTokenizer:
        pieces = {1: "Okay", 2: "Ġso", 3: "ĊSTOP", 4: "Ġignored"}

        def decode(self, tokens, *, skip_special_tokens: bool) -> str:
            del skip_special_tokens
            if isinstance(tokens, int):
                return self.pieces[tokens]
            text = "".join(self.pieces[token] for token in tokens)
            return text.replace("Ġ", " ").replace("Ċ", "\n")

    provider.tokenizer = ContextAwareTokenizer()

    tokens, text = provider._truncate_at_stop([1, 2, 3, 4], ("\nSTOP",))

    assert tokens == [1, 2, 3]
    assert text == "Okay so"


def test_stop_decoding_uses_earliest_stop(provider: HuggingFaceProvider) -> None:
    class Tokenizer:
        pieces = {1: "alpha", 2: " END", 3: " later STOP"}

        def decode(self, tokens, *, skip_special_tokens: bool) -> str:
            del skip_special_tokens
            if isinstance(tokens, int):
                return self.pieces[tokens]
            return "".join(self.pieces[token] for token in tokens)

    provider.tokenizer = Tokenizer()

    tokens, text = provider._truncate_at_stop([1, 2, 3], ("STOP", "END"))

    assert tokens == [1, 2]
    assert text == "alpha "

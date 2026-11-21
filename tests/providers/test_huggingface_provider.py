"""Unit tests for HuggingFaceProvider generate-kwargs construction."""

from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
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


class CharTokenizer:
    """Character tokenizer without a BOS token, as the OLMo tokenizers are."""

    bos_token_id = None
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_id: int, skip_special_tokens: bool = False) -> str:
        return chr(token_id)


class FlatModel:
    """A model whose next-token distribution is uniform over 256 tokens."""

    def __init__(self) -> None:
        self.inputs: list[list[int]] = []

    def __call__(self, input_ids):
        import torch

        self.inputs.append(input_ids[0].tolist())
        return SimpleNamespace(logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], 256))


def _scoring_provider(add_bos_token: bool | None) -> tuple[HuggingFaceProvider, FlatModel]:
    model = FlatModel()
    instance = HuggingFaceProvider.__new__(HuggingFaceProvider)
    instance.model = model
    instance.tokenizer = CharTokenizer()
    instance.device = "cpu"
    instance.is_multimodal = False
    instance._add_bos_token = add_bos_token
    return instance, model


def _empty_context_request() -> LMRequest:
    return LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="", continuations=("hi",))


def test_logprobs_puts_eos_ahead_of_an_empty_context_by_default() -> None:
    provider, model = _scoring_provider(add_bos_token=None)

    [[output]] = provider.logprobs([_empty_context_request()])

    assert model.inputs == [[2, ord("h"), ord("i")]]
    assert [entry["token_id"] for entry in output.logprobs] == [ord("h"), ord("i")]


def test_logprobs_scores_an_empty_context_from_a_cold_start_when_bos_is_disabled() -> None:
    provider, model = _scoring_provider(add_bos_token=False)

    [[output]] = provider.logprobs([_empty_context_request()])

    assert model.inputs == [[ord("h"), ord("i")]]
    assert [entry["token_id"] for entry in output.logprobs] == [ord("i")]


def test_init_keeps_add_bos_token_for_scoring(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_init_text(self, model_name, tokenizer, model_kwargs):
        captured.update(model_kwargs)
        self.model = SimpleNamespace(to=lambda device: None, eval=lambda: None)

    monkeypatch.setattr(HuggingFaceProvider, "_init_text", fake_init_text)
    provider = HuggingFaceProvider("m", device="cpu", add_bos_token=False, max_model_len=8)

    assert provider._add_bos_token is False
    assert "add_bos_token" not in captured
    assert "max_model_len" not in captured

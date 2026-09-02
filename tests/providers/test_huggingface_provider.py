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


class _FakeChatTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        rendered = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"<chat>{rendered}<assistant>"


def test_text_path_renders_chat_messages(provider: HuggingFaceProvider) -> None:
    # A CHAT request keeps its content on `messages` and leaves `prompt` empty. Reading
    # `prompt` alone produced "" -> zero tokens, which surfaced from inside attention as
    # "cannot reshape tensor of 0 elements into shape [1, 0, -1, 128]" and failed all 900
    # MMMU instances on a text-only backbone run.
    provider.tokenizer = _FakeChatTokenizer()
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "vqa2: What is 2+2?"},),
    )

    assert provider._format_text_prompt(request) == "<chat>user:vqa2: What is 2+2?<assistant>"


def test_text_path_leaves_completion_prompts_alone(provider: HuggingFaceProvider) -> None:
    provider.tokenizer = _FakeChatTokenizer()
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="raw text")

    assert provider._format_text_prompt(request) == "raw text"

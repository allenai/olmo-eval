"""Focused tests for provider prompt truncation semantics."""

from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.huggingface import HuggingFaceProvider
from olmo_eval.inference.providers.litellm import LiteLLMProvider
from olmo_eval.inference.providers.olmo_core import OlmoCoreProvider
from olmo_eval.inference.providers.vllm import VLLMProvider
from olmo_eval.inference.tokenizer_utils import truncate_token_ids


class FakeTokenizer:
    bos_token_id = 99
    eos_token_id = 0
    truncation_side = "left"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = list(range(1, len(text) + 1))
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def decode(self, token_ids, **kwargs) -> str:
        del kwargs
        return "".join(chr(int(token_id)) for token_id in token_ids)


class FakeLiteLLM:
    @staticmethod
    def encode(model: str, text: str) -> list[int]:
        del model
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(model: str, tokens: list[int]) -> str:
        del model
        return bytes(tokens).decode("utf-8")

    @staticmethod
    def token_counter(model: str, messages: list[dict]) -> int:
        del model
        return 2 * len(messages) + sum(
            len(message.get("content", ""))
            for message in messages
            if isinstance(message.get("content"), str)
        )

    @staticmethod
    def get_model_info(model: str) -> dict[str, int]:
        del model
        return {"max_input_tokens": 8}


def test_truncate_token_ids_matches_vllm_side_semantics():
    token_ids = [1, 2, 3, 4, 5]

    assert truncate_token_ids(token_ids, 3, "left") == [3, 4, 5]
    assert truncate_token_ids(token_ids, 3, "right") == [1, 2, 3]

    tokenizer = FakeTokenizer()
    assert truncate_token_ids(token_ids, 3, None, tokenizer=tokenizer) == [3, 4, 5]
    assert truncate_token_ids(token_ids, -1, "right", max_input_tokens=2) == [1, 2]


def test_truncate_token_ids_rejects_invalid_limits():
    with pytest.raises(ValueError, match="positive integer"):
        truncate_token_ids([1, 2], 0)
    with pytest.raises(ValueError, match="requires a positive max_input_tokens"):
        truncate_token_ids([1, 2], -1)


def test_huggingface_provider_truncates_pre_tokenized_prompt():
    provider = object.__new__(HuggingFaceProvider)
    provider.tokenizer = FakeTokenizer()
    provider.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=16))

    params = SamplingParams(max_tokens=1, truncate_prompt_tokens=3, truncation_side="left")
    assert provider._prompt_token_ids("abcdef", params) == [4, 5, 6]


def test_vllm_provider_truncates_prompt_ids_without_reencoding():
    tokenizer = FakeTokenizer()
    provider = object.__new__(VLLMProvider)
    provider.llm = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        llm_engine=SimpleNamespace(model_config=SimpleNamespace(max_model_len=16)),
    )
    provider._add_bos_token = False

    params = SamplingParams(max_tokens=1, truncate_prompt_tokens=3, truncation_side="right")
    assert provider._prompt_token_ids("abcdef", params) == [1, 2, 3]


def test_olmo_core_provider_uses_requested_side():
    provider = object.__new__(OlmoCoreProvider)
    provider.tokenizer = FakeTokenizer()
    provider.max_length = 16

    params = SamplingParams(max_tokens=1, truncate_prompt_tokens=3, truncation_side="left")
    assert provider._apply_requested_prompt_truncation([1, 2, 3, 4, 5], params) == [3, 4, 5]


def test_litellm_provider_truncates_message_content_and_preserves_structure():
    provider = object.__new__(LiteLLMProvider)
    provider.model_name = "fake/model"
    provider._litellm = FakeLiteLLM()
    request = LMRequest(request_type=RequestType.CHAT, max_length=8)

    right = provider._truncate_messages(
        [{"role": "user", "content": "abcdef"}],
        request,
        SamplingParams(max_tokens=1, truncate_prompt_tokens=5, truncation_side="right"),
    )
    left = provider._truncate_messages(
        [{"role": "user", "content": "abcdef"}],
        request,
        SamplingParams(max_tokens=1, truncate_prompt_tokens=5, truncation_side="left"),
    )

    assert right == [{"role": "user", "content": "abc"}]
    assert left == [{"role": "user", "content": "def"}]


def test_litellm_minus_one_uses_request_input_budget():
    provider = object.__new__(LiteLLMProvider)
    provider.model_name = "fake/model"
    provider._litellm = FakeLiteLLM()
    request = LMRequest(request_type=RequestType.CHAT, max_length=6)

    truncated = provider._truncate_messages(
        [{"role": "user", "content": "abcdef"}],
        request,
        SamplingParams(max_tokens=1, truncate_prompt_tokens=-1, truncation_side="right"),
    )

    # max_length 6 minus one output token leaves a five-token prompt budget;
    # the fake chat template contributes two overhead tokens, so content keeps three.
    assert truncated == [{"role": "user", "content": "abc"}]

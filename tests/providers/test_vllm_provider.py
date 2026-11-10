"""Unit tests for the inline VLLMProvider."""

import re
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from olmo_eval.common.types import (
    CONTEXT_CUT_TOKENS_KEY,
    ContextFit,
    LMRequest,
    RequestType,
    SamplingParams,
)
from olmo_eval.inference.providers.vllm import VLLMProvider


class FakeVllmModule(ModuleType):
    SamplingParams: object


def test_build_sampling_params_passes_none_max_tokens_through(monkeypatch) -> None:
    # vLLM treats max_tokens=None as "generate to the context limit", so the
    # provider must forward None unchanged rather than crashing or coercing it.
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    provider = VLLMProvider.__new__(VLLMProvider)
    built = provider._build_sampling_params(SamplingParams(max_tokens=None, do_sample=False))

    assert built["max_tokens"] is None


class FakeTokenizer:
    def __init__(self) -> None:
        self.template_calls: list[dict[str, object]] = []
        self.encode_calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        self.template_calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered_messages = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"<chat>{rendered_messages}<assistant>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        return [ord(char) for char in text]


class FakeLLM:
    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer
        self.generate_calls: list[dict[str, object]] = []

    def get_tokenizer(self) -> FakeTokenizer:
        return self.tokenizer

    def generate(
        self,
        prompts: list[str] | list[dict[str, list[int]]],
        sampling_params: object,
        use_tqdm: bool = False,
    ) -> list[object]:
        self.generate_calls.append(
            {
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
            }
        )
        completion = SimpleNamespace(
            text="ok", logprobs=None, finish_reason="length", token_ids=[7]
        )
        return [SimpleNamespace(outputs=[completion]) for _ in prompts]


@pytest.fixture
def fake_provider() -> tuple[VLLMProvider, FakeLLM, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    llm = FakeLLM(tokenizer)
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._add_bos_token = None
    provider._build_sampling_params = lambda params: "fake-sampling-params"
    return provider, llm, tokenizer


def test_generate_formats_chat_messages_with_template(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    outputs = provider.generate([request], SamplingParams(max_tokens=1))

    assert outputs[0][0].text == "ok"
    assert tokenizer.template_calls == [
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "tokenize": False,
            "add_generation_prompt": True,
        }
    ]
    assert llm.generate_calls[0]["prompts"] == ["<chat>user:Hello<assistant>"]


def test_generate_tokenizes_formatted_chat_prompt_when_bos_disabled(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    provider._add_bos_token = False
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello"},),
    )

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.encode_calls == [
        {
            "text": "<chat>user:Hello<assistant>",
            "add_special_tokens": False,
        }
    ]
    assert llm.generate_calls[0]["prompts"] == [
        {"prompt_token_ids": [ord(char) for char in "<chat>user:Hello<assistant>"]}
    ]


def test_generate_keeps_completion_prompt_unchanged(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Complete me")

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.template_calls == []
    assert llm.generate_calls[0]["prompts"] == ["Complete me"]


def test_generate_preserves_termination_without_logprobs(fake_provider) -> None:
    provider, _, _ = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Q")
    output = provider.generate([request], SamplingParams(max_tokens=1))[0][0]
    assert output.metadata["finish_reason"] == "length"
    assert output.metadata["completion_tokens"] == 1


class WordTokenizer(FakeTokenizer):
    """One token per whitespace-separated word, with character offsets."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        super().encode(text, add_special_tokens)
        return list(range(len(text.split())))

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):  # noqa: ANN001, ANN204
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


CONTEXT = "one two three four five six seven eight"


def _cutting_provider(max_length: int) -> tuple[VLLMProvider, FakeLLM]:
    llm = FakeLLM(WordTokenizer())
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._add_bos_token = None
    provider._max_length = max_length
    provider._build_sampling_params = lambda params: "fake-sampling-params"
    return provider, llm


def _completion(budget: int | None = None) -> LMRequest:
    return LMRequest(
        request_type=RequestType.COMPLETION,
        prompt=f"Read this. {CONTEXT} Question: q\nAnswer:",
        context_fit=ContextFit(text=CONTEXT, max_prompt_tokens=budget),
    )


def test_a_prompt_within_budget_is_sent_unchanged() -> None:
    provider, llm = _cutting_provider(max_length=100)
    request = _completion(budget=100)

    output = provider.generate([request], SamplingParams(max_tokens=1))[0][0]

    assert llm.generate_calls[0]["prompts"] == [request.prompt]
    assert CONTEXT_CUT_TOKENS_KEY not in output.metadata


def test_the_overflow_is_cut_from_the_end_of_the_context() -> None:
    # 13 words against a budget of 10: the last three context words go, and the
    # instruction before the context and the question after it survive, as the
    # vllm_server provider cuts them.
    provider, llm = _cutting_provider(max_length=100)

    output = provider.generate([_completion(budget=10)], SamplingParams(max_tokens=1))[0][0]

    assert llm.generate_calls[0]["prompts"] == [
        "Read this. one two three four five  Question: q\nAnswer:"
    ]
    assert output.metadata[CONTEXT_CUT_TOKENS_KEY] == 3


def test_the_window_caps_the_budget_after_the_generation() -> None:
    # a 20-token window with 8 tokens to generate leaves room for 12
    provider, llm = _cutting_provider(max_length=20)

    output = provider.generate([_completion(budget=100)], SamplingParams(max_tokens=8))[0][0]

    assert len(llm.generate_calls[0]["prompts"][0].split()) == 12
    assert output.metadata[CONTEXT_CUT_TOKENS_KEY] == 1


def test_a_chat_request_is_counted_with_its_template() -> None:
    provider, llm = _cutting_provider(max_length=100)
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": f"Read this. {CONTEXT} Question: q"},),
        context_fit=ContextFit(text=CONTEXT, max_prompt_tokens=10),
    )

    output = provider.generate([request], SamplingParams(max_tokens=1))[0][0]

    # counted as rendered: the template fuses into the first and last words, 12 against 10
    assert llm.generate_calls[0]["prompts"] == [
        "<chat>user:Read this. one two three four five six  Question: q<assistant>"
    ]
    assert output.metadata[CONTEXT_CUT_TOKENS_KEY] == 2


def test_text_that_is_not_in_the_prompt_is_refused() -> None:
    provider, _ = _cutting_provider(max_length=100)
    request = replace(
        _completion(budget=1), context_fit=ContextFit(text="missing", max_prompt_tokens=1)
    )

    with pytest.raises(ValueError, match="exactly once"):
        provider.generate([request], SamplingParams(max_tokens=1))

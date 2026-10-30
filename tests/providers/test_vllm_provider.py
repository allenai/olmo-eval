"""Unit tests for the inline VLLMProvider."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from olmo_eval.common.types import TRUNCATABLE_SPAN_KEY, LMRequest, RequestType, SamplingParams
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

    def __call__(
        self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False
    ) -> dict[str, list]:
        encoded: dict[str, list] = {"input_ids": [ord(char) for char in text]}
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return encoded


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


def test_generate_cuts_the_end_of_the_truncatable_span_to_fit(fake_provider) -> None:
    """Like HELMET's tokenize(): drop the span's last tokens until prompt + generation
    fit the context, keeping the text around the span."""
    provider, llm, _ = fake_provider
    provider._max_length = 20
    prompt = "Head:" + "abcdefghij" + "|Tail"  # 20 one-character tokens
    request = LMRequest(
        request_type=RequestType.COMPLETION,
        prompt=prompt,
        metadata={TRUNCATABLE_SPAN_KEY: (5, 15)},
    )

    output = provider.generate([request], SamplingParams(max_tokens=4))[0][0]

    assert llm.generate_calls[0]["prompts"] == ["Head:abcdef|Tail"]
    assert output.metadata["prompt_truncated_tokens"] == 4


def test_generate_leaves_a_fitting_or_spanless_prompt_alone(fake_provider) -> None:
    provider, llm, _ = fake_provider
    provider._max_length = 20
    fits = LMRequest(
        request_type=RequestType.COMPLETION,
        prompt="Head:abc|Tail",
        metadata={TRUNCATABLE_SPAN_KEY: (5, 8)},
    )
    spanless = LMRequest(request_type=RequestType.COMPLETION, prompt="x" * 30)

    outputs = provider.generate([fits, spanless], SamplingParams(max_tokens=4))

    assert llm.generate_calls[0]["prompts"] == ["Head:abc|Tail", "x" * 30]
    assert all("prompt_truncated_tokens" not in o[0].metadata for o in outputs)

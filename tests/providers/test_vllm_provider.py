"""Unit tests for the inline VLLMProvider."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.vllm import VLLMProvider
from olmo_eval.inference.tokenizer_utils import default_prefix_ids


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
    """Character-level tokenizer; with a bos_token_id it adds BOS like Gemma or Llama 3."""

    def __init__(self, bos_token_id: int | None = None) -> None:
        self.bos_token_id = bos_token_id
        self.eos_token_id = None
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
        # Like Gemma's and Llama 3's templates, write BOS into the text
        bos = chr(self.bos_token_id) if self.bos_token_id is not None else ""
        return f"{bos}<chat>{rendered_messages}<assistant>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
            }
        )
        ids = [ord(char) for char in text]
        if add_special_tokens and self.bos_token_id is not None:
            return [self.bos_token_id, *ids]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


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
        return [
            SimpleNamespace(
                outputs=[completion],
                # Every input token is the argmax with log-prob -0.5
                prompt_logprobs=[None] + [{t: -0.5} for t in p["prompt_token_ids"][1:]]
                if isinstance(p, dict)
                else None,
            )
            for p in prompts
        ]


@pytest.fixture
def fake_provider() -> tuple[VLLMProvider, FakeLLM, FakeTokenizer]:
    tokenizer = FakeTokenizer()
    llm = FakeLLM(tokenizer)
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._prefix_ids = []
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
    assert tokenizer.encode_calls == [
        {"text": "<chat>user:Hello<assistant>", "add_special_tokens": True}
    ]
    assert llm.generate_calls[0]["prompts"] == [
        {"prompt_token_ids": [ord(char) for char in "<chat>user:Hello<assistant>"]}
    ]


def _bos_provider(monkeypatch) -> tuple[VLLMProvider, FakeLLM]:
    """A provider whose tokenizer adds BOS (id 1) in front of every text."""
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    llm = FakeLLM(FakeTokenizer(bos_token_id=1))
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._max_length = 4096
    provider._prefix_ids = default_prefix_ids(llm.tokenizer)
    provider._build_sampling_params = lambda params: "fake-sampling-params"
    return provider, llm


def test_chat_prompt_gets_one_bos_when_the_template_writes_it(monkeypatch) -> None:
    provider, llm = _bos_provider(monkeypatch)
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hi"},),
    )

    provider.generate([request], SamplingParams(max_tokens=1))

    ids = llm.generate_calls[0]["prompts"][0]["prompt_token_ids"]
    assert ids[:2] == [1, ord("<")]


def test_generation_and_logprobs_start_with_the_tokenizer_bos(monkeypatch) -> None:
    """Log-likelihood inputs start with BOS like generation prompts, and the
    scored span is exactly the continuation."""
    provider, llm = _bos_provider(monkeypatch)

    provider.generate(
        [LMRequest(request_type=RequestType.COMPLETION, prompt="ab")],
        SamplingParams(max_tokens=1),
    )
    outputs = provider.logprobs(
        [
            LMRequest(
                request_type=RequestType.LOGLIKELIHOOD,
                prompt="ab",
                continuations=(" cd",),
            )
        ]
    )

    generated = llm.generate_calls[0]["prompts"][0]["prompt_token_ids"]
    scored = llm.generate_calls[1]["prompts"][0]["prompt_token_ids"]
    assert generated == [1, ord("a"), ord("b")]
    assert scored == [1, ord("a"), ord("b"), ord(" "), ord("c"), ord("d")]
    assert [e["token_id"] for e in outputs[0][0].logprobs] == [ord(" "), ord("c"), ord("d")]


def test_add_bos_token_is_rejected() -> None:
    with pytest.raises(ValueError, match="add_bos_token was removed"):
        VLLMProvider("test-model", add_bos_token=False)


def test_generate_keeps_completion_prompt_unchanged(
    fake_provider: tuple[VLLMProvider, FakeLLM, FakeTokenizer],
) -> None:
    provider, llm, tokenizer = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Complete me")

    provider.generate([request], SamplingParams(max_tokens=1))

    assert tokenizer.template_calls == []
    assert llm.generate_calls[0]["prompts"] == [
        {"prompt_token_ids": [ord(char) for char in "Complete me"]}
    ]


def test_generate_preserves_termination_without_logprobs(fake_provider) -> None:
    provider, _, _ = fake_provider
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Q")
    output = provider.generate([request], SamplingParams(max_tokens=1))[0][0]
    assert output.metadata["finish_reason"] == "length"
    assert output.metadata["completion_tokens"] == 1

"""Unit tests for the inline VLLMProvider."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
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


@pytest.mark.parametrize(
    "specials",
    [
        ("<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|return|>"),
        ("<|meta_sep|>", "<|im_sep|>", "<|im_end|>", "<|im_start|>", "<|fim_suffix|>"),
    ],
)
@pytest.mark.parametrize("string_stop", [False, True])
def test_harmony_token_boundaries_reach_runner_predictions(
    fake_provider, monkeypatch, specials, string_stop
) -> None:
    import asyncio
    from dataclasses import replace

    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    from olmo_eval.common.types import Instance, Response
    from olmo_eval.evals.tasks.common import get_task
    from olmo_eval.runners.asynq.preparation import finalize_task
    from olmo_eval.runners.asynq.types import TaskTracker

    provider, llm, tokenizer = fake_provider
    channel, message, end, start, done = specials
    tokens = [
        channel,
        "analysis",
        message,
        "wrong",
        end,
        start,
        "assistant",
        channel,
        "final",
        message,
        "good",
        "HALT" if string_stop else done,
    ]
    raw = "".join(tokens)
    codec = Tokenizer(
        models.WordLevel(
            {
                word: i
                for i, word in enumerate(
                    ["analysis", "wrong", "assistant", "final", "good", "HALT"]
                )
            }
        )
    )
    codec.decoder = decoders.Fuse()
    real_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=codec, additional_special_tokens=list(specials)
    )
    token_ids = real_tokenizer.convert_tokens_to_ids(tokens)
    decoded = real_tokenizer.decode(token_ids, skip_special_tokens=True)
    if string_stop:
        decoded = decoded.removesuffix("HALT")
    assert decoded == "analysiswrongassistantfinalgood"
    assert real_tokenizer.decode(token_ids, skip_special_tokens=False) == raw
    monkeypatch.setattr(tokenizer, "get_added_vocab", real_tokenizer.get_added_vocab, raising=False)
    monkeypatch.setattr(tokenizer, "decode", real_tokenizer.decode, raising=False)
    completion = SimpleNamespace(
        text=decoded,
        token_ids=token_ids,
        logprobs=None,
        finish_reason="stop",
        stop_reason="HALT" if string_stop else 200002,
    )
    monkeypatch.setattr(
        llm, "generate", lambda *args, **kwargs: [SimpleNamespace(outputs=[completion])]
    )
    task = get_task("ifeval", {"response_parser": "harmony"})
    instance = Instance(
        question="Avoid wrong.",
        metadata={
            "prompt": "Avoid wrong.",
            "instruction_id_list": ["keywords:forbidden_words"],
            "kwargs": [{"forbidden_words": ["wrong"]}],
        },
    )
    request = task.format_request(instance)
    params = replace(task.config.sampling_params, stop_sequences=("<|im_end|>", "HALT"))
    outputs = provider.generate([request], params)[0]
    assert outputs[0].text == decoded
    response = Response(instance=instance, request=request, outputs=outputs)
    tracker = TaskTracker(
        model_name="m",
        spec="ifeval",
        task=task,
        total_instances=1,
        completed_count=1,
        responses={0: response},
    )
    result = asyncio.run(finalize_task(tracker))
    assert result.error is None
    row = result.predictions[0]["model_output"][0]
    assert row["text"] == "good"
    assert row["original_text"] == decoded
    assert row["text_with_special_tokens"] == (raw.removesuffix("HALT") if string_stop else raw)
    assert row["response_parser_found"] is True
    assert outputs[0].metadata["ifeval"] == {"strict": [True], "loose": [True]}

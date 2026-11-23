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


@pytest.mark.parametrize(
    "specials", [("<|channel|>", "<|message|>", "<|end|>", "<|start|>", "<|return|>")]
)
@pytest.mark.parametrize("stop_case", ["eos", "string", "delimiter_substring", "spanning_channels"])
@pytest.mark.parametrize("special_registration", ["named", "decoder"])
def test_harmony_token_boundaries_reach_runner_predictions(
    fake_provider, monkeypatch, specials, stop_case, special_registration
) -> None:
    import asyncio
    from dataclasses import replace

    from tokenizers import AddedToken, Tokenizer, decoders, models
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
        done,
    ]
    stop_reason = {
        "eos": 200002,
        "string": "HALT",
        "delimiter_substring": "end",
        "spanning_channels": "wrongassistantfinalgood",
    }[stop_case]
    if stop_case == "spanning_channels":
        tokens.pop()
    elif stop_case != "eos":
        tokens[-1] = stop_reason
    raw = "".join(tokens)
    codec = Tokenizer(
        models.WordLevel(
            {
                word: i
                for i, word in enumerate(
                    ["analysis", "wrong", "assistant", "final", "good", "HALT", "end"]
                )
            }
        )
    )
    codec.decoder = decoders.Fuse()
    if special_registration == "decoder":
        codec.add_special_tokens([AddedToken(token, special=True) for token in specials])
        real_tokenizer = PreTrainedTokenizerFast(tokenizer_object=codec)
    else:
        real_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=codec, additional_special_tokens=list(specials)
        )
    token_ids = real_tokenizer.convert_tokens_to_ids(tokens)
    decoded = real_tokenizer.decode(token_ids, skip_special_tokens=True)
    if isinstance(stop_reason, str):
        decoded = decoded.split(stop_reason, 1)[0]
    assert decoded == (
        "analysis" if stop_case == "spanning_channels" else "analysiswrongassistantfinalgood"
    )
    assert real_tokenizer.decode(token_ids, skip_special_tokens=False) == raw
    monkeypatch.setattr(tokenizer, "get_added_vocab", real_tokenizer.get_added_vocab, raising=False)
    monkeypatch.setattr(tokenizer, "decode", real_tokenizer.decode, raising=False)
    monkeypatch.setattr(
        tokenizer,
        "all_special_tokens",
        real_tokenizer.all_special_tokens if special_registration == "named" else [],
        raising=False,
    )
    monkeypatch.setattr(
        tokenizer, "added_tokens_decoder", real_tokenizer.added_tokens_decoder, raising=False
    )
    completion = SimpleNamespace(
        text=decoded,
        token_ids=token_ids,
        logprobs=None,
        finish_reason="stop",
        stop_reason=stop_reason,
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
    params = replace(
        task.config.sampling_params,
        stop_sequences=("<|im_end|>", "HALT", "end", "wrongassistantfinalgood"),
    )
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
    assert row["text"] == ("" if stop_case == "spanning_channels" else "good")
    assert row["original_text"] == decoded
    expected_raw = {
        "eos": raw,
        "string": raw.removesuffix("HALT"),
        "delimiter_substring": raw.removesuffix("end"),
        "spanning_channels": channel + "analysis",
    }[stop_case]
    assert row["text_with_special_tokens"] == expected_raw
    assert row["response_parser_found"] is True
    passed = stop_case != "spanning_channels"
    assert outputs[0].metadata["ifeval"] == {"strict": [passed], "loose": [passed]}


def test_harmony_mapping_failure_only_errors_when_parsing_is_enabled(
    fake_provider, monkeypatch
) -> None:
    from olmo_eval.common.types import Instance, Response
    from olmo_eval.evals.tasks.common import get_task
    from olmo_eval.runners.io.builders import build_predictions

    provider, llm, tokenizer = fake_provider
    raw = "<|channel|>analysis<|message|>trace"
    monkeypatch.setattr(
        tokenizer, "get_added_vocab", lambda: {"<|channel|>": 200005}, raising=False
    )
    monkeypatch.setattr(
        tokenizer, "all_special_tokens", ["<|channel|>", "<|message|>"], raising=False
    )
    monkeypatch.setattr(tokenizer, "decode", lambda *args, **kwargs: raw, raising=False)
    completion = SimpleNamespace(
        text="unmatched visible output",
        token_ids=[200005],
        logprobs=None,
        finish_reason="stop",
        stop_reason="end",
    )
    monkeypatch.setattr(
        llm, "generate", lambda *args, **kwargs: [SimpleNamespace(outputs=[completion])]
    )
    request = LMRequest(request_type=RequestType.COMPLETION, prompt="Q")
    output = provider.generate([request])[0][0]
    response = Response(instance=Instance(question="Q"), request=request, outputs=[output])
    get_task("ifeval").strip_thinking_traces([response])
    assert output.text == completion.text
    with pytest.raises(ValueError, match="Cannot align Harmony"):
        get_task("ifeval", {"response_parser": "harmony"}).strip_thinking_traces([response])
    row = build_predictions([response])[0]["model_output"][0]
    assert row["text"] == completion.text
    assert row["text_with_special_tokens"] == raw
    assert "Cannot align Harmony" in row["response_parser_error"]
    assert not response.scores


class WordTokenizer(FakeTokenizer):
    """One token per whitespace-separated word, with character offsets; with a
    bos_token_id it adds BOS, and reads a BOS its chat template wrote as one token."""

    def __init__(self, bos_token_id: int | None = None) -> None:
        super().__init__(bos_token_id)
        self.pieces: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        super().encode(text, add_special_tokens)
        ids = []
        if add_special_tokens and self.bos_token_id is not None:
            ids.append(self.bos_token_id)
        if self.bos_token_id is not None and text.startswith(chr(self.bos_token_id)):
            ids.append(self.bos_token_id)
            text = text[1:]
        for piece in re.findall(r"\s*\S+\s*", text):
            if piece not in self.pieces:
                self.pieces.append(piece)
            ids.append(1000 + self.pieces.index(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.pieces[i - 1000] for i in ids if i >= 1000)

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False):  # noqa: ANN001, ANN204
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


CONTEXT = "one two three four five six seven eight"


def _cutting_provider(
    max_length: int, bos_token_id: int | None = None
) -> tuple[VLLMProvider, FakeLLM]:
    llm = FakeLLM(WordTokenizer(bos_token_id))
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._prefix_ids = default_prefix_ids(llm.tokenizer)
    provider._max_length = max_length
    provider._build_sampling_params = lambda params: "fake-sampling-params"
    return provider, llm


def _sent(llm: FakeLLM) -> list[str]:
    """The prompts the provider sent, decoded back to text."""
    return [llm.tokenizer.decode(p["prompt_token_ids"]) for p in llm.generate_calls[0]["prompts"]]


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

    assert _sent(llm) == [request.prompt]
    assert CONTEXT_CUT_TOKENS_KEY not in output.metadata


def test_the_overflow_is_cut_from_the_end_of_the_context() -> None:
    # 13 words against a budget of 10: the last three context words go, and the
    # instruction before the context and the question after it survive, as the
    # vllm_server provider cuts them.
    provider, llm = _cutting_provider(max_length=100)

    output = provider.generate([_completion(budget=10)], SamplingParams(max_tokens=1))[0][0]

    assert _sent(llm) == ["Read this. one two three four five  Question: q\nAnswer:"]
    assert output.metadata[CONTEXT_CUT_TOKENS_KEY] == 3


def test_the_window_caps_the_budget_after_the_generation() -> None:
    # a 20-token window with 8 tokens to generate leaves room for 12
    provider, llm = _cutting_provider(max_length=20)

    output = provider.generate([_completion(budget=100)], SamplingParams(max_tokens=8))[0][0]

    assert len(_sent(llm)[0].split()) == 12
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
    assert _sent(llm) == [
        "<chat>user:Read this. one two three four five six  Question: q<assistant>"
    ]
    assert output.metadata[CONTEXT_CUT_TOKENS_KEY] == 2


def test_the_bos_counts_against_the_budget() -> None:
    # 13 words and the BOS the tokenizer adds: 14 fits a budget of 14, and 13 cuts a word
    for budget, cut in ((14, 0), (13, 1)):
        provider, llm = _cutting_provider(max_length=100, bos_token_id=1)

        output = provider.generate([_completion(budget=budget)], SamplingParams(max_tokens=1))[0][0]

        sent = llm.generate_calls[0]["prompts"][0]["prompt_token_ids"]
        assert len(sent) == budget
        assert output.metadata.get(CONTEXT_CUT_TOKENS_KEY, 0) == cut


def test_a_bos_the_chat_template_writes_is_counted_once() -> None:
    # 12 rendered words and one BOS, as sent: the template's BOS is not doubled
    provider, llm = _cutting_provider(max_length=100, bos_token_id=1)
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": f"Read this. {CONTEXT} Question: q"},),
        context_fit=ContextFit(text=CONTEXT, max_prompt_tokens=13),
    )

    output = provider.generate([request], SamplingParams(max_tokens=1))[0][0]

    sent = llm.generate_calls[0]["prompts"][0]["prompt_token_ids"]
    assert sent[:2] != [1, 1]
    assert len(sent) == 13
    assert CONTEXT_CUT_TOKENS_KEY not in output.metadata


def test_text_that_is_not_in_the_prompt_is_refused() -> None:
    provider, _ = _cutting_provider(max_length=100)
    request = replace(
        _completion(budget=1), context_fit=ContextFit(text="missing", max_prompt_tokens=1)
    )

    with pytest.raises(ValueError, match="exactly once"):
        provider.generate([request], SamplingParams(max_tokens=1))


class ScoringTokenizer(FakeTokenizer):
    """One token per character (its code point), with BOS 1."""

    eos_token_id = 2

    def __init__(self) -> None:
        super().__init__(bos_token_id=1)


def _logprob(prev: int, tid: int) -> float:
    """A bigram logprob: depends on the token and the one before it."""
    return -(tid + 3 * prev) / 10_000


class ScoringLLM(FakeLLM):
    """Scores every input token after the first with a bigram logprob."""

    def generate(self, prompts, sampling_params, use_tqdm=False):  # noqa: ANN001, ANN201
        self.generate_calls.append({"prompts": prompts})
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            prompt_logprobs = [None] + [
                {tid: SimpleNamespace(logprob=_logprob(prev, tid), decoded_token=chr(tid))}
                for prev, tid in zip(ids, ids[1:], strict=False)
            ]
            outputs.append(SimpleNamespace(prompt_logprobs=prompt_logprobs))
        return outputs


def _full_context_logprob(text: str, bos: int = 1) -> float:
    ids = [bos] + [ord(c) for c in text]
    return sum(_logprob(prev, tid) for prev, tid in zip(ids, ids[1:], strict=False))


def _scoring_provider(monkeypatch, max_length: int) -> tuple[VLLMProvider, ScoringLLM]:  # noqa: ANN001
    fake_vllm = FakeVllmModule("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    llm = ScoringLLM(ScoringTokenizer())
    provider = VLLMProvider.__new__(VLLMProvider)
    provider.model_name = "test-model"
    provider.llm = llm
    provider._add_bos_token = None
    provider._max_length = max_length
    provider._prefix_ids = default_prefix_ids(llm.tokenizer)
    return provider, llm


TEXT = "the quick brown fox jumps over the lazy dog"


def test_a_rolling_request_scores_a_text_longer_than_the_context_in_full(monkeypatch) -> None:  # noqa: ANN001
    provider, llm = _scoring_provider(monkeypatch, max_length=9)
    request = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,), rolling=True)

    [[output]] = provider.logprobs([request])

    inputs = llm.generate_calls[0]["prompts"]
    assert len(inputs) > 1
    assert all(len(p["prompt_token_ids"]) <= 8 for p in inputs)  # max_length - 1
    assert [entry["token_id"] for entry in output.logprobs] == [ord(c) for c in TEXT]
    # Each window conditions its first scored token on the token before it, so with a
    # bigram model rolling scores match scoring the whole text in one context.
    assert output.metadata["sum_logits"] == pytest.approx(_full_context_logprob(TEXT))
    assert output.metadata["num_tokens"] == len(TEXT)
    assert output.metadata["num_windows"] == len(inputs)
    assert output.metadata["window_tokens"] == 8
    assert output.metadata["num_tokens_all"] == sum(len(p["prompt_token_ids"]) for p in inputs)
    assert output.metadata["prompt_truncated_tokens"] == 0
    assert output.text == TEXT


def test_a_rolling_request_that_fits_is_scored_like_an_ordinary_one(monkeypatch) -> None:  # noqa: ANN001
    provider, _ = _scoring_provider(monkeypatch, max_length=100)
    rolling = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,), rolling=True)
    plain = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,))

    [[rolled]], [[scored]] = provider.logprobs([rolling]), provider.logprobs([plain])

    assert rolled.logprobs == scored.logprobs
    assert rolled.metadata["num_windows"] == 1
    assert "num_windows" not in scored.metadata


def test_an_ordinary_request_is_still_left_truncated(monkeypatch) -> None:  # noqa: ANN001
    provider, _ = _scoring_provider(monkeypatch, max_length=9)
    request = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,))

    [[output]] = provider.logprobs([request])

    assert output.metadata["prompt_truncated_tokens"] == len(TEXT) + 1 - 8
    assert [entry["token_id"] for entry in output.logprobs] == [ord(c) for c in TEXT[-7:]]
    # The tokens actually sent, as before rolling requests existed.
    assert output.metadata["num_tokens_all"] == 8
    assert "num_windows" not in output.metadata and "window_tokens" not in output.metadata


def test_rolling_and_ordinary_requests_share_a_batch(monkeypatch) -> None:  # noqa: ANN001
    provider, _ = _scoring_provider(monkeypatch, max_length=9)
    requests = [
        LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,), rolling=True),
        LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=("ab", "abc")),
        LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT[:5],), rolling=True),
    ]

    outputs = provider.logprobs(requests)

    assert [len(o) for o in outputs] == [1, 2, 1]
    assert outputs[0][0].metadata["num_tokens"] == len(TEXT)
    assert [o.metadata["num_tokens"] for o in outputs[1]] == [2, 3]
    assert outputs[2][0].metadata["num_tokens"] == 5


def test_a_rolling_request_uses_its_own_shorter_max_length(monkeypatch) -> None:  # noqa: ANN001
    provider, llm = _scoring_provider(monkeypatch, max_length=100)
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,), rolling=True, max_length=9
    )

    [[output]] = provider.logprobs([request])

    assert all(len(p["prompt_token_ids"]) <= 8 for p in llm.generate_calls[0]["prompts"])
    assert output.metadata["window_tokens"] == 8
    assert output.metadata["sum_logits"] == pytest.approx(_full_context_logprob(TEXT))


def test_a_rolling_window_larger_than_the_engine_is_refused(monkeypatch) -> None:  # noqa: ANN001
    provider, _ = _scoring_provider(monkeypatch, max_length=9)
    request = LMRequest(
        request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,), rolling=True, max_length=64
    )

    with pytest.raises(ValueError, match="exceeds the engine's max_model_len"):
        provider.logprobs([request])


def test_an_empty_rolling_text_scores_no_tokens(monkeypatch) -> None:  # noqa: ANN001
    provider, _ = _scoring_provider(monkeypatch, max_length=9)
    request = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=("",), rolling=True)

    [[output]] = provider.logprobs([request])

    assert output.logprobs == []
    assert output.metadata["num_tokens"] == 0


def test_a_rolling_output_missing_a_token_logprob_raises(monkeypatch) -> None:  # noqa: ANN001
    provider, llm = _scoring_provider(monkeypatch, max_length=9)
    generate = llm.generate

    def drop_one(prompts, sampling_params, use_tqdm=False):  # noqa: ANN001, ANN202
        outputs = generate(prompts, sampling_params, use_tqdm)
        outputs[0].prompt_logprobs[2] = {}
        return outputs

    llm.generate = drop_one
    request = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(TEXT,), rolling=True)

    with pytest.raises(RuntimeError, match="scored .* of .* tokens"):
        provider.logprobs([request])

"""Tests for rolling loglikelihood requests: validation and provider refusal."""

from __future__ import annotations

import pytest

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams
from olmo_eval.harness import Harness, HarnessConfig
from olmo_eval.harness.harness import apply_harness_config
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.providers.mock import MockProvider
from olmo_eval.inference.providers.vllm import VLLMProvider


class NonRollingProvider(InferenceProvider):
    """A provider that would truncate a long text instead of rolling over it."""

    def generate(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        return [[LMOutput(text="")] for _ in requests]

    def logprobs(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        return [[LMOutput(text="")] for _ in requests]


def harness_with(provider: InferenceProvider) -> Harness:
    harness = Harness(HarnessConfig(name="test"))
    harness._provider = provider
    return harness


def rolling_request(text: str = "a long text") -> LMRequest:
    return LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=(text,), rolling=True)


def test_providers_declare_no_rolling_support_by_default() -> None:
    assert InferenceProvider.supports_rolling_logprobs is False
    assert VLLMProvider.supports_rolling_logprobs is True
    assert MockProvider.supports_rolling_logprobs is True


def test_a_rolling_request_is_refused_by_a_provider_that_would_truncate_it() -> None:
    with pytest.raises(ValueError, match="does not score rolling requests"):
        harness_with(NonRollingProvider("stub"))._apply_config(rolling_request())


def test_a_supporting_provider_gets_the_rolling_flag_through_the_harness() -> None:
    assert harness_with(MockProvider("mock"))._apply_config(rolling_request()).rolling is True
    assert apply_harness_config(HarnessConfig(name="test"), rolling_request()).rolling is True


def test_an_ordinary_loglikelihood_request_is_unaffected() -> None:
    request = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=("x",))
    assert harness_with(NonRollingProvider("stub"))._apply_config(request).rolling is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"request_type": RequestType.COMPLETION}, "loglikelihood requests"),
        ({"prompt": "context"}, "empty prompt"),
        ({"continuation_prompts": ("context",)}, "empty prompt"),
        ({"continuations": ("a", "b")}, "exactly one continuation"),
        ({"continuations": None}, "exactly one continuation"),
    ],
)
def test_a_rolling_request_must_score_one_text_from_an_empty_prompt(kwargs, message) -> None:  # noqa: ANN001
    fields = {"request_type": RequestType.LOGLIKELIHOOD, "continuations": ("text",)}
    with pytest.raises(ValueError, match=message):
        LMRequest(**{**fields, **kwargs}, rolling=True)


def test_the_refusal_names_the_provider_behind_the_metrics_wrapper() -> None:
    from olmo_eval.inference.metrics.core.collector import InstrumentedProvider

    harness = harness_with(InstrumentedProvider(NonRollingProvider("stub")))
    with pytest.raises(ValueError, match="^NonRollingProvider does not score rolling"):
        harness.logprobs([rolling_request()])


def test_the_metrics_wrapper_passes_rolling_support_through() -> None:
    from olmo_eval.inference.metrics.core.collector import InstrumentedProvider

    [[output]] = harness_with(InstrumentedProvider(MockProvider("mock"))).logprobs(
        [rolling_request("text")]
    )
    assert output.text == "text"


def test_saved_request_rows_record_rolling_only_when_set() -> None:
    from olmo_eval.common.types import Instance, Response
    from olmo_eval.runners.io.builders import build_requests_from_responses

    def row(request: LMRequest) -> dict:
        response = Response(
            instance=Instance(question="", metadata={"id": 0}),
            request=request,
            outputs=[LMOutput(text="t")],
        )
        return build_requests_from_responses([response], "task")[0]["request"]

    plain = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=("t",))
    assert row(rolling_request("t"))["rolling"] is True
    assert "rolling" not in row(plain)


def test_the_resume_hash_tells_rolling_from_ordinary_requests() -> None:
    from olmo_eval.common.types import Instance
    from olmo_eval.runners.asynq.progress import request_hash
    from olmo_eval.runners.asynq.types import QueueItem

    def item(request: LMRequest) -> QueueItem:
        instance = Instance(question="", gold_answer="t", metadata={"id": 0})
        return QueueItem(
            model_name="m", task_id="t", instance_idx=0, instance=instance, request=request
        )

    plain = LMRequest(request_type=RequestType.LOGLIKELIHOOD, continuations=("t",))
    assert request_hash(item(rolling_request("t"))) != request_hash(item(plain))

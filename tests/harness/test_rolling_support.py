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

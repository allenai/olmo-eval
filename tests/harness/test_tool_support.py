"""Tests for refusing requests whose tool schemas would be silently dropped."""

from __future__ import annotations

import asyncio

import pytest

from olmo_eval.common.types import LMOutput, LMRequest, RequestType, SamplingParams, ToolSchema
from olmo_eval.harness import Harness, HarnessConfig
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.inference.errors import ToolCallingUnsupportedError
from olmo_eval.inference.providers.vllm_server import VLLMServerProvider

TOOL = ToolSchema(name="lookup", description="Look something up.", parameters={})


class ToollessProvider(InferenceProvider):
    """A provider that would drop the tools it is handed."""

    def generate(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        return [[LMOutput(text="")] for _ in requests]

    def logprobs(
        self, requests: list[LMRequest], sampling_params: SamplingParams | None = None
    ) -> list[list[LMOutput]]:
        return [[LMOutput(text="")] for _ in requests]


class ToolCarryingProvider(ToollessProvider):
    supports_tools = True


def harness_with(provider: InferenceProvider) -> Harness:
    harness = Harness(HarnessConfig(name="test"))
    harness._provider = provider
    return harness


def tool_request() -> LMRequest:
    return LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Look up the weather."},),
        tools=(TOOL,),
    )


def test_a_provider_declares_no_tool_support_by_default() -> None:
    assert InferenceProvider.supports_tools is False


def test_a_request_with_tools_is_refused_by_a_provider_that_would_drop_them() -> None:
    harness = harness_with(ToollessProvider("stub"))

    with pytest.raises(ToolCallingUnsupportedError, match="does not send tool schemas"):
        harness._apply_config(tool_request())


def test_a_request_without_tools_is_unaffected() -> None:
    harness = harness_with(ToollessProvider("stub"))
    request = LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "Hello."},),
    )

    assert harness._apply_config(request).tools is None


def test_a_provider_that_carries_tools_passes_them_through() -> None:
    harness = harness_with(ToolCarryingProvider("stub"))

    assert harness._apply_config(tool_request()).tools == (TOOL,)


def test_the_vllm_server_provider_carries_tools() -> None:
    assert VLLMServerProvider.supports_tools is True


def unstarted_server_provider(tool_calls_parsed: bool | None) -> VLLMServerProvider:
    """A provider object without the server its constructor would start."""
    provider = object.__new__(VLLMServerProvider)
    provider._tool_calls_parsed = tool_calls_parsed
    return provider


def test_a_server_started_without_tool_choice_refuses_a_tool_request() -> None:
    provider = unstarted_server_provider(False)

    with pytest.raises(ToolCallingUnsupportedError, match="enable-auto-tool-choice"):
        asyncio.run(
            provider._generate_chat(
                client=None,  # ty: ignore[invalid-argument-type]
                request=tool_request(),
                params=SamplingParams(),
            )
        )


def test_an_external_server_is_given_the_benefit_of_the_doubt() -> None:
    # Whether an existing server parses tool calls is not knowable from here,
    # so the request goes out rather than being refused.
    provider = unstarted_server_provider(None)

    with pytest.raises(Exception) as caught:
        asyncio.run(
            provider._generate_chat(
                client=None,  # ty: ignore[invalid-argument-type]
                request=tool_request(),
                params=SamplingParams(),
            )
        )

    assert not isinstance(caught.value, ToolCallingUnsupportedError)

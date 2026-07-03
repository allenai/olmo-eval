"""Tests for Scaffold implementations."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds import (
    SCAFFOLD_REGISTRY,
    Scaffold,
    get_scaffold,
    list_scaffolds,
    register_scaffold,
)
from olmo_eval.harness.scaffolds.openai_agents import (
    TOOL_NOT_FOUND_TOOL_NAME,
    OpenAIAgentsScaffold,
    _create_tool_not_found_tool,
    _get_tool_call_correcting_model_class,
)
from olmo_eval.harness.tools import Tool


def _sdk_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _function_call(name: str, arguments: str = "{}"):
    from openai.types.responses import ResponseFunctionToolCall

    return ResponseFunctionToolCall(
        arguments=arguments,
        call_id=f"call_{name}",
        name=name,
        type="function_call",
    )


def _model_response(output):
    from agents.items import ModelResponse
    from agents.usage import Usage

    return ModelResponse(output=output, usage=Usage(), response_id=None)


def _correcting_model():
    from openai import AsyncOpenAI

    model_class = _get_tool_call_correcting_model_class()
    return model_class(
        openai_client=AsyncOpenAI(api_key="test", base_url="http://example.com/v1"),
        model="test-model",
    )


async def _call_correcting_model(model, tools):
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing

    return await model.get_response(
        system_instructions=None,
        input="input",
        model_settings=ModelSettings(),
        tools=tools,
        output_schema=None,
        handoffs=[],
        tracing=ModelTracing.DISABLED,
    )


class TestGetScaffold:
    """Tests for get_scaffold function."""

    def test_get_unknown_scaffold(self):
        """Test getting an unknown scaffold raises error."""
        with pytest.raises(ValueError, match="Unknown scaffold"):
            get_scaffold("nonexistent")

    def test_register_custom_scaffold(self):
        """Test registering a custom scaffold using the decorator."""

        @register_scaffold("custom")
        class CustomScaffold(Scaffold):
            async def run(
                self,
                provider,
                config,
                request,
                sampling_params=None,
                trace_metadata=None,
                **kwargs,
            ):
                pass

        assert "custom" in SCAFFOLD_REGISTRY
        assert "custom" in list_scaffolds()

        # Clean up
        del SCAFFOLD_REGISTRY["custom"]


class TestOpenAIAgentsToolCorrection:
    @pytest.mark.anyio
    async def test_fallback_tool_execute_mentions_requested_and_available_tools(self):
        from agents import function_tool
        from agents.tool_context import ToolContext

        fallback_tool = _create_tool_not_found_tool(function_tool, ["search", "fetch_page"])
        payload = json.dumps({"requested_tool": "web_lookup", "arguments": '{"query":"olmo"}'})
        ctx = ToolContext(
            context=None,
            tool_name=TOOL_NOT_FOUND_TOOL_NAME,
            tool_call_id="call_1",
            tool_arguments=payload,
        )

        message = await fallback_tool.on_invoke_tool(ctx, payload)

        assert "web_lookup" in message
        assert "search" in message
        assert "fetch_page" in message

        fallback_tool = _create_tool_not_found_tool(function_tool, [])
        message = await fallback_tool.on_invoke_tool(ctx, payload)

        assert message == "Error: tool 'web_lookup' does not exist. No tools are available."

    @pytest.mark.anyio
    async def test_get_response_rewrites_unknown_tools_and_hides_fallback(self, monkeypatch):
        from agents import OpenAIChatCompletionsModel

        known_call = _function_call("search", '{"query":"known"}')
        unknown_call = _function_call("web_lookup", '{"query":"unknown"}')
        other_unknown_call = _function_call("fetch_url", '{"url":"https://example.com"}')
        unknown_call_id = unknown_call.call_id
        other_unknown_call_id = other_unknown_call.call_id
        captured = {}

        async def fake_get_response(
            self,
            *,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        ):
            captured["tools"] = tools
            return _model_response([known_call, unknown_call, other_unknown_call])

        monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", fake_get_response)

        response = await _call_correcting_model(
            _correcting_model(),
            [_sdk_tool("search"), _sdk_tool(TOOL_NOT_FOUND_TOOL_NAME)],
        )

        assert response.output[0] is known_call
        assert known_call.name == "search"
        assert known_call.arguments == '{"query":"known"}'
        assert unknown_call.name == TOOL_NOT_FOUND_TOOL_NAME
        assert unknown_call.call_id == unknown_call_id
        assert json.loads(unknown_call.arguments) == {
            "requested_tool": "web_lookup",
            "arguments": '{"query":"unknown"}',
        }
        assert other_unknown_call.name == TOOL_NOT_FOUND_TOOL_NAME
        assert other_unknown_call.call_id == other_unknown_call_id
        assert json.loads(other_unknown_call.arguments) == {
            "requested_tool": "fetch_url",
            "arguments": '{"url":"https://example.com"}',
        }
        assert [tool.name for tool in captured["tools"]] == ["search"]

    @pytest.mark.anyio
    async def test_get_response_forwards_sdk_kwargs(self, monkeypatch):
        from agents import OpenAIChatCompletionsModel
        from agents.model_settings import ModelSettings
        from agents.models.interface import ModelTracing

        captured = {}

        async def fake_get_response(
            self,
            *,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=None,
            conversation_id=None,
            prompt=None,
            **kwargs,
        ):
            captured["previous_response_id"] = previous_response_id
            captured["conversation_id"] = conversation_id
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return _model_response([])

        monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", fake_get_response)
        prompt_sentinel = {"id": "prompt-sentinel"}

        await _correcting_model().get_response(
            system_instructions=None,
            input="input",
            model_settings=ModelSettings(),
            tools=[_sdk_tool("search"), _sdk_tool(TOOL_NOT_FOUND_TOOL_NAME)],
            output_schema=None,
            handoffs=[],
            tracing=ModelTracing.DISABLED,
            previous_response_id="resp-sentinel",
            conversation_id="conv-sentinel",
            prompt=prompt_sentinel,
        )

        assert captured["previous_response_id"] == "resp-sentinel"
        assert captured["conversation_id"] == "conv-sentinel"
        assert captured["prompt"] is prompt_sentinel
        assert captured["kwargs"] == {}

    def test_create_agent_wires_fallback_tool_and_correcting_model(self):
        from openai import AsyncOpenAI

        async def search(query: str) -> str:
            return query

        class ProviderStub:
            model_name = "test-model"

            def get_openai_client(self):
                return AsyncOpenAI(api_key="test", base_url="http://example.com/v1")

        config = HarnessConfig(
            name="test",
            tools=(
                Tool.from_function(
                    search,
                    name="search",
                    description="Search for information.",
                ),
            ),
        )

        agent = OpenAIAgentsScaffold()._create_agent(ProviderStub(), config)

        tool_names = [tool.name for tool in agent.tools]
        assert tool_names.count(TOOL_NOT_FOUND_TOOL_NAME) == 1
        assert "search" in tool_names
        assert isinstance(agent.model, _get_tool_call_correcting_model_class())

    def test_create_agent_rejects_reserved_tool_name(self):
        from openai import AsyncOpenAI

        async def tool_not_found(query: str) -> str:
            return query

        class ProviderStub:
            model_name = "test-model"

            def get_openai_client(self):
                return AsyncOpenAI(api_key="test", base_url="http://example.com/v1")

        config = HarnessConfig(
            name="test",
            tools=(
                Tool.from_function(
                    tool_not_found,
                    name=TOOL_NOT_FOUND_TOOL_NAME,
                    description="Conflicting tool name.",
                ),
            ),
        )

        with pytest.raises(
            ValueError,
            match=f"Tool name {TOOL_NOT_FOUND_TOOL_NAME!r} is reserved by the scaffold",
        ):
            OpenAIAgentsScaffold()._create_agent(ProviderStub(), config)

    @pytest.mark.anyio
    async def test_get_response_logs_warning_only_for_unknown_tools(self, monkeypatch, caplog):
        from agents import OpenAIChatCompletionsModel

        responses = [
            _model_response([_function_call("web_lookup")]),
            _model_response([_function_call("search")]),
        ]

        async def fake_get_response(
            self,
            *,
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            **kwargs,
        ):
            return responses.pop(0)

        monkeypatch.setattr(OpenAIChatCompletionsModel, "get_response", fake_get_response)
        caplog.set_level(logging.WARNING, logger="olmo_eval.harness.scaffolds.openai_agents")
        tools = [_sdk_tool("search"), _sdk_tool(TOOL_NOT_FOUND_TOOL_NAME)]

        await _call_correcting_model(_correcting_model(), tools)

        assert "web_lookup" in caplog.text
        assert TOOL_NOT_FOUND_TOOL_NAME in caplog.text

        caplog.clear()
        await _call_correcting_model(_correcting_model(), tools)

        assert "web_lookup" not in caplog.text
        assert "unknown tool" not in caplog.text

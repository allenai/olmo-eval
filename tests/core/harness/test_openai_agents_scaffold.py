"""Tests for the OpenAI Agents scaffold (requires the optional agents/openai deps)."""

import pytest

pytest.importorskip("agents")
pytest.importorskip("openai")

import contextlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from olmo_eval.common.types import LMRequest, RequestType  # noqa: E402
from olmo_eval.harness.config import HarnessConfig  # noqa: E402
from olmo_eval.harness.scaffolds.openai_agents import (  # noqa: E402
    FORCED_FINAL_ANSWER_INSTRUCTION,
    TOOL_NOT_FOUND_TOOL_NAME,
    OpenAIAgentsScaffold,
    _create_tool_not_found_tool,
    _get_tool_call_correcting_model_class,
)
from olmo_eval.harness.tools import Tool  # noqa: E402


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


def _tool_turn_items(agent, output: str = "Search result text"):
    from agents.items import ToolCallItem, ToolCallOutputItem

    call = _function_call("search", '{"query":"olmo"}')
    return [
        ToolCallItem(agent=agent, raw_item=call),
        ToolCallOutputItem(
            agent=agent,
            raw_item={
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            },
            output=output,
        ),
    ]


class _FakeRunData:
    def __init__(self, *, new_items, final_output=None):
        self.new_items = new_items
        self.final_output = final_output


def _max_turns_exceeded_with_run_data(run_data, message: str = "Max turns (1) exceeded"):
    from agents.exceptions import MaxTurnsExceeded

    exc = MaxTurnsExceeded(message)
    exc.run_data = run_data
    return exc


def _agent_request():
    return LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": "What did you find?"},),
    )


def _patch_scaffold_agent(monkeypatch, agent):
    import agents

    monkeypatch.setattr(
        OpenAIAgentsScaffold,
        "_get_or_create_agent",
        lambda self, provider, config, sandbox_manager=None: agent,
    )
    monkeypatch.setattr(agents, "trace", lambda *args, **kwargs: contextlib.nullcontext())


class TestOpenAIAgentsMaxTurns:
    @pytest.mark.anyio
    async def test_max_turns_preserves_trajectory_and_forces_final_answer(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        partial_run_data = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Tool result before cap"),
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                raise _max_turns_exceeded_with_run_data(partial_run_data)
            return SimpleNamespace(final_output="Forced final answer")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=1),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.max_turns_reached is True
        assert result.error is None
        assert result.final_output.text == "Forced final answer"
        assert result.trajectory is not None
        assert result.trajectory.total_tool_calls == 1
        assert result.trajectory.tool_result_sequence[0].content == "Tool result before cap"
        assert result.trajectory.tool_result_sequence[0].tool_call_id == "call_search"

        assert len(run_calls) == 2
        assert run_calls[0]["max_turns"] == 1
        final_call = run_calls[1]
        assert final_call["max_turns"] == 1
        assert final_call["starting_agent"].tools == []
        assert final_call["starting_agent"].handoffs == []
        assert final_call["starting_agent"].mcp_servers == []
        assert final_call["starting_agent"].model_settings.tool_choice == "none"
        assert final_call["input"][-1] == {
            "role": "user",
            "content": FORCED_FINAL_ANSWER_INSTRUCTION,
        }

    @pytest.mark.anyio
    async def test_forced_final_answer_failure_falls_back_to_old_result(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        partial_run_data = _FakeRunData(
            new_items=_tool_turn_items(agent),
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                raise _max_turns_exceeded_with_run_data(partial_run_data)
            raise RuntimeError("connection failed")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=1),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.max_turns_reached is True
        assert result.final_output.text == "[Max turns exceeded]"
        assert result.trajectory is not None
        assert result.trajectory.turns == ()
        assert result.error == "Max turns (1) exceeded"
        assert len(run_calls) == 2

    @pytest.mark.anyio
    async def test_normal_completion_does_not_force_final_answer(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_result = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Normal tool result"),
            final_output="Normal final answer",
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            return run_result

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.max_turns_reached is False
        assert result.error is None
        assert result.final_output.text == "Normal final answer"
        assert result.trajectory is not None
        assert result.trajectory.total_tool_calls == 1
        assert result.trajectory.tool_result_sequence[0].content == "Normal tool result"
        assert len(run_calls) == 1


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


class TestOpenAIAgentsToolCorrection:
    def test_get_response_forwarded_parameters_exist_on_sdk_parent(self):
        """Forwarded keyword names must exist upstream or super() raises TypeError at runtime."""
        import inspect

        from agents import OpenAIChatCompletionsModel

        sdk_parameters = inspect.signature(OpenAIChatCompletionsModel.get_response).parameters
        override_parameters = inspect.signature(
            _get_tool_call_correcting_model_class().get_response
        ).parameters
        sdk_has_var_keyword = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in sdk_parameters.values()
        )
        missing_parameters = [
            name
            for name, parameter in override_parameters.items()
            if name != "self"
            and parameter.kind is not inspect.Parameter.VAR_KEYWORD
            and name not in sdk_parameters
            and not sdk_has_var_keyword
        ]

        assert missing_parameters == []

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


def _chat_message(*, content=None, tool_calls=None):
    from openai.types.chat import ChatCompletionMessage

    return ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)


def _chat_tool_call(call_id: str, name: str, arguments: str = "{}"):
    from openai.types.chat import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_function_tool_call import Function

    return ChatCompletionMessageFunctionToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def _chat_completion(message):
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice

    finish_reason = "tool_calls" if message.tool_calls else "stop"
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(finish_reason=finish_reason, index=0, message=message)],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _calls_tool(name: str, arguments: str = "{}"):
    return _chat_message(tool_calls=[_chat_tool_call(f"call_{name}", name, arguments)])


class _ScriptedClient:
    """A real AsyncOpenAI whose completions are scripted, recording what the model was sent."""

    def __init__(self, script):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key="test", base_url="http://example.com/v1")
        self.script = list(script)
        self.sent_messages = []
        self.sent_tool_names = []

        async def create(**kwargs):
            self.sent_messages.append(kwargs.get("messages"))
            tools = kwargs.get("tools") or []
            self.sent_tool_names.append(
                [t.get("function", {}).get("name") for t in tools if isinstance(t, dict)]
            )
            message = self.script.pop(0) if self.script else _chat_message(content="fallback")
            return _chat_completion(message)

        self.client.chat.completions.create = create  # type: ignore[method-assign]

    def tool_results_seen(self):
        """Every tool result the model was handed, read off the last request's history."""
        results = []
        for message in (self.sent_messages[-1] if self.sent_messages else []) or []:
            if isinstance(message, dict) and message.get("role") == "tool":
                results.append(str(message.get("content", "")))
        return results


def _search_config(**kwargs):
    async def search(query: str) -> str:
        return f"results for {query}"

    return HarnessConfig(
        name="test",
        tools=(Tool.from_function(search, name="search", description="Search for information."),),
        **kwargs,
    )


async def _run_scaffold(scripted, config, *, scaffold=None, agent=None):
    class ProviderStub:
        model_name = "test-model"

        def get_openai_client(self):
            return scripted.client

    scaffold = scaffold or OpenAIAgentsScaffold()
    if agent is not None:
        scaffold._get_or_create_agent = (  # type: ignore[method-assign]
            lambda provider, config, sandbox_manager=None: agent
        )
    return await scaffold.run(
        provider=ProviderStub(),
        config=config,
        request=_agent_request(),
        enable_compaction=False,
    )


class TestUnknownToolCallEndToEnd:
    """Drive the real SDK dispatch path, not a stubbed Runner.

    These exercise agents' actual tool dispatch, which is where an unknown tool name used to abort
    the run and replace the answer with an error string.
    """

    @pytest.mark.anyio
    async def test_unknown_tool_call_recovers_and_answer_survives(self):
        scripted = _ScriptedClient(
            [
                _calls_tool("thought", '{"text":"let me think"}'),
                _calls_tool("search", '{"query":"olmo"}'),
                _chat_message(content="The answer survived."),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=6))

        assert result.final_output.text == "The answer survived."
        assert result.error is None
        assert result.max_turns_reached is False

        tool_results = scripted.tool_results_seen()
        assert any("thought" in r and "does not exist" in r for r in tool_results)
        assert any("Available tools: search" in r for r in tool_results)
        assert "results for olmo" in tool_results

        # The fallback tool must never be advertised to the model.
        for advertised in scripted.sent_tool_names:
            assert TOOL_NOT_FOUND_TOOL_NAME not in advertised
            assert "search" in advertised

        assert result.trajectory.total_tool_calls == 2

    @pytest.mark.anyio
    async def test_unknown_tool_call_without_the_fix_destroys_the_answer(self):
        """The failure this change exists to prevent, pinned against the SDK's own behavior."""
        from agents import Agent, OpenAIChatCompletionsModel

        scripted = _ScriptedClient(
            [
                _calls_tool("thought", '{"text":"let me think"}'),
                _chat_message(content="Never reached."),
            ]
        )
        # An agent built without the correcting model or the fallback tool.
        plain_agent = Agent(
            name="openai_agents",
            instructions="",
            model=OpenAIChatCompletionsModel(openai_client=scripted.client, model="test-model"),
            tools=[],
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=6), agent=plain_agent)

        assert result.final_output.text == (
            "[Tool error: Tool thought not found in agent openai_agents]"
        )
        assert result.error == "Tool thought not found in agent openai_agents"

    @pytest.mark.anyio
    async def test_repeated_unknown_tool_calls_still_finish(self):
        scripted = _ScriptedClient(
            [
                _calls_tool("thought"),
                _calls_tool("thought"),
                _calls_tool("ponder"),
                _chat_message(content="Answer after three bad calls."),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=8))

        assert result.final_output.text == "Answer after three bad calls."
        assert result.error is None
        tool_results = scripted.tool_results_seen()
        assert len([r for r in tool_results if "does not exist" in r]) == 3
        assert any("ponder" in r for r in tool_results)

    @pytest.mark.anyio
    async def test_unknown_tool_calls_consume_the_turn_budget(self):
        """Endless unknown calls must hit the existing max-turns path, not spin forever."""
        scripted = _ScriptedClient([_calls_tool("thought") for _ in range(10)])

        result = await _run_scaffold(scripted, _search_config(max_turns=3))

        assert result.max_turns_reached is True
        assert len(scripted.sent_messages) <= 10

    @pytest.mark.anyio
    async def test_normal_run_is_unaffected(self):
        scripted = _ScriptedClient(
            [
                _calls_tool("search", '{"query":"olmo"}'),
                _chat_message(content="Normal answer."),
            ]
        )

        result = await _run_scaffold(scripted, _search_config(max_turns=6))

        assert result.final_output.text == "Normal answer."
        assert result.error is None
        assert scripted.tool_results_seen() == ["results for olmo"]
        assert result.trajectory.total_tool_calls == 1

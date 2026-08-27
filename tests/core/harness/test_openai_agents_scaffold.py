"""Tests for the OpenAI Agents scaffold (requires the optional agents deps)."""

import contextlib
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds.openai_agents import (
    FORCED_FINAL_ANSWER_INSTRUCTION,
    OpenAIAgentsScaffold,
    _make_tool_error_formatter,
)

pytest.importorskip("agents")


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


def _named_tool(name: str):
    from olmo_eval.harness.tools import Tool

    async def _execute(query: str = "") -> str:
        return "ok"

    return Tool(
        name=name,
        description=f"{name} description",
        execute=_execute,
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


class TestToolErrorFormatter:
    def test_message_names_the_bad_tool_and_lists_the_valid_ones(self):
        formatter = _make_tool_error_formatter(["paper_search", "snippet_search"])

        message = formatter(
            SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        )

        assert message == (
            "Error: tool 'web_search' does not exist. "
            "Available tools: paper_search, snippet_search. Call one of these exact names."
        )

    def test_defers_to_sdk_default_for_other_error_kinds(self):
        formatter = _make_tool_error_formatter(["paper_search"])

        rejected = formatter(
            SimpleNamespace(kind="approval_rejected", tool_name="paper_search", call_id="c1")
        )

        assert rejected is None

    def test_handles_an_empty_tool_inventory(self):
        formatter = _make_tool_error_formatter([])

        message = formatter(
            SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        )

        assert message == "Error: tool 'web_search' does not exist. No tools are available."


class TestNativeUnknownToolRecovery:
    @pytest.mark.anyio
    async def test_run_opts_into_native_recovery_with_the_real_tool_names(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_calls = []

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            return SimpleNamespace(final_output="done", new_items=[])

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(
                name="test",
                max_turns=3,
                tools=(_named_tool("paper_search"), _named_tool("snippet_search")),
            ),
            request=_agent_request(),
            enable_compaction=False,
        )

        run_config = run_calls[0]["run_config"]
        assert run_config.tool_not_found_behavior == "return_error_to_model"

        message = run_config.tool_error_formatter(
            SimpleNamespace(kind="tool_not_found", tool_name="web_search", call_id="c1")
        )
        assert "paper_search" in message
        assert "snippet_search" in message

    @pytest.mark.anyio
    async def test_unknown_tool_recovers_against_the_real_sdk_dispatch(self):
        """Pin the opt-in against the SDK's own dispatch, not a mock of it.

        Scripts a chat-completions endpoint that calls a tool that was never registered,
        then answers normally, and asserts the SDK fed our message back as a tool result
        instead of raising ModelBehaviorError.
        """
        import json

        import httpx
        from agents import (
            Agent,
            OpenAIChatCompletionsModel,
            RunConfig,
            Runner,
            function_tool,
        )
        from openai import AsyncOpenAI

        requests: list[dict] = []

        def completion(message: dict, finish: str) -> dict:
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(json.loads(request.content.decode()))
            if len(requests) == 1:
                return httpx.Response(
                    200,
                    json=completion(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_bogus",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": '{"query":"olmo"}',
                                    },
                                }
                            ],
                        },
                        "tool_calls",
                    ),
                )
            return httpx.Response(
                200, json=completion({"role": "assistant", "content": "recovered"}, "stop")
            )

        @function_tool(strict_mode=False)
        async def paper_search(query: str) -> str:
            """Search papers."""
            return "results"

        client = AsyncOpenAI(
            api_key="test-key-not-real",
            base_url="http://test.invalid/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        agent = Agent(
            name="test-agent",
            instructions="Use tools.",
            model=OpenAIChatCompletionsModel(openai_client=client, model="test-model"),
            tools=[paper_search],
        )

        result = await Runner.run(
            starting_agent=agent,
            input="find something",
            max_turns=5,
            run_config=RunConfig(
                tool_not_found_behavior="return_error_to_model",
                tool_error_formatter=_make_tool_error_formatter(["paper_search"]),
            ),
        )

        assert result.final_output == "recovered"
        tool_messages = [m for m in requests[1]["messages"] if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["content"] == (
            "Error: tool 'web_search' does not exist. "
            "Available tools: paper_search. Call one of these exact names."
        )


class TestModelRefusal:
    @pytest.mark.anyio
    async def test_refusal_is_reported_instead_of_taking_down_the_run(self, monkeypatch):
        from agents import Agent, Runner
        from agents.exceptions import ModelRefusalError

        agent = Agent(name="test-agent", instructions="Use tools.")

        async def fake_run(**kwargs):
            raise ModelRefusalError("I cannot help with that.")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert "I cannot help with that." in result.final_output.text
        assert result.error is not None

"""Tests for the OpenAI Agents scaffold (requires the optional agents deps)."""

import contextlib
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds.openai_agents import (
    FORCED_FINAL_ANSWER_INSTRUCTION,
    OpenAIAgentsScaffold,
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


class TestOpenAIAgentsModelApi:
    @pytest.mark.parametrize(
        ("agent_api", "expected_model_type"),
        [
            ("chat_completions", "OpenAIChatCompletionsModel"),
            ("responses", "OpenAIResponsesModel"),
        ],
    )
    def test_create_agent_uses_provider_api(self, agent_api, expected_model_type):
        provider = SimpleNamespace(
            agent_api=agent_api,
            model_name="test-model",
            get_openai_client=lambda: SimpleNamespace(base_url="http://localhost/v1"),
        )

        agent = OpenAIAgentsScaffold()._create_agent(
            provider=provider,
            config=HarnessConfig(name="test"),
        )

        assert type(agent.model).__name__ == expected_model_type

    def test_create_agent_defaults_to_chat_completions(self):
        provider = SimpleNamespace(
            model_name="test-model",
            get_openai_client=lambda: SimpleNamespace(base_url="http://localhost/v1"),
        )

        agent = OpenAIAgentsScaffold()._create_agent(
            provider=provider,
            config=HarnessConfig(name="test"),
        )

        assert type(agent.model).__name__ == "OpenAIChatCompletionsModel"

    def test_create_agent_rejects_unknown_api(self):
        provider = SimpleNamespace(
            agent_api="completions",
            model_name="test-model",
            get_openai_client=lambda: SimpleNamespace(base_url="http://localhost/v1"),
        )

        with pytest.raises(ValueError, match="provider.agent_api must be one of"):
            OpenAIAgentsScaffold()._create_agent(
                provider=provider,
                config=HarnessConfig(name="test"),
            )


class TestOpenAIAgentsMaxTurns:
    @pytest.mark.parametrize(
        ("agent_api", "model_name", "expected_tool_choice", "expected_reasoning_effort"),
        [
            ("chat_completions", "test-model", "none", None),
            ("responses", "openai/gpt-oss-20b", None, "low"),
        ],
    )
    @pytest.mark.anyio
    async def test_max_turns_preserves_trajectory_and_forces_final_answer(
        self,
        monkeypatch,
        agent_api,
        model_name,
        expected_tool_choice,
        expected_reasoning_effort,
    ):
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
            provider=SimpleNamespace(agent_api=agent_api, model_name=model_name),
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
        assert final_call["starting_agent"].model_settings.tool_choice == expected_tool_choice
        reasoning = final_call["starting_agent"].model_settings.reasoning
        assert (reasoning.effort if reasoning is not None else None) == expected_reasoning_effort
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

    @pytest.mark.anyio
    async def test_empty_gpt_oss_response_forces_low_effort_final_answer(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        partial_result = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Evidence before empty completion"),
            final_output="",
        )
        run_calls = []

        def fail_run_streamed(**kwargs):
            raise AssertionError("Runner.run_streamed should not be used")

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                return partial_result
            return SimpleNamespace(final_output="Recovered final answer")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run_streamed", staticmethod(fail_run_streamed))
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(agent_api="responses", model_name="openai/gpt-oss-20b"),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.error is None
        assert result.final_output.text == "Recovered final answer"
        assert result.trajectory is not None
        assert result.trajectory.total_tool_calls == 1
        assert len(run_calls) == 2
        final_agent = run_calls[1]["starting_agent"]
        assert final_agent.tools == []
        assert final_agent.model_settings.tool_choice is None
        assert final_agent.model_settings.reasoning.effort == "low"

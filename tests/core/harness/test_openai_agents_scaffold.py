"""Tests for the OpenAI Agents scaffold (requires the optional agents deps)."""

import contextlib
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import LMRequest, RequestType
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds.openai_agents import (
    FORCED_FINAL_ANSWER_INSTRUCTION,
    OpenAIAgentsScaffold,
    strip_reasoning_prefix,
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


class TestStripReasoningPrefix:
    """Tests for the scaffold's reasoning-leak guard."""

    def test_plain_answer_is_untouched(self):
        assert strip_reasoning_prefix("# Report\n\nBody") == "# Report\n\nBody"
        assert strip_reasoning_prefix("") == ""

    def test_unopened_close_tag_is_stripped(self):
        """This is the shape a thinking template produces: only the closing tag.

        The opening <think> lives in the generation prompt, so a paired
        <think>...</think> pattern would not match what the model returns.
        """
        leaked = (
            "I have gathered enough sources. Now I will write the report."
            "</think> # Comprehensive Analysis\n\nBody"
        )
        assert strip_reasoning_prefix(leaked) == "# Comprehensive Analysis\n\nBody"

    def test_paired_block_is_stripped_too(self):
        assert strip_reasoning_prefix("<think>scratch</think>\n\nAnswer") == "Answer"

    def test_cuts_at_the_first_close_tag(self):
        """Matches the ResearchQA and DeepResearch Bench extractors."""
        assert strip_reasoning_prefix("a</think>b</think>c") == "b</think>c"


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

    @pytest.mark.anyio
    async def test_forced_final_answer_drops_a_leaked_monologue(self, monkeypatch):
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        partial_run_data = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Tool result before cap"),
        )
        run_calls = []

        async def fake_run(**kwargs):
            run_calls.append(kwargs)
            if len(run_calls) == 1:
                raise _max_turns_exceeded_with_run_data(partial_run_data)
            return SimpleNamespace(final_output="wrapping up now</think>Forced final answer")

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=1),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.final_output.text == "Forced final answer"
        assert result.trajectory.tool_result_sequence[0].content == "Tool result before cap"
        assert result.trajectory.tool_result_sequence[0].tool_call_id == "call_search"

        assert len(run_calls) == 2
        assert run_calls[0]["max_turns"] == 1
        final_call = run_calls[1]
        assert final_call["max_turns"] == 1
        assert final_call["starting_agent"].tools == []
        assert final_call["starting_agent"].handoffs == []
        assert final_call["starting_agent"].mcp_servers == []
        # `Send no tool_choice on the closing call` deliberately stopped setting this:
        # tools=[] paired with tool_choice="none" is an HTTP 400 on OpenAI, and it failed
        # quietly -- the instance kept its partial answer and counted as a success. The
        # assertion was left on the old contract, so this test has been red ever since.
        assert final_call["starting_agent"].model_settings.tool_choice is None
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
    async def test_normal_completion_drops_a_leaked_monologue(self, monkeypatch):
        """A model served without a reasoning parser returns monologue + answer.

        The scaffold must store only the answer; the raw text stays in the
        trajectory so the leak is still visible when debugging a run.
        """
        from agents import Agent, Runner

        agent = Agent(name="test-agent", instructions="Use tools.")
        run_result = _FakeRunData(
            new_items=_tool_turn_items(agent, output="Normal tool result"),
            final_output=(
                "I have gathered the sources and will now write the report.</think># Report\n\nBody"
            ),
        )

        async def fake_run(**kwargs):
            return run_result

        _patch_scaffold_agent(monkeypatch, agent)
        monkeypatch.setattr(Runner, "run", staticmethod(fake_run))

        result = await OpenAIAgentsScaffold().run(
            provider=SimpleNamespace(),
            config=HarnessConfig(name="test", max_turns=3),
            request=_agent_request(),
            enable_compaction=False,
        )

        assert result.final_output.text == "# Report\n\nBody"

class TestRecoverAnswerFromReasoning:
    """The answer a reasoning model wrote into `reasoning` instead of `content`.

    The shape below is copied from a real run in the dev matrix --
    `litsearch__single` case 10, which issued 4 tool calls and produced 5 generations, and whose
    last message is `{"content": null, "tool_calls": [], "reasoning": "Based on the search
    results..."}`. The Agents SDK reads `content`, so the run scored as a non-answer. 8 of 50
    litsearch runs and 9 of 50 researchqa runs looked like that.

    This is the mirror of `strip_reasoning_prefix`, which handles the same split failing the
    other way.
    """

    class _Item:
        def __init__(self, raw):
            self.raw_item = raw

    class _Result:
        def __init__(self, items, final_output=""):
            self.new_items = items
            self.final_output = final_output

    def test_answer_in_reasoning_is_recovered(self):
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        result = self._Result([
            self._Item({"role": "assistant", "content": None, "tool_calls": [],
                        "reasoning": "Based on the search results, the paper is X."}),
        ])
        assert recover_answer_from_reasoning(result) == "Based on the search results, the paper is X."

    def test_deepseek_style_reasoning_content_is_also_read(self):
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        result = self._Result([
            self._Item({"role": "assistant", "content": "", "reasoning_content": "The answer."}),
        ])
        assert recover_answer_from_reasoning(result) == "The answer."

    def test_a_real_answer_in_content_is_not_overridden(self):
        """A run that answered normally must be untouched, reasoning present or not."""
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        result = self._Result([
            self._Item({"role": "assistant", "content": "The paper is X.",
                        "reasoning": "let me think about this"}),
        ])
        assert recover_answer_from_reasoning(result) == ""

    def test_nothing_anywhere_returns_empty(self):
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        assert recover_answer_from_reasoning(self._Result([])) == ""
        assert recover_answer_from_reasoning(
            self._Result([self._Item({"role": "assistant", "content": None})])) == ""

class TestReasoningFieldAlias:
    """vLLM emits `reasoning`; the Agents SDK reads `reasoning_content`.

    `chatcmpl_converter.message_to_output_items` gates on
    `hasattr(message, "reasoning_content")`. One suffix apart, so the SDK builds no reasoning item
    and the answer is discarded before `new_items` exists. Measured on litsearch__single: 11 of 11
    empty runs had 2,105 characters of answer sitting in `reasoning`.

    The first fix for this searched `new_items` and found nothing, because by then the field was
    already gone. These tests cover the alias at the client boundary, which is the last point at
    which it still exists.
    """

    class _Msg:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _Choice:
        def __init__(self, message):
            self.message = message

    class _Completion:
        def __init__(self, choices):
            self.choices = choices

    def test_reasoning_is_aliased_onto_reasoning_content(self):
        from olmo_eval.harness.scaffolds.openai_agents import _alias_reasoning_field

        msg = self._Msg(content=None, reasoning="the answer")
        _alias_reasoning_field(self._Completion([self._Choice(msg)]))
        assert getattr(msg, "reasoning_content", None) == "the answer"

    def test_an_existing_reasoning_content_is_not_overwritten(self):
        from olmo_eval.harness.scaffolds.openai_agents import _alias_reasoning_field

        msg = self._Msg(content=None, reasoning="from vllm", reasoning_content="already here")
        _alias_reasoning_field(self._Completion([self._Choice(msg)]))
        assert msg.reasoning_content == "already here"

    def test_a_message_without_reasoning_is_untouched(self):
        from olmo_eval.harness.scaffolds.openai_agents import _alias_reasoning_field

        msg = self._Msg(content="a normal answer")
        _alias_reasoning_field(self._Completion([self._Choice(msg)]))
        assert getattr(msg, "reasoning_content", None) in (None, "")


class TestRecoverFromReasoningItem:
    """Once aliased, the SDK stores it as a ResponseReasoningItem, not as a message.

    Its text is in `summary[].text`. The earlier recovery looked for a `reasoning` attribute and
    so still found nothing -- the two halves of this fix each fail alone.
    """

    class _Item:
        def __init__(self, raw):
            self.raw_item = raw

    class _Result:
        def __init__(self, items):
            self.new_items = items
            self.final_output = ""

    def test_text_in_a_summary_is_recovered(self):
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        result = self._Result([
            self._Item({"summary": [{"text": "the recovered answer", "type": "summary_text"}]}),
        ])
        assert recover_answer_from_reasoning(result) == "the recovered answer"

    def test_multiple_summary_parts_are_joined(self):
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        result = self._Result([
            self._Item({"summary": [{"text": "first"}, {"text": "second"}]}),
        ])
        assert recover_answer_from_reasoning(result) == "first\n\nsecond"

    def test_a_message_with_content_still_wins(self):
        from olmo_eval.harness.scaffolds.openai_agents import recover_answer_from_reasoning

        result = self._Result([
            self._Item({"role": "assistant", "content": "a real answer"}),
        ])
        assert recover_answer_from_reasoning(result) == ""


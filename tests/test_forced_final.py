"""The forced-final call must not send tool_choice, because it sends no tools.

The scaffold used to pair tools=[] with tool_choice="none". vLLM and DeepSeek accept that; OpenAI
returns HTTP 400 "'tool_choice' is only allowed when 'tools' are specified", so every instance
that reached max_turns lost its final synthesis -- 2 of 100 on a gpt-5.6-sol smoke, silently,
because the partial answer was kept and nothing downstream said the closing call had failed.

The pairing was redundant anyway: with no tools there is nothing to choose.
"""

from __future__ import annotations

import asyncio
import types

from agents.models.chatcmpl_converter import Converter
from openai import Omit

from olmo_eval.harness.scaffolds.openai_agents import OpenAIAgentsScaffold


class _CapturingRunner:
    """Stands in for agents.Runner and keeps the agent it was handed."""

    captured: dict = {}

    @classmethod
    async def run(cls, *, starting_agent, input, max_turns):  # noqa: A002
        cls.captured["agent"] = starting_agent
        cls.captured["max_turns"] = max_turns
        return types.SimpleNamespace(final_output="a closing answer")


def _run_forced_final():
    from agents import Agent, ModelSettings

    agent = Agent(
        name="openai_agents",
        instructions="",
        model=None,
        tools=[],
        model_settings=ModelSettings(),
    )
    scaffold = OpenAIAgentsScaffold()
    _CapturingRunner.captured = {}
    text = asyncio.run(
        scaffold._force_final_answer(
            Runner=_CapturingRunner,
            agent=agent,
            partial_result=None,
            original_input="a question",
        )
    )
    return text, _CapturingRunner.captured["agent"]


def test_the_forced_final_agent_sends_no_tool_choice():
    _text, final_agent = _run_forced_final()
    assert final_agent.tools == []
    assert final_agent.model_settings.tool_choice is None


def test_that_setting_is_what_makes_the_parameter_disappear():
    """None is not merely falsy here -- it is what the converter turns into an omitted field."""
    assert isinstance(Converter.convert_tool_choice(None), Omit)
    assert Converter.convert_tool_choice("none") == "none"


def test_the_closing_call_still_returns_its_answer():
    text, _agent = _run_forced_final()
    assert text == "a closing answer"

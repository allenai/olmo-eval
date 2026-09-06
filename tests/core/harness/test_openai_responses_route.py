"""W0149: the Responses route the scaffold takes for a thinking model on the managed API.

The managed OpenAI endpoint refuses a chat completion that carries ``tools`` at any
``reasoning_effort`` but ``none``, and the 5.x models default to ``medium``. The remedy the
preset documents, ``reasoning_effort=none``, buys the tools by turning the thinking off; this
route buys both, at the price of a reasoning **summary** instead of the model's own chain.

Everything here is offline: the route is chosen from the client's base url and the run's own
settings, and the request body comes from the SDK's own builder rather than from a call.
"""

from types import SimpleNamespace

import pytest

from olmo_eval.common.types.trajectory import AgentTurn
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.scaffolds.openai_agents import (
    OpenAIAgentsScaffold,
    responses_api_required,
    with_reasoning_summary,
)

pytest.importorskip("agents")

OPENAI = "https://api.openai.com/v1"
VLLM = "http://localhost:8000/v1"
DEEPSEEK = "https://api.deepseek.com/v1"


#: Absent rather than None: only the self-hosted provider family has the attribute at all,
#: which is how the scaffold knows a body field is safe to send.
_MISSING = object()


def provider(base_url=OPENAI, model_name="gpt-5.6-sol", chat_template_kwargs=_MISSING):
    """A provider stub; ``chat_template_kwargs`` is present only for the self-hosted family."""

    client = SimpleNamespace(base_url=base_url)
    fields = {"model_name": model_name, "get_openai_client": lambda: client}
    if chat_template_kwargs is not _MISSING:
        fields["chat_template_kwargs"] = chat_template_kwargs
    return SimpleNamespace(**fields)


def agent_for(effort=None, base_url=OPENAI, tools=(), **provider_kwargs):
    settings = {"reasoning_effort": effort} if effort is not None else None
    config = HarnessConfig(
        name="test",
        tools=tuple(tools),
        scaffold_kwargs={"model_settings": settings} if settings else {},
    )
    scaffold = OpenAIAgentsScaffold()
    agent = scaffold._create_agent(provider(base_url=base_url, **provider_kwargs), config)
    return scaffold, agent


def model_class(agent) -> str:
    return type(agent.model).__name__


class TestRouteSelection:
    def test_the_managed_api_with_an_effort_takes_the_responses_route(self):
        for effort in ("low", "medium", "high", "xhigh", "max"):
            _, agent = agent_for(effort=effort)
            assert model_class(agent) == "OpenAIResponsesModel", effort

    def test_effort_none_keeps_the_chat_completions_route(self):
        """The remedy the preset documents still works, and still means chat completions."""

        _, agent = agent_for(effort="none")
        assert model_class(agent) == "ReasoningFieldChatCompletionsModel"

    def test_an_unpinned_effort_keeps_the_chat_completions_route(self):
        """The same endpoint serves models with no reasoning at all, and a ``reasoning`` block
        is its own 400 for those, so the run has to say which it wants."""

        _, agent = agent_for(effort=None)
        assert model_class(agent) == "ReasoningFieldChatCompletionsModel"
        assert agent.model_settings is None or agent.model_settings.reasoning is None

    def test_no_other_provider_moves(self):
        for base_url in (VLLM, DEEPSEEK, "https://api.openai.com.evil.test/v1"):
            for effort in (None, "none", "medium"):
                _, agent = agent_for(effort=effort, base_url=base_url, chat_template_kwargs=None)
                assert model_class(agent) == "ReasoningFieldChatCompletionsModel", (
                    base_url,
                    effort,
                )

    def test_the_predicate_reads_the_client_and_the_settings_and_nothing_else(self):
        from agents import ModelSettings
        from openai.types.shared import Reasoning

        openai_client = SimpleNamespace(base_url=OPENAI)
        vllm_client = SimpleNamespace(base_url=VLLM)
        medium = ModelSettings(reasoning=Reasoning(effort="medium"))
        assert responses_api_required(openai_client, medium)
        assert not responses_api_required(vllm_client, medium)
        assert not responses_api_required(openai_client, ModelSettings())
        assert not responses_api_required(openai_client, None)
        assert not responses_api_required(
            openai_client, ModelSettings(reasoning=Reasoning(effort="none"))
        )


class TestRequestShape:
    def build(self, effort="medium", tools=()):
        scaffold, agent = agent_for(effort=effort, tools=tools)
        return agent.model._build_response_create_kwargs(
            system_instructions=agent.instructions,
            input="write the related work section",
            model_settings=agent.model_settings,
            tools=agent.tools,
            output_schema=None,
            handoffs=[],
            previous_response_id=None,
            conversation_id=None,
            stream=False,
        )

    def test_the_effort_the_run_asked_for_goes_out_beside_a_summary_request(self):
        body = self.build()
        assert body["reasoning"].effort == "medium"
        assert body["reasoning"].summary == "auto"
        assert body["model"] == "gpt-5.6-sol"

    def test_the_summary_is_added_without_disturbing_anything_else(self):
        from agents import ModelSettings
        from openai.types.shared import Reasoning

        settings = ModelSettings(reasoning=Reasoning(effort="high"), max_tokens=4096)
        asked = with_reasoning_summary(settings)
        assert asked.reasoning.effort == "high"
        assert asked.reasoning.summary == "auto"
        assert asked.max_tokens == 4096
        # An unpinned effort stays unpinned: the provider's own default is not second-guessed.
        assert with_reasoning_summary(None).reasoning.effort is None
        assert with_reasoning_summary(None).reasoning.summary == "auto"

    def test_nothing_a_chat_completion_needs_and_this_route_rejects_is_sent(self):
        body = self.build()
        for field in ("reasoning_effort", "chat_template_kwargs", "messages", "max_tokens"):
            assert field not in body, field
        assert "input" in body

    def test_the_tool_and_the_loop_are_the_presets(self):
        from olmo_eval.harness.presets import HarnessPresets

        preset = HarnessPresets.arxiv_paper_search_agent
        config = HarnessConfig(
            name="test",
            tools=preset.tools,
            system_prompt=preset.system_prompt,
            max_turns=preset.max_turns,
            scaffold_kwargs={"model_settings": {"reasoning_effort": "medium"}},
        )
        scaffold = OpenAIAgentsScaffold()
        agent = scaffold._create_agent(provider(), config)
        assert type(agent.model).__name__ == "OpenAIResponsesModel"
        assert [tool.name for tool in agent.tools] == ["arxiv_paper_search"]
        body = agent.model._build_response_create_kwargs(
            system_instructions=agent.instructions,
            input="write it",
            model_settings=agent.model_settings,
            tools=agent.tools,
            output_schema=None,
            handoffs=[],
            previous_response_id=None,
            conversation_id=None,
            stream=False,
        )
        assert [tool["name"] for tool in body["tools"]] == ["arxiv_paper_search"]
        assert body["instructions"] == preset.system_prompt


class TestUsageIsKept:
    """A reasoning arm can only be costed from the usage the SDK already has: the reasoning
    tokens are billed as output, they are what an effort setting changes, and they appear
    nowhere in the trajectory because the provider returns a summary rather than the thinking.
    """

    def _response(self, **usage):
        from agents.items import ModelResponse
        from agents.usage import Usage
        from openai.types.responses.response_usage import (
            InputTokensDetails,
            OutputTokensDetails,
        )

        return ModelResponse(
            output=[],
            usage=Usage(
                requests=1,
                input_tokens=usage.get("input", 0),
                output_tokens=usage.get("output", 0),
                total_tokens=usage.get("input", 0) + usage.get("output", 0),
                input_tokens_details=InputTokensDetails(
                    cached_tokens=usage.get("cached", 0), cache_write_tokens=0
                ),
                output_tokens_details=OutputTokensDetails(
                    reasoning_tokens=usage.get("reasoning", 0)
                ),
            ),
            response_id="resp_1",
        )

    def test_every_model_call_is_recorded_in_order(self):
        scaffold, agent = agent_for(effort="medium")
        result = SimpleNamespace(
            new_items=[],
            raw_responses=[
                self._response(input=1200, output=900, reasoning=800, cached=1024),
                self._response(input=4000, output=300),
            ],
        )
        trajectory = scaffold._convert_trajectory(result)
        assert trajectory.metadata["model_calls"] == [
            {
                "input_tokens": 1200,
                "output_tokens": 900,
                "total_tokens": 2100,
                "reasoning_tokens": 800,
                "cached_input_tokens": 1024,
            },
            {"input_tokens": 4000, "output_tokens": 300, "total_tokens": 4300},
        ]

    def test_a_run_that_made_no_call_records_nothing(self):
        scaffold, agent = agent_for(effort="medium")
        for result in (
            SimpleNamespace(new_items=[], raw_responses=[]),
            SimpleNamespace(new_items=[]),
        ):
            assert scaffold._convert_trajectory(result).metadata == {}

    def test_an_old_route_keeps_the_trajectory_it_has_always_written(self):
        """The SDK builds a Usage object for every call whether or not the provider reported
        one, so attaching this on every route would put a `model_calls` key -- zeros included --
        on every self-hosted trajectory saved from here on. It goes on the new route only.
        """

        scaffold, agent = agent_for(effort="medium", base_url=VLLM, chat_template_kwargs=None)
        assert type(agent.model).__name__ == "ReasoningFieldChatCompletionsModel"
        result = SimpleNamespace(
            new_items=[],
            raw_responses=[self._response(input=100, output=20), self._response()],
        )
        trajectory = scaffold._convert_trajectory(result)
        assert trajectory.metadata == {}
        assert "metadata" not in trajectory.to_dict()


class TestReasoningIsLabelled:
    def _items(self, agent, summary: str):
        from agents.items import ReasoningItem
        from openai.types.responses import ResponseReasoningItem
        from openai.types.responses.response_reasoning_item import Summary

        raw = ResponseReasoningItem(
            id="rs_1",
            type="reasoning",
            summary=[Summary(text=summary, type="summary_text")],
        )
        return [ReasoningItem(agent=agent, raw_item=raw)]

    def _message(self, agent, text: str):
        from agents.items import MessageOutputItem
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        raw = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
        )
        return MessageOutputItem(agent=agent, raw_item=raw)

    def test_a_summary_is_recorded_and_said_to_be_a_summary(self):
        scaffold, agent = agent_for(effort="medium")
        items = self._items(agent, "Searched arXiv, then wrote the section.")
        items.append(self._message(agent, "=== FINAL REPORT ==="))
        trajectory = scaffold._convert_trajectory(SimpleNamespace(new_items=items))
        turn = trajectory.turns[0]
        assert turn.reasoning == "Searched arXiv, then wrote the section."
        assert turn.metadata == {"reasoning_kind": "summary"}
        assert turn.to_dict()["metadata"] == {"reasoning_kind": "summary"}

    def test_a_chain_is_recorded_with_no_label_at_all(self):
        """Every self-hosted arm records the thinking itself, so its turns say nothing extra
        and a turn saved before this existed round-trips unchanged."""

        scaffold, agent = agent_for(effort="medium", base_url=VLLM, chat_template_kwargs=None)
        items = self._items(agent, "<the model's own thinking>")
        items.append(self._message(agent, "answer"))
        turn = scaffold._convert_trajectory(SimpleNamespace(new_items=items)).turns[0]
        assert turn.reasoning == "<the model's own thinking>"
        assert turn.metadata == {}
        assert "metadata" not in turn.to_dict()

    def test_a_turn_with_no_reasoning_claims_no_summary(self):
        """One model response carries one reasoning block, and the Responses route answers a
        search turn with six parallel calls; only the turn that holds the summary says so.
        Found on the first live case, where 55 of 61 turns claimed a summary they did not hold.
        """

        scaffold, agent = agent_for(effort="medium")
        items = self._items(agent, "One thought.")
        items.append(self._message(agent, "answer"))
        turns = scaffold._convert_trajectory(SimpleNamespace(new_items=items)).turns
        assert [t.metadata for t in turns] == [{"reasoning_kind": "summary"}]
        assert AgentTurn.assistant(content="x", reasoning_kind="summary").metadata == {}
        # A response that reasoned and returned no summary still says what the field is.
        assert AgentTurn.assistant(
            content="x", reasoning="", reasoning_kind="summary"
        ).metadata == {"reasoning_kind": "summary"}

    def test_the_turn_type_still_round_trips(self):
        labelled = AgentTurn.assistant(content="x", reasoning="y", reasoning_kind="summary")
        assert AgentTurn.from_dict(labelled.to_dict()) == labelled
        plain = AgentTurn.assistant(content="x", reasoning="y")
        assert AgentTurn.from_dict(plain.to_dict()) == plain
        assert plain.metadata == {}

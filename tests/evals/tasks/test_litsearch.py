"""Tests for the LitSearch agentic literature-search task."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.common.types.tools import Function, ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.litsearch import (
    SEARCH_TOOL_NAME,
    extract_corpus_ids,
    score_litsearch,
)


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def task():
    return get_task("litsearch")


def _trajectory(*result_contents: str, query: str = "q") -> AgentTrajectory:
    turns = []
    for i, content in enumerate(result_contents):
        call = ToolCall(
            id=f"c{i}",
            function=Function(name=SEARCH_TOOL_NAME, arguments=f'{{"query": "{query}"}}'),
        )
        turns.append(AgentTurn.assistant(tool_calls=[call]))
        turns.append(AgentTurn.tool(results=[ToolResult(tool_call_id=f"c{i}", content=content)]))
    return AgentTrajectory(turns=tuple(turns))


def _response(instance: Instance, trajectory: AgentTrajectory | None) -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text="done")],
        scores={},
        trajectory=trajectory,
    )


class TestRegistration:
    def test_registered(self):
        assert "litsearch" in list_tasks()

    def test_metrics(self, task):
        assert {m.name for m in task.config.metrics} == {"found_rate", "gold_recall"}
        assert task.config.get_primary_metric().name == "found_rate"

    def test_data_source(self, task):
        assert task.config.data_source.path == "princeton-nlp/LitSearch"
        assert task.config.data_source.subset == "query"


class TestProcessDoc:
    def test_keeps_gold_ids(self, task):
        doc = {"query": "kd papers?", "corpusids": [202719327, 111], "query_set": "inline_acl"}
        instance = task.process_doc(doc, index=5)
        assert instance is not None
        assert instance.question == "kd papers?"
        assert instance.metadata["gold_corpusids"] == [202719327, 111]
        assert instance.metadata["case_id"] == "litsearch_5"

    def test_skips_empty_query(self, task):
        assert task.process_doc({"query": "", "corpusids": [1]}, 0) is None

    def test_skips_no_gold(self, task):
        assert task.process_doc({"query": "q", "corpusids": []}, 0) is None


class TestFormatRequest:
    def test_embeds_query_and_tool(self, task):
        request = task.format_request(Instance(question="find papers on X", metadata={}))
        assert request.request_type == RequestType.CHAT
        content = request.messages[0]["content"]
        assert "find papers on X" in content
        assert SEARCH_TOOL_NAME in content


class TestExtractCorpusIds:
    def test_parses_ids_across_results(self):
        traj = _trajectory(
            "**A** (2020)\nCorpus ID: 202719327\n",
            "**B**\nCorpus ID: 999\n**C**\nCorpus ID: 12\n",
        )
        response = _response(Instance(question="q", metadata={}), traj)
        assert extract_corpus_ids(response) == {202719327, 999, 12}

    def test_no_trajectory_is_empty(self):
        response = _response(Instance(question="q", metadata={}), None)
        assert extract_corpus_ids(response) == set()


class TestScoreLitsearch:
    def test_hit_and_partial_recall(self):
        assert score_litsearch({1, 2}, {1, 99}) == {"found_rate": 1.0, "gold_recall": 0.5}

    def test_full_recall(self):
        assert score_litsearch({1, 2}, {1, 2, 3}) == {"found_rate": 1.0, "gold_recall": 1.0}

    def test_miss(self):
        assert score_litsearch({1, 2}, {7, 8}) == {"found_rate": 0.0, "gold_recall": 0.0}

    def test_no_gold(self):
        assert score_litsearch(set(), {1}) == {"found_rate": 0.0, "gold_recall": 0.0}


class TestScoreResponses:
    @pytest.mark.anyio
    async def test_scores_from_trajectory(self, task):
        instance = Instance(question="q", metadata={"gold_corpusids": [202719327, 111]})
        traj = _trajectory("**A**\nCorpus ID: 202719327\n**B**\nCorpus ID: 999\n")
        response = _response(instance, traj)

        await task.score_responses([response])

        assert response.scores == {"found_rate": 1.0, "gold_recall": 0.5}
        meta = response.outputs[0].metadata
        assert meta["litsearch_seen_corpusids"] == [999, 202719327]
        assert meta["litsearch_num_searches"] == 1

    @pytest.mark.anyio
    async def test_no_trajectory_scores_zero(self, task):
        instance = Instance(question="q", metadata={"gold_corpusids": [1]})
        response = _response(instance, None)

        await task.score_responses([response])

        assert response.scores == {"found_rate": 0.0, "gold_recall": 0.0}
        assert response.outputs[0].metadata["litsearch_num_searches"] == 0

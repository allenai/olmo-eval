"""Tests for the SAGE short-form retrieval task."""

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks
from olmo_eval.evals.tasks.sage import (
    SageExactMatchMetric,
    SageExactMatchScorer,
    SageWeightedRecallMetric,
    SageWeightedRecallScorer,
)


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def short_form_doc():
    return {
        "paper_id": "seed-paper-id",
        "paper_title": "Seed title",
        "complete_query": "Find the paper that introduced a compact retrieval benchmark.",
        "domain": "cs",
        "ground_truth": {
            "paperId": "gold-paper-id",
            "title": "Compact Retrieval Benchmarks for Deep Research Agents",
            "abstract": "A benchmark paper.",
            "arxiv_id": "2602.05975",
            "doi": "10.0000/example",
            "corpus_id": "12345",
        },
    }


@pytest.fixture
def open_ended_doc():
    return {
        "case_id": "open-case-id",
        "question": "Which papers introduced compact and open evidence retrieval benchmarks?",
        "domain": "cs",
        "ground_truth": {
            "most_relevant": [
                {
                    "paperId": "most-1",
                    "title": "Compact Retrieval Benchmarks for Deep Research Agents",
                    "abstract": "A compact benchmark paper.",
                    "arxiv_id": "2602.05975",
                    "doi": "10.0000/most1",
                    "corpus_id": "12345",
                },
                {
                    "paperId": "most-2",
                    "title": "Deep Citations for Scientific Search",
                    "abstract": "A citation search paper.",
                    "arxivId": "2603.12345",
                    "DOI": "10.0000/most2",
                    "corpusId": "23456",
                },
            ],
            "relevant": [
                {
                    "paperId": "rel-1",
                    "title": "Open Evidence Maps for Biology",
                    "abstract": "A biology evidence paper.",
                    "arxiv_id": "2604.00001",
                    "doi": "10.0000/rel1",
                    "corpus_id": "34567",
                },
                {
                    "paper_id": "rel-2",
                    "title": "Literature Agents Need Grounded Retrieval",
                    "abstract": "An agentic retrieval paper.",
                },
            ],
        },
    }


@pytest.fixture
def task():
    return get_task("sage_short_form")


@pytest.fixture
def open_task():
    return get_task("sage_open_ended")


def test_task_registered():
    assert "sage_short_form" in list_tasks()
    assert "sage_open_ended" in list_tasks()


def test_process_doc_builds_instance(task, short_form_doc):
    instance = task.process_doc(short_form_doc, index=7)

    assert instance is not None
    assert isinstance(instance, Instance)
    assert instance.question == short_form_doc["complete_query"]
    assert instance.metadata["case_id"] == "seed-paper-id"
    assert instance.metadata["domain"] == "cs"
    assert instance.metadata["index"] == 7
    assert instance.metadata["gold"] == {
        "paperId": "gold-paper-id",
        "title": "Compact Retrieval Benchmarks for Deep Research Agents",
        "abstract": "A benchmark paper.",
        "arxiv_id": "2602.05975",
        "doi": "10.0000/example",
        "corpus_id": "12345",
    }


@pytest.mark.parametrize(
    "patch",
    (
        {"complete_query": ""},
        {"ground_truth": {"paperId": "gold-paper-id"}},
        {"ground_truth": None},
    ),
)
def test_process_doc_skips_missing_query_or_gold(task, short_form_doc, patch):
    doc = {**short_form_doc, **patch}
    assert task.process_doc(doc, index=0) is None


def test_format_request_is_harness_agnostic_chat(task, short_form_doc):
    instance = task.process_doc(short_form_doc)
    assert instance is not None

    request = task.format_request(instance)

    assert isinstance(request, LMRequest)
    assert request.request_type == RequestType.CHAT
    assert len(request.messages) == 1
    prompt = request.messages[0]["content"]
    assert instance.question in prompt
    assert "single best answer" in prompt
    assert "no match was found" in prompt
    assert "title" in prompt
    assert "semantic_scholar" not in prompt.lower()
    assert "serper" not in prompt.lower()
    assert "dr_tulu" not in prompt.lower()
    assert "paper_search_agent" not in prompt.lower()


@pytest.mark.anyio
async def test_score_responses_with_default_normalized_matcher(task, short_form_doc):
    instance = task.process_doc(short_form_doc)
    assert instance is not None
    request = task.format_request(instance)
    hit = Response(
        instance=instance,
        request=request,
        outputs=[
            LMOutput(
                text=("The paper I found is Compact Retrieval Benchmarks for Deep Research Agents.")
            )
        ],
    )
    miss = Response(
        instance=instance,
        request=request,
        outputs=[LMOutput(text="The paper I found is a different retrieval benchmark.")],
    )

    scored = await task.score_responses([hit, miss])

    assert scored[0].scores["exact_match"] == 1.0
    assert scored[0].outputs[0].metadata["sage_matched"] is True
    assert scored[0].outputs[0].metadata["exact_match"] == 1.0
    assert scored[1].scores["exact_match"] == 0.0
    assert scored[1].outputs[0].metadata["sage_matched"] is False
    assert scored[1].outputs[0].metadata["exact_match"] == 0.0
    assert SageExactMatchMetric().compute(scored) == pytest.approx(0.5)


@pytest.mark.anyio
async def test_sage_exact_match_scorer_reads_scored_output_metadata(task, short_form_doc):
    instance = task.process_doc(short_form_doc)
    assert instance is not None
    request = task.format_request(instance)
    hit = Response(
        instance=instance,
        request=request,
        outputs=[
            LMOutput(
                text=("The paper I found is Compact Retrieval Benchmarks for Deep Research Agents.")
            )
        ],
    )
    miss = Response(
        instance=instance,
        request=request,
        outputs=[LMOutput(text="The paper I found is a different retrieval benchmark.")],
    )

    scored = await task.score_responses([hit, miss])
    scorer = SageExactMatchScorer()

    assert scorer.score(instance, scored[0].outputs[0]) == 1.0
    assert scorer.score(instance, scored[1].outputs[0]) == 0.0


def test_open_ended_process_doc_builds_weighted_golds(open_task, open_ended_doc):
    instance = open_task.process_doc(open_ended_doc, index=11)

    assert instance is not None
    assert isinstance(instance, Instance)
    assert instance.question == open_ended_doc["question"]
    assert instance.metadata["case_id"] == "open-case-id"
    assert instance.metadata["domain"] == "cs"
    assert instance.metadata["index"] == 11

    golds = instance.metadata["golds"]
    assert [relevance for _, relevance in golds] == [2, 2, 1, 1]
    assert [gold["paperId"] for gold, _ in golds] == ["most-1", "most-2", "rel-1", "rel-2"]
    assert [gold["title"] for gold, _ in golds] == [
        "Compact Retrieval Benchmarks for Deep Research Agents",
        "Deep Citations for Scientific Search",
        "Open Evidence Maps for Biology",
        "Literature Agents Need Grounded Retrieval",
    ]
    assert golds[0][0]["arxiv_id"] == "2602.05975"
    assert golds[1][0]["arxiv_id"] == "2603.12345"
    assert golds[1][0]["doi"] == "10.0000/most2"
    assert golds[1][0]["corpus_id"] == "23456"


@pytest.mark.parametrize(
    "patch",
    (
        {"question": ""},
        {"ground_truth": {"most_relevant": [], "relevant": []}},
        {"ground_truth": None},
    ),
)
def test_open_ended_process_doc_skips_missing_question_or_golds(open_task, open_ended_doc, patch):
    doc = {**open_ended_doc, **patch}
    assert open_task.process_doc(doc, index=0) is None


def test_open_ended_format_request_is_harness_agnostic_chat(open_task, open_ended_doc):
    instance = open_task.process_doc(open_ended_doc)
    assert instance is not None

    request = open_task.format_request(instance)

    assert isinstance(request, LMRequest)
    assert request.request_type == RequestType.CHAT
    assert len(request.messages) == 1
    prompt = request.messages[0]["content"]
    assert instance.question in prompt
    assert "final answer" in prompt
    assert "titles" in prompt
    assert "semantic_scholar" not in prompt.lower()
    assert "serper" not in prompt.lower()
    assert "dr_tulu" not in prompt.lower()
    assert "paper_search_agent" not in prompt.lower()


@pytest.mark.anyio
async def test_open_ended_score_responses_weighted_recall(open_task, open_ended_doc):
    instance = open_task.process_doc(open_ended_doc)
    assert instance is not None
    request = open_task.format_request(instance)
    response = Response(
        instance=instance,
        request=request,
        outputs=[
            LMOutput(
                text=(
                    "The relevant papers include Compact Retrieval Benchmarks for "
                    "Deep Research Agents and Open Evidence Maps for Biology."
                )
            )
        ],
    )

    scored = await open_task.score_responses([response])

    assert scored[0].scores["weighted_recall"] == pytest.approx(0.5)
    assert scored[0].outputs[0].metadata["weighted_recall"] == pytest.approx(0.5)
    assert SageWeightedRecallMetric().compute(scored) == pytest.approx(0.5)
    assert SageWeightedRecallScorer().score(instance, scored[0].outputs[0]) == pytest.approx(0.5)

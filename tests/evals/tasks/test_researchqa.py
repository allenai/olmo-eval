"""Tests for the ResearchQA agentic scholarly-QA task."""

import logging
from collections.abc import Sequence

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks import researchqa
from olmo_eval.evals.tasks.common import get_task, task_exists
from olmo_eval.evals.tasks.researchqa import (
    RESEARCHQA_GENERATION_PROMPT,
    RESEARCHQA_LABEL_SCORES,
    build_coverage_judge_prompt,
    build_researchqa_judge_fn,
    normalize_coverage_label,
    parse_coverage_labels,
)


@pytest.fixture
def task():
    return get_task("researchqa")


def _rubric_item(
    text: str,
    types: Sequence[str] = ("Other",),
    citation_metadata=None,
):
    return {
        "rubric_item": text,
        "type": list(types),
        "citation_metadata": citation_metadata,
    }


def _doc(rubric=None, query="How does TiO₂ photocatalysis remove pollutants?"):
    return {
        "id": "174539438139989436-s6",
        "general_domain": "Physical & Theoretical Sciences",
        "subdomain": "Chemical Sciences",
        "field": "Materials Chemistry",
        "query": query,
        "date": "2021-11-03",
        "rubric": rubric
        if rubric is not None
        else [
            _rubric_item(
                "Does the response compare TiO₂ with ZnO?",
                ("Comparison", "Example"),
            ),
            _rubric_item(
                "Does the response cite Alestalo (1971)?",
                ("Citation",),
                {
                    "first_author": "Alestalo",
                    "title": "Dendrochronological interpretation of geomorphic processes",
                    "url": "http://fennia.journal.fi/article/view/40757",
                    "year": 1971,
                },
            ),
        ],
    }


def _response(instance: Instance, text: str = "A researched answer.") -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT, messages=()),
        outputs=[LMOutput(text=text)],
        scores={},
    )


def _instance_with_rubric(rubric) -> Instance:
    return Instance(question="A research question?", metadata={"rubric": rubric})


class TestRegistration:
    def test_both_splits_are_registered(self):
        assert task_exists("researchqa")
        assert task_exists("researchqa:mini")

        full = get_task("researchqa")
        mini = get_task("researchqa:mini")
        assert full.config.data_source.path == "realliyifei/ResearchQA"
        assert full.config.data_source.subset == "default"
        assert full.config.data_source.split == "test"
        assert mini.config.data_source.split == "test_mini"

    def test_primary_and_secondary_metrics(self, task):
        assert task.config.get_primary_metric().name == "rubric_coverage"
        assert {metric.name for metric in task.config.metrics} == {
            "rubric_coverage",
            "coverage_citation",
            "coverage_limitation",
            "coverage_comparison",
            "coverage_example",
            "coverage_impact",
            "coverage_other",
        }


class TestProcessDoc:
    def test_maps_real_schema_and_preserves_unicode(self, task):
        doc = _doc()
        instance = task.process_doc(doc, index=7)

        assert instance is not None
        assert instance.question == "How does TiO₂ photocatalysis remove pollutants?"
        assert instance.metadata["id"] == "174539438139989436-s6"
        assert instance.metadata["case_id"] == "174539438139989436-s6"
        assert instance.metadata["general_domain"] == "Physical & Theoretical Sciences"
        assert instance.metadata["subdomain"] == "Chemical Sciences"
        assert instance.metadata["field"] == "Materials Chemistry"
        assert instance.metadata["date"] == "2021-11-03"
        assert instance.metadata["index"] == 7

        rubric = instance.metadata["rubric"]
        assert rubric[0]["rubric_item"] == "Does the response compare TiO₂ with ZnO?"
        assert rubric[0]["type"] == ["Comparison", "Example"]
        assert rubric[0]["citation_metadata"] is None
        assert rubric[1]["citation_metadata"] == doc["rubric"][1]["citation_metadata"]

    def test_accepts_one_item_rubric_and_nullable_citation_fields(self, task):
        citation = {
            "first_author": None,
            "title": "A scientific paper",
            "url": "https://example.test/paper",
            "year": None,
        }
        instance = task.process_doc(
            _doc([_rubric_item("Does the response cite the paper?", ("Citation",), citation)])
        )

        assert instance is not None
        assert len(instance.metadata["rubric"]) == 1
        assert instance.metadata["rubric"][0]["citation_metadata"] == citation

    def test_rejects_scalar_type_from_incorrect_loader_schema(self, task):
        doc = _doc([_rubric_item("Does it explain the effect?")])
        doc["rubric"][0]["type"] = "Impact"

        with pytest.raises(TypeError, match=r"rubric\[0\]\.type must be a list"):
            task.process_doc(doc)

    def test_skips_missing_query_or_empty_rubric(self, task):
        assert task.process_doc(_doc(query="")) is None
        assert task.process_doc(_doc(rubric=[])) is None


class TestPrompts:
    def test_generation_prompt_invariants_and_ordering(self, task):
        query = "What limits solid-state batteries?"
        instance = task.process_doc(_doc(query=query))
        assert instance is not None
        request = task.format_request(instance)
        content = request.messages[0]["content"]

        assert request.request_type == RequestType.CHAT
        assert "semantic_scholar_snippet_search" in content
        assert "serper_google_webpage_search" in content
        assert "serper_fetch_webpage_content" in content
        assert "about 250 words" in content
        assert "Prefer sources available on or before 2021-11-03." in content
        assert "Search first, then fetch and read, and only then answer" in content
        assert "Do not answer from memory" in content
        assert content.endswith(query)
        assert content.index("Format and length requirements") < content.index(
            "Workflow (follow this order)"
        )
        assert content.index("Prefer sources available on or before 2021-11-03.") < content.index(
            "Workflow (follow this order)"
        )
        assert RESEARCHQA_GENERATION_PROMPT.endswith("Question: ")

    def test_generation_prompt_omits_cutoff_when_date_is_missing(self, task):
        doc = _doc()
        doc["date"] = None
        instance = task.process_doc(doc)

        assert instance is not None
        content = task.format_request(instance).messages[0]["content"]
        assert "Prefer sources available on or before" not in content

    def test_coverage_prompt_matches_official_rendering_byte_for_byte(self):
        expected = (
            "Please judge the following questions based on the response below.\n"
            "For each question, select one of the following ratings to indicate the extent "
            "to which the response addresses the question:\n"
            "Not at all, Barely, Moderately, Mostly, Completely\n\n"
            "Definitions:\n"
            "- Not at all: *totally uninferable*\n"
            "- Barely: *unmentioned but inferrable*\n"
            "- Moderately: *mentioned but misses important details*\n"
            "- Mostly: *mentioned but misses some details*\n"
            "- Completely: *mentioned with sufficient details*\n\n"
            "Only output one of the five phrases for each question, separated by newlines, "
            "and nothing else.\n\n"
            "Response: A\n"
            "Questions:\n"
            "q1?\n"
            "q2?\n\n"
            "Output:"
        )

        assert build_coverage_judge_prompt("A", ["q1?", "q2?"]) == expected


class TestExtractAnswer:
    def test_strips_think_block(self, task):
        output = LMOutput(
            text="<think>\nplanning noise\n</think>\n\nThe real answer covers X and Y."
        )
        assert task.extract_answer(output) == "The real answer covers X and Y."

        output = LMOutput(text="A clean answer.")
        assert task.extract_answer(output) == "A clean answer."


class TestParsingAndScoring:
    def test_label_map_and_normalization_are_exact(self):
        assert RESEARCHQA_LABEL_SCORES == {
            "Not at all": 1,
            "Barely": 2,
            "Moderately": 3,
            "Mostly": 4,
            "Completely": 5,
        }
        assert [
            normalize_coverage_label(label)
            for label in ["Not at all", "Barely", "Moderately", "Mostly", "Completely"]
        ] == [0.0, 0.25, 0.5, 0.75, 1.0]
        assert parse_coverage_labels("Barely\n\nCompletely\n", 2) == [
            "Barely",
            "Completely",
        ]
        assert parse_coverage_labels("Unknown", 1) is None

    @pytest.mark.anyio
    async def test_chunks_nine_items_into_two_judge_calls(self, task, monkeypatch):
        rubric = [_rubric_item(f"Does it cover item {index}?") for index in range(9)]
        response = _response(_instance_with_rubric(rubric))
        prompts = []
        outputs = iter(["\n".join(["Completely"] * 8), "Not at all"])

        async def judge(prompt):
            prompts.append(prompt)
            return next(outputs)

        monkeypatch.setattr(researchqa, "build_researchqa_judge_fn", lambda: judge)
        await task.score_responses([response])

        assert len(prompts) == 2
        assert prompts[0].endswith(
            "Questions:\n"
            + "\n".join(f"Does it cover item {index}?" for index in range(8))
            + "\n\nOutput:"
        )
        assert prompts[1].endswith("Questions:\nDoes it cover item 8?\n\nOutput:")
        assert response.scores["rubric_coverage"] == pytest.approx(8 / 9)
        assert response.outputs[0].metadata["researchqa_rubric_scores"] == [1.0] * 8 + [0.0]

    @pytest.mark.anyio
    async def test_primary_aggregation_means_instances_not_pooled_items(self, task, monkeypatch):
        responses = [
            _response(_instance_with_rubric([_rubric_item("One?")])),
            _response(
                _instance_with_rubric(
                    [_rubric_item("Two?"), _rubric_item("Three?"), _rubric_item("Four?")]
                )
            ),
        ]
        outputs = iter(["Completely", "Not at all\nNot at all\nNot at all"])

        async def judge(_prompt):
            return next(outputs)

        monkeypatch.setattr(researchqa, "build_researchqa_judge_fn", lambda: judge)
        await task.score_responses(responses)

        primary = task.config.get_primary_metric()
        assert [response.scores["rubric_coverage"] for response in responses] == [1.0, 0.0]
        assert primary.compute(responses) == 0.5

    @pytest.mark.anyio
    async def test_per_type_metrics_pool_items_and_multilabel_counts_twice(self, task, monkeypatch):
        first_rubric = [
            _rubric_item("A?", ("Comparison", "Example")),
            _rubric_item("B?", ("Comparison",)),
            _rubric_item("C?", ("Citation",)),
            _rubric_item("D?", ("Other", "Impact", "Limitation")),
        ]
        second_rubric = [_rubric_item("E?", ("Comparison",))]
        responses = [
            _response(_instance_with_rubric(first_rubric)),
            _response(_instance_with_rubric(second_rubric)),
        ]
        outputs = iter(["Completely\nModerately\nBarely\nMostly", "Not at all"])

        async def judge(_prompt):
            return next(outputs)

        monkeypatch.setattr(researchqa, "build_researchqa_judge_fn", lambda: judge)
        await task.score_responses(responses)

        assert responses[0].scores["coverage_comparison"] == 0.75
        assert responses[0].scores["coverage_example"] == 1.0
        assert responses[0].scores["coverage_citation"] == 0.25
        assert responses[0].scores["coverage_limitation"] == 0.75
        assert responses[0].scores["coverage_impact"] == 0.75
        assert responses[0].scores["coverage_other"] == 0.75

        metrics = {metric.name: metric.compute(responses) for metric in task.config.metrics}
        assert metrics["coverage_comparison"] == 0.5
        assert metrics["coverage_example"] == 1.0
        assert metrics["coverage_citation"] == 0.25
        assert metrics["coverage_limitation"] == 0.75
        assert metrics["coverage_impact"] == 0.75
        assert metrics["coverage_other"] == 0.75

    @pytest.mark.anyio
    async def test_parse_failure_retries_three_times_keeps_zero_instance(
        self, task, monkeypatch, caplog
    ):
        response = _response(
            _instance_with_rubric([_rubric_item("Does the response explain it?", ("Impact",))])
        )
        calls = 0
        delays = []

        async def judge(_prompt):
            nonlocal calls
            calls += 1
            return "garbage"

        async def sleep(delay):
            delays.append(delay)

        monkeypatch.setattr(researchqa, "build_researchqa_judge_fn", lambda: judge)
        monkeypatch.setattr(researchqa.asyncio, "sleep", sleep)
        with caplog.at_level(logging.WARNING, logger=researchqa.__name__):
            scored = await task.score_responses([response])

        assert calls == 3
        assert delays == [1, 2]
        assert scored == [response]
        assert response.scores["rubric_coverage"] == 0.0
        assert response.scores["coverage_impact"] == 0.0
        assert task.config.get_primary_metric().compute(scored) == 0.0
        assert response.outputs[0].metadata["researchqa_rubric_labels"] == [None]
        assert "1 unparseable batch(es) covering 1 rubric item(s)" in caplog.text
        assert "kept every instance in the denominator" in caplog.text


@pytest.mark.parametrize(
    ("env_spec", "expected_model", "expected_effort"),
    [
        (None, "gpt-5.4-mini", None),
        ("judge-override", "judge-override", None),
        ("gpt-5.4-mini:high", "gpt-5.4-mini", "high"),
    ],
)
def test_judge_spec_resolution(monkeypatch, env_spec, expected_model, expected_effort):
    captured = {}

    async def sentinel(_prompt):
        return ""

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return sentinel

    if env_spec is None:
        monkeypatch.delenv("OLMO_EVAL_JUDGE", raising=False)
    else:
        monkeypatch.setenv("OLMO_EVAL_JUDGE", env_spec)
    monkeypatch.setattr(researchqa, "build_openai_judge_fn", fake_builder)

    assert build_researchqa_judge_fn() is sentinel
    assert captured == {
        "model": expected_model,
        "temperature": 0.0,
        "max_tokens": 4096,
        "scorer_name": "ResearchQA",
        "reasoning_effort": expected_effort,
    }

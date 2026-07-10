"""Tests for the ExpertQA attributed long-form QA task."""

import json

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task, list_tasks


@pytest.fixture(autouse=True)
def _setup_registry():
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.fixture
def task():
    return get_task("expertqa")


class TestRegistration:
    def test_task_registered(self):
        assert "expertqa" in list_tasks()

    def test_metrics_exclude_ingredient_recall(self, task):
        metric_names = {m.name for m in task.config.metrics}
        assert metric_names == {
            "global_avg",
            "citation_precision",
            "citation_recall",
            "answer_precision",
        }

    def test_primary_metric(self, task):
        # Primary metric is a citation tier, not the global_avg aggregate.
        assert task.config.get_primary_metric().name == "citation_recall"

    def test_declares_judge_secret(self, task):
        assert task.config.required_secrets == ("OPENAI_API_KEY",)

    def test_data_source(self, task):
        source = task.config.data_source
        assert source.path == "cmalaviya/expertqa"
        assert source.subset == "main"


class TestProcessDoc:
    def test_keeps_field_metadata(self, task):
        doc = {
            "question": "What causes treatment resistance in oncology?",
            "metadata": {
                "field": "Healthcare / Medicine",
                "specific_field": "Oncology",
                "question_type": "Directed question",
            },
        }
        instance = task.process_doc(doc, index=3)
        assert instance is not None
        assert instance.question.startswith("What causes")
        assert instance.metadata["field"] == "Healthcare / Medicine"
        assert instance.metadata["specific_field"] == "Oncology"
        assert instance.metadata["case_id"] == "expertqa_3"
        assert instance.metadata["index"] == 3

    def test_missing_question_skipped(self, task):
        assert task.process_doc({"question": ""}, index=0) is None

    def test_missing_metadata_defaults_empty(self, task):
        instance = task.process_doc({"question": "Q?"}, index=0)
        assert instance is not None
        assert instance.metadata["field"] == ""
        assert instance.metadata["specific_field"] == ""


class TestExtractAnswer:
    def test_valid_json_response(self, task):
        output = LMOutput(text='{"sections": [{"text": "hello", "citations": []}]}')
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result
        assert output.metadata["parsed_response"] == result

    def test_invalid_json(self, task):
        output = LMOutput(text="not json at all")
        assert task.extract_answer(output) is None

    def test_strips_think_block(self, task):
        text = (
            '<think>\nplan: {"junk": 1}\n</think>\n{"sections": [{"text": "hi", "citations": []}]}'
        )
        output = LMOutput(text=text)
        result = task.extract_answer(output)
        assert result is not None
        assert "sections" in result


class TestFormatRequest:
    def test_chat_request_embeds_question(self, task):
        instance = Instance(question="What is attention?", metadata={})
        request = task.format_request(instance)
        assert request.request_type == RequestType.CHAT
        assert len(request.messages) == 1
        content = request.messages[0]["content"]
        assert "What is attention?" in content
        assert "Generate a report" in content


class TestScoreSingle:
    def _response(self, parsed):
        instance = Instance(question="What causes X?", metadata={})
        output = LMOutput(text="", metadata={"parsed_response": parsed})
        return Response(
            instance=instance,
            request=LMRequest(request_type=RequestType.CHAT, messages=()),
            outputs=[output],
            scores={},
        )

    @pytest.mark.anyio
    async def test_no_parsed_response_scores_zero(self, task):
        response = self._response(None)
        scores = await task._score_single(response, _unused_judge)
        assert scores == {
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "answer_precision": 0.0,
            "global_avg": 0.0,
        }

    @pytest.mark.anyio
    async def test_global_avg_is_mean_of_three_axes(self, task):
        parsed = {
            "sections": [
                {
                    "text": "Claim A [1]. Claim B [1].",
                    "citations": [
                        {"id": "[1]", "snippets": ["evidence from paper"], "title": "Paper"}
                    ],
                }
            ]
        }
        response = self._response(parsed)
        scores = await task._score_single(response, _citation_split_judge)
        # One of two claims attributable -> recall 0.5; precision averages 0.5;
        # no irrelevant paragraphs -> answer_precision 1.0.
        assert scores["citation_recall"] == pytest.approx(0.5)
        assert scores["citation_precision"] == pytest.approx(0.5)
        assert scores["answer_precision"] == pytest.approx(1.0)
        assert scores["global_avg"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)


async def _unused_judge(prompt, **kwargs):
    raise AssertionError("judge should not be called")


async def _citation_split_judge(prompt, **kwargs):
    """Stub judge: no irrelevant paragraphs, one supported and one unsupported claim."""
    if "irrelevant paragraphs" in prompt:
        return json.dumps({"irrelevant_paragraphs": []})
    return json.dumps(
        {
            "claims": [
                {
                    "text": "Claim A",
                    "supporting": ["[1]"],
                    "non_supporting": [],
                    "is_fully_supported": True,
                },
                {
                    "text": "Claim B",
                    "supporting": [],
                    "non_supporting": ["[1]"],
                    "is_fully_supported": False,
                },
            ]
        }
    )

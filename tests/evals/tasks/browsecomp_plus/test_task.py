"""BrowseComp-Plus reference grading and native task integration."""

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.common.types.tools import Function, ToolCall, ToolResult
from olmo_eval.common.types.trajectory import AgentTrajectory, AgentTurn
from olmo_eval.evals.tasks.browsecomp_plus.data import DATASET_FILENAME
from olmo_eval.evals.tasks.browsecomp_plus.reference import (
    GRADER_TEMPLATE,
    QUERY_TEMPLATE,
    QUERY_TEMPLATE_NO_GET_DOCUMENT,
    parse_judge_response,
)
from olmo_eval.evals.tasks.browsecomp_plus.task import BrowseCompCalibration
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.harness.presets import get_harness_preset
from olmo_eval.runners.io.builders import build_predictions


@pytest.mark.parametrize(
    "prompt,expected",
    [
        (QUERY_TEMPLATE, "ab1002c41cba24ff89fee295905b857f8ed6dd220a51c8748229d8738eb2afef"),
        (
            QUERY_TEMPLATE_NO_GET_DOCUMENT,
            "4a24b6a8b8186eef688cdb4dffa50e305f206546f8f2c6e95d6bc90e48ec53ad",
        ),
        (GRADER_TEMPLATE, "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"),
    ],
)
def test_prompts_match_pinned_upstream_text(prompt, expected):
    assert hashlib.sha256(prompt.encode()).hexdigest() == expected


@pytest.fixture
def judge(monkeypatch):
    client = AsyncMock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=(
            "extracted_final_answer: Paris\nreasoning: Matches.\ncorrect: yes\nconfidence: 80"
        )
    )
    client.__aenter__.return_value = client
    # The OpenAI SDK is an optional extra, so stand in for the module itself.
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda: client))
    return client.responses.create


def make_response(text="Explanation: See [11].\nExact Answer: Paris\nConfidence: 80%"):
    instance = Instance(
        question="Which city?",
        gold_answer="Paris",
        metadata={
            "id": "q1",
            "evidence_docids": ["11", "12"],
        },
    )
    calls = [
        ("search", [{"docid": "11", "snippet": "Paris"}, {"docid": "99", "snippet": "other"}]),
        ("search", [{"docid": "11", "snippet": "Paris"}]),
        ("get_document", {"docid": "12", "text": "A full document"}),
    ]
    turns = []
    for index, (name, result) in enumerate(calls):
        call = ToolCall(id=str(index), function=Function(name=name, arguments="{}"))
        turns += [
            AgentTurn.assistant(tool_calls=[call]),
            AgentTurn.tool(
                results=[
                    ToolResult(tool_call_id=call.id, content=json.dumps(result)),
                ]
            ),
        ]
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.CHAT),
        outputs=[LMOutput(text=text)],
        trajectory=AgentTrajectory(turns=tuple(turns)),
    )


@pytest.mark.parametrize("get_document", [False, True])
def test_presets_roundtrip_and_matching_prompts(get_document):
    suffix = "_get_document" if get_document else ""
    config = get_harness_preset(f"browsecomp_plus{suffix}")
    restored = HarnessConfig.from_dict(config.to_dict())
    assert restored.scaffold == "openai_agents"
    assert restored.scaffold_kwargs == {}
    assert restored.system_prompt is None
    assert [tool.name for tool in restored.resolved_tools] == (
        ["search", "get_document"] if get_document else ["search"]
    )
    assert list(restored.tool_names) == (["search", "get_document"] if get_document else ["search"])
    task = get_task("browsecomp_plus:get_document" if get_document else "browsecomp_plus")
    question = "Which {city}?"
    prompt = QUERY_TEMPLATE if get_document else QUERY_TEMPLATE_NO_GET_DOCUMENT
    assert task.format_request(Instance(question=question)).messages[0]["content"] == prompt.format(
        Question=question
    )
    assert task.config.judge_model == "gpt-4.1"
    assert task.config.judge_max_tokens == 1024


def test_missing_preparation_has_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSECOMP_PLUS_HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="browsecomp_plus.prepare"):
        list(get_task("browsecomp_plus").instances)


def test_load_prepared_data_and_validate_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("BROWSECOMP_PLUS_HOME", str(tmp_path))
    record = {
        "query_id": "007",
        "query": " \tWhich\tcity?\n",
        "answer": "Paris",
        "evidence_docs": [
            {"docid": "0011", "text": "The evidence names Paris.", "url": "https://example.org"},
        ],
        "gold_docs": [],
        "negative_docs": [],
    }
    (tmp_path / DATASET_FILENAME).write_text(json.dumps(record) + "\n")
    task = get_task("browsecomp_plus")
    instances = list(task.instances)
    assert len(instances) == 1
    assert instances[0].metadata["id"] == "007"
    assert instances[0].gold_answer == "Paris"
    assert instances[0].question == record["query"]
    assert instances[0].metadata["evidence_docids"] == ["0011"]
    prompt = task.format_request(instances[0]).messages[0]["content"]
    assert "Which city?" in prompt
    assert "Paris" not in prompt
    assert "https://example.org" not in prompt
    with pytest.raises(ValueError, match="Incomplete"):
        task.process_doc({**record, "evidence_docs": []})


@pytest.mark.anyio
async def test_official_request_and_trajectory_metrics(judge):
    task = get_task("browsecomp_plus")
    response = make_response()
    await task.score_responses([response])
    assert judge.call_args.kwargs == {
        "model": "gpt-4.1",
        "max_output_tokens": 1024,
        "input": GRADER_TEMPLATE.format(
            question="Which city?", response=response.outputs[0].text, correct_answer="Paris"
        ),
    }
    assert response.scores["accuracy"] == 1
    assert response.scores["retrieval_recall"] == 0.5  # get_document doesn't add searched evidence
    assert response.scores["search_calls"] == 2
    assert response.scores["get_document_calls"] == 1
    assert response.scores["citation_precision"] == 1
    assert response.scores["citation_recall"] == 0.5
    prediction = build_predictions([response], task.config.metrics)[0]
    verdict = prediction["model_output"][0]["judge_result"]
    assert verdict["retrieved_docids"] == ["11", "99"]
    assert verdict["confidence"] == 80
    assert verdict["raw_response"]
    assert prediction["trajectory"]
    assert all(metric.scorer.name == metric.name for metric in task.config.metrics)
    assert task.config.metrics[0].scorer.score(response.instance, response.outputs[0]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize(
    "text,limited", [("", False), ("  ", False), ("Forced final answer", True)]
)
async def test_incomplete_responses_keep_retrieval_and_skip_judge(judge, text, limited):
    response = make_response(text)
    response.outputs[0].metadata["max_turns_reached"] = limited
    await get_task("browsecomp_plus").score_responses([response])
    judge.assert_not_called()
    assert response.scores["accuracy"] == 0
    assert response.scores["retrieval_recall"] == 0.5
    assert response.outputs[0].metadata["judge_result"]["parse_error"]


@pytest.mark.anyio
async def test_malformed_verdict_scores_zero_and_is_auditable(judge, caplog):
    judge.return_value = SimpleNamespace(output_text="not a verdict")
    response = make_response()
    await get_task("browsecomp_plus").score_responses([response])
    assert response.scores["accuracy"] == 0
    assert response.outputs[0].metadata["judge_result"]["raw_response"] == "not a verdict"
    assert "could not be parsed" in caplog.text


@pytest.mark.anyio
async def test_api_failure_is_logged_and_raised_by_task(judge, caplog):
    judge.side_effect = RuntimeError("service unavailable")
    with pytest.raises(RuntimeError, match="service unavailable"):
        await get_task("browsecomp_plus").score_responses([make_response()])
    assert "q1" in caplog.text
    assert get_harness_preset("browsecomp_plus").max_hard_failure_rate == 0


@pytest.mark.anyio
async def test_multiple_samples_are_rejected(judge):
    response = make_response()
    response.outputs.append(LMOutput(text="second"))
    with pytest.raises(ValueError, match="one agent trajectory"):
        await get_task("browsecomp_plus").score_responses([response])
    judge.assert_not_called()


@pytest.mark.parametrize("label", ["correct: yes", "**correct:** yes", "**correct**: YES"])
def test_reference_parser_accepts_supported_label_formats(label):
    assert parse_judge_response(label)["correct"] is True
    assert parse_judge_response(label)["parse_error"] is False


def test_reference_calibration_preserves_upstream_last_bin_behavior():
    responses = [make_response() for _ in range(100)]
    for response in responses:
        response.outputs[0].metadata["judge_result"] = {
            "correct": False,
            "confidence": 100,
            "parse_error": False,
        }
    metric = BrowseCompCalibration()
    assert metric.compute(responses) == 0
    assert metric.compute_instance(responses[0]) is None
    assert not metric.supports_pairwise_scorer_fallback()


@pytest.mark.anyio
async def test_citation_averages_condition_on_cited_responses(judge):
    task = get_task("browsecomp_plus")
    responses = [make_response(), make_response("Exact Answer: Paris")]
    await task.score_responses(responses)
    metrics = {metric.name: metric.compute(responses) for metric in task.config.metrics}
    assert metrics["citation_coverage"] == 0.5
    assert metrics["citation_precision"] == 1
    assert metrics["citation_recall"] == 0.5

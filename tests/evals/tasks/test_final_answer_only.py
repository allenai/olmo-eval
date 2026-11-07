"""Final-answer correctness gating and artifact-only rescoring."""

import asyncio
import copy
import json
from dataclasses import replace

import pytest
from click.testing import CliRunner

from olmo_eval.cli.rescore import rescore
from olmo_eval.cli.run.config import RunConfigBuilder
from olmo_eval.common.metrics import (
    AccuracyMetric,
    F1Metric,
    PassAtKMetric,
    PassPowKMetric,
    SafetyErrorMetric,
)
from olmo_eval.common.scorers import ExactMatchScorer, SafetyScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig, get_task
from olmo_eval.evals.tasks.common.final_answer import (
    score_with_answer_region,
    validate_final_answer_only,
)
from olmo_eval.runners.asynq.preparation import configure_task, finalize_task
from olmo_eval.runners.asynq.results import _score_response
from olmo_eval.runners.asynq.types import TaskTracker
from olmo_eval.runners.io.builders import build_predictions, build_requests_from_responses
from olmo_eval.runners.processing.utils import compute_task_hash
from olmo_eval.runners.rescore import rescore_predictions


class PlainTask(Task):
    @property
    def instances(self):
        return iter(())

    def format_request(self, instance):
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)


def plain_task(**kwargs):
    return PlainTask(TaskConfig(name="plain", metrics=(AccuracyMetric(),), **kwargs))


def response(*texts, gold="A"):
    return Response(
        instance=Instance(question="Q", gold_answer=gold, metadata={"id": "item"}),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text=text,
                metadata={"completion_tokens": 9, "prompt_tokens": 3, "finish_reason": "length"},
            )
            for text in texts
        ],
    )


@pytest.mark.parametrize(
    ("text", "expected", "reason", "score"),
    [
        ("<think>a", "", "missing_delimiter", 0),
        ("chemistry (D)", "", "missing_delimiter", 0),
        ("Cis-propenyl", "", "missing_delimiter", 0),
        ("A", "", "missing_delimiter", 0),
        ("", "", "missing_delimiter", 0),
        ("reason</think>", "", "empty_final_answer", 0),
        ("reason</think> \n\t", " \n\t", "empty_final_answer", 0),
        ("reason</think>\n A \t", "\n A \t", "final_answer", 1),
        ("reason</think>B</think> A", " A", "final_answer", 1),
        ("reason</think>A</think>\n", "\n", "empty_final_answer", 0),
    ],
)
def test_policy_and_repeated_processing(text, expected, reason, score):
    task = plain_task(
        reward_final_answer_only=True,
        strip_thinking=True,
        strip_unclosed_thinking=True,
        thinking_prefilled=True,
    )
    resp = response(text)
    for _ in range(2):
        task.strip_thinking_traces([resp])
        asyncio.run(task.score_responses([resp]))
        out = resp.outputs[0]
        assert out.text == expected
        assert out.metadata["original_text"] == text
        assert out.metadata["final_answer_gate"] == {
            "applied": True,
            "reason": reason,
            "zeroed": reason != "final_answer",
        }
        assert resp.scores == {"exact_match": score}
        assert "scoring_errors" not in out.metadata
    row = build_predictions([resp], metrics=task.config.metrics)[0]["model_output"][0]
    assert row["completion_tokens"] == 9
    assert row["prompt_tokens"] == 3
    assert row["finish_reason"] == "length"
    assert row["original_text"] == text
    assert row["final_answer_gate"]["reason"] == reason


def test_default_and_explicit_false_keep_legacy_behavior_and_identity():
    task = configure_task("gpqa_diamond:cot")
    legacy = task.config.to_dict()
    assert "reward_final_answer_only" not in legacy
    false = configure_task("gpqa_diamond:cot", {"reward_final_answer_only": False})
    assert false.config.to_dict() == legacy
    gated = configure_task("gpqa_diamond:cot", {"reward_final_answer_only": True})
    assert gated.config.to_dict()["reward_final_answer_only"] is True
    assert compute_task_hash(legacy) != compute_task_hash(gated.config.to_dict())
    for raw, gold in [("a", "A"), ("chemistry (D)", "D"), ("Cis-propenyl", "C")]:
        resp = response(raw, gold=gold)
        asyncio.run(score_with_answer_region(task, [resp]))
        assert resp.scores["exact_match"] == 1
        asyncio.run(score_with_answer_region(gated, [resp]))
        assert resp.scores["exact_match"] == 0


@pytest.mark.parametrize("metric", [AccuracyMetric(), F1Metric()])
def test_empty_answer_cannot_match_empty_gold(metric):
    task = plain_task(reward_final_answer_only=True)
    task.config.metrics = (metric,)
    resp = response("reason</think>", gold="")
    asyncio.run(task.score_responses([resp]))
    assert list(resp.scores.values()) == [0]


def test_missing_answer_skips_extraction_and_scorer(monkeypatch):
    task = plain_task(reward_final_answer_only=True)

    def fail(*args):
        raise AssertionError("A missing answer must not invoke extraction or scoring")

    monkeypatch.setattr(task, "_extract_answers", fail)
    monkeypatch.setattr(ExactMatchScorer, "score", fail)
    resp = response("unfinished")
    asyncio.run(task.score_responses([resp]))
    assert resp.scores["exact_match"] == 0


@pytest.mark.parametrize("aggregation", ["first", "max"])
def test_samples_and_denominators(aggregation):
    task = plain_task(reward_final_answer_only=True, output_score_aggregation=aggregation)
    task.config.metrics = (
        AccuracyMetric(),
        PassAtKMetric(k=1, scorer=ExactMatchScorer),
        PassPowKMetric(k=1, scorer=ExactMatchScorer),
    )
    responses = [response("A", "reason</think>A"), response("unfinished", "</think>")]
    asyncio.run(task.score_responses(responses))
    assert [o.metadata["score:exact_match"] for r in responses for o in r.outputs] == [0, 1, 0, 0]
    # Each instance needs its own id for pass metrics.
    responses[1].instance.metadata["id"] = "second"
    metrics = task.compute_metrics(responses)
    assert metrics["accuracy"]["exact_match"] == (0.5 if aggregation == "max" else 0)
    assert metrics["pass_at_1"]["exact_match"] == 0.25
    assert metrics["pass_pow_1"]["exact_match"] == 0.25


@pytest.mark.parametrize("spec", ["gpqa_diamond:mc", "gpqa_diamond:bpb", "math500"])
def test_unsupported_tasks_fail_before_dataset_access(spec):
    with pytest.raises(ValueError, match="reward_final_answer_only.*unsupported"):
        configure_task(spec, {"reward_final_answer_only": True})


@pytest.mark.parametrize("metric", [SafetyErrorMetric(), AccuracyMetric(scorer=SafetyScorer)])
def test_unsupported_metrics_are_rejected_even_for_missing_answers(metric):
    task = plain_task(reward_final_answer_only=True)
    task.config.metrics = (metric,)
    with pytest.raises(ValueError, match="unsupported"):
        asyncio.run(task.score_responses([response("unfinished")]))


def test_subclasses_do_not_inherit_scorer_authorization():
    class Inverted(ExactMatchScorer):
        def score(self, instance, output):
            return 1 - super().score(instance, output)

    task = plain_task(reward_final_answer_only=True)
    task.config.metrics = (AccuracyMetric(scorer=Inverted),)
    with pytest.raises(ValueError, match="Inverted"):
        validate_final_answer_only(task)


@pytest.mark.parametrize("entrypoint", ["live", "finalize"])
def test_runner_rejects_custom_scoring_override(entrypoint):
    class Custom(PlainTask):
        async def score_responses(self, responses, context=None):
            raise AssertionError("Runner bypassed the gate")

    task = Custom(
        TaskConfig(name="custom", metrics=(AccuracyMetric(),), reward_final_answer_only=True)
    )
    resp = response("A")
    with pytest.raises(ValueError, match="custom score_responses"):
        if entrypoint == "live":
            asyncio.run(_score_response(task, resp, None))
        else:
            tracker = TaskTracker(
                model_name="m",
                spec="custom",
                task=task,
                total_instances=1,
                completed_count=1,
                responses={0: resp},
            )
            asyncio.run(finalize_task(tracker))


def test_runner_keeps_gated_zero_in_denominator():
    task = plain_task(reward_final_answer_only=True)
    responses = [response("A"), response("</think>A")]
    tracker = TaskTracker(
        model_name="m",
        spec="plain",
        task=task,
        total_instances=2,
        completed_count=2,
        responses=dict(enumerate(responses)),
    )
    result = asyncio.run(finalize_task(tracker))
    assert result.num_instances == 2
    assert result.metrics["accuracy"]["exact_match"] == 0.5
    assert not result.error


def saved_rows():
    task = get_task("gpqa_diamond:cot")
    resp = response("reason</think>Therefore, the answer is (A)", "chemistry (D)", gold="A")
    # Ordinary saved artifacts only preserve original_text when stripping occurred.
    asyncio.run(score_with_answer_region(task, [resp]))
    return build_predictions([resp], task.config.metrics), build_requests_from_responses(
        [resp], task.config.name
    )


def test_offline_rescoring_requires_originals_and_clears_old_scores():
    task = configure_task("gpqa_diamond:cot", {"reward_final_answer_only": True})
    predictions, requests = saved_rows()
    untouched = copy.deepcopy(predictions)
    with pytest.raises(ValueError, match="Original text unavailable"):
        asyncio.run(rescore_predictions(task, predictions, requests))
    result = asyncio.run(rescore_predictions(task, predictions, requests, text_is_original=True))
    assert result["num_instances"] == 1
    rows = result["predictions"][0]["model_output"]
    assert [r["sample_metrics"]["exact_match"]["exact_match"] for r in rows] == [1, 0]
    assert rows[0]["original_text"].startswith("reason</think>")
    assert rows[1]["original_text"] == "chemistry (D)"
    assert predictions == untouched
    again = asyncio.run(rescore_predictions(task, result["predictions"], requests))
    assert again["predictions"] == result["predictions"]


def test_offline_cli_and_separate_variant_identity(tmp_path):
    predictions, requests = saved_rows()
    pred = tmp_path / "predictions.jsonl"
    req = tmp_path / "requests.jsonl"
    pred.write_text("\n".join(map(json.dumps, predictions)))
    req.write_text("\n".join(map(json.dumps, requests)))
    output = tmp_path / "rescored"
    args = [
        "--task",
        "gpqa_diamond:cot",
        "--predictions",
        str(pred),
        "--requests",
        str(req),
        "--output-dir",
        str(output),
        "--text-is-original",
    ]
    runner = CliRunner()
    default = runner.invoke(rescore, args)
    assert default.exit_code == 0, default.output
    gated = runner.invoke(rescore, args + ["--reward-final-answer-only"])
    assert gated.exit_code == 0, gated.output
    assert len(list(output.glob("*-results.json"))) == 2
    assert json.loads(default.output)["task_hash"] != json.loads(gated.output)["task_hash"]
    duplicate = runner.invoke(rescore, args)
    assert duplicate.exit_code != 0
    assert "already exist" in duplicate.output


def test_existing_strip_can_precede_gate():
    task = plain_task(strip_thinking=True)
    resp = response("reason</think>\n A ")
    task.strip_thinking_traces([resp])
    task.config = replace(task.config, reward_final_answer_only=True)
    asyncio.run(task.score_responses([resp]))
    assert resp.outputs[0].text == "\n A "
    assert resp.scores["exact_match"] == 1


def test_non_thinking_default_still_scores_normally():
    task = plain_task()
    resp = response("A")
    asyncio.run(score_with_answer_region(task, [resp]))
    assert resp.scores["exact_match"] == 1
    assert "final_answer_gate" not in resp.outputs[0].metadata


def test_real_scoring_error_is_not_a_gate_zero(monkeypatch):
    task = plain_task(reward_final_answer_only=True)

    def unavailable(*args):
        raise RuntimeError("scorer unavailable")

    monkeypatch.setattr(ExactMatchScorer, "score", unavailable)
    resp = response("reason</think>A")
    with pytest.raises(RuntimeError, match="scorer unavailable"):
        asyncio.run(_score_response(task, resp, None))
    assert "exact_match" not in resp.scores
    assert resp.outputs[0].metadata["final_answer_gate"]["zeroed"] is False


def test_repeat_scoring_replaces_stale_errors():
    task = plain_task(reward_final_answer_only=True)
    resp = response("unfinished")
    resp.outputs[0].metadata["scoring_errors"] = {"exact_match": {"message": "old failure"}}
    resp.scores["obsolete"] = 1
    asyncio.run(task.score_responses([resp]))
    assert resp.scores == {"exact_match": 0}
    assert "scoring_errors" not in resp.outputs[0].metadata
    assert task.compute_metrics([resp])["accuracy"]["exact_match"] == 0


@pytest.mark.parametrize(
    "bad", ["duplicates", "missing_request", "label", "task", "outputs", "tools"]
)
def test_offline_invalid_artifacts_fail_clearly(bad):
    task = configure_task("gpqa_diamond:cot", {"reward_final_answer_only": True})
    predictions, requests = saved_rows()
    if bad == "duplicates":
        predictions *= 2
    elif bad == "missing_request":
        requests = []
    elif bad == "label":
        requests[0]["label"] = "B"
    elif bad == "task":
        requests[0]["task_name"] = "another_task"
    elif bad == "outputs":
        predictions[0]["model_output"] = []
    elif bad == "tools":
        requests[0]["request"]["tools"] = [{"name": "tool"}]
    with pytest.raises(ValueError):
        asyncio.run(rescore_predictions(task, predictions, requests, text_is_original=True))


def test_offline_plain_text_and_exact_gold_round_trip():
    task = plain_task(reward_final_answer_only=True)
    resp = response("reason</think>the answer", gold="the answer")
    # Saving a choice index must not lose the actual scorer target.
    resp.instance.metadata["gold_idx"] = 0
    resp.instance = replace(resp.instance, choices=("the answer", "other"))
    asyncio.run(task.score_responses([resp]))
    predictions = build_predictions([resp], task.config.metrics)
    requests = build_requests_from_responses([resp], task.config.name)
    result = asyncio.run(rescore_predictions(task, predictions, requests))
    assert result["metrics"]["accuracy"]["exact_match"] == 1
    del predictions[0]["gold_answer"]
    with pytest.raises(ValueError, match="Ambiguous saved label"):
        asyncio.run(rescore_predictions(task, predictions, requests))


def test_no_outputs_is_not_an_empty_answer():
    task = plain_task(reward_final_answer_only=True)
    with pytest.raises(ValueError, match="requires an output"):
        asyncio.run(task.score_responses([response()]))


def test_ordinary_cli_override_is_a_boolean():
    config = RunConfigBuilder(
        model="mock",
        task=("gpqa_diamond:cot",),
        output_dir="unused",
        cli_task_overrides={"gpqa_diamond:cot": ["reward_final_answer_only=true"]},
    ).build()
    task = configure_task("gpqa_diamond:cot", config.task_overrides["gpqa_diamond:cot"])
    assert task.config.reward_final_answer_only is True

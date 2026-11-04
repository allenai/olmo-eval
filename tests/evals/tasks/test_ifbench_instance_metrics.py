"""Prediction exports preserve the evidence behind all four IF accuracies."""

import json

import pytest

from olmo_eval.common.metrics import (
    IFEvalInstLooseAccuracy,
    IFEvalInstStrictAccuracy,
    IFEvalPromptLooseAccuracy,
    IFEvalPromptStrictAccuracy,
)
from olmo_eval.common.metrics.predictions import augment_prediction_instance_metrics
from olmo_eval.common.scorers import IFEvalScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.runners.io import build_predictions, write_predictions_jsonl

METRICS = (
    IFEvalPromptStrictAccuracy(),
    IFEvalPromptLooseAccuracy(),
    IFEvalInstStrictAccuracy(),
    IFEvalInstLooseAccuracy(),
)


def scored_response(text="Extras: 5\nI have 3 apples and 4 oranges.", counts=(2, 3)):
    instance = Instance(
        question="Follow the number constraints.",
        metadata={
            "instruction_id_list": ["count:numbers"] * len(counts),
            "kwargs": [{"N": count} for count in counts],
        },
    )
    output = LMOutput(text=text)
    score = IFEvalScorer().score(instance, output)
    return Response(
        instance=instance,
        request=LMRequest(RequestType.COMPLETION, prompt=instance.question),
        outputs=[output],
        scores={"ifeval": score},
    )


def test_builder_and_jsonl_preserve_outcomes_and_weighted_aggregates(tmp_path):
    responses = [scored_response(), scored_response("3 apples and 4 oranges", counts=(2,))]
    expected = [0.5, 1.0, 2 / 3, 1.0]
    assert [m.compute(responses) for m in METRICS] == pytest.approx(expected)
    predictions = build_predictions(responses, METRICS)
    write_predictions_jsonl(str(tmp_path), "ifeval", predictions, "mock")
    rows = [json.loads(line) for line in next(tmp_path.rglob("*.jsonl")).read_text().splitlines()]
    assert rows[0]["model_output"][0]["scorer_results"]["ifeval"] == {
        "strict": [False, True],
        "loose": [True, True],
    }
    assert [rows[0]["instance_metrics"][m.name]["ifeval"] for m in METRICS] == [0, 1, 0.5, 1]
    for mode in ("strict", "loose"):
        flags = [r["model_output"][0]["scorer_results"]["ifeval"][mode] for r in rows]
        prompt = sum(bool(f) and all(f) for f in flags) / len(rows)
        instruction = sum(sum(f) for f in flags) / sum(map(len, flags))
        assert prompt == METRICS[mode == "loose"].compute(responses)
        assert instruction == METRICS[2 + (mode == "loose")].compute(responses)
    assert sum(r["instance_metrics"]["inst_level_strict_acc"]["ifeval"] for r in rows) / 2 == 0.75
    # Reconstruct without running a verifier, even if stale exact keys were present.
    for row in rows:
        expected_metrics = row["instance_metrics"]
        row["instance_metrics"] = {m.name: {"ifeval": 0.25} for m in METRICS}
        augment_prediction_instance_metrics(row, METRICS)
        assert row["instance_metrics"] == {m.name: expected_metrics[m.name] for m in METRICS}
    assert [m.compute(responses) for m in METRICS] == pytest.approx(expected)


@pytest.mark.parametrize("metadata", [None, {}, {"ifeval": {}}, {"ifeval": None}])
def test_missing_outcomes_only_allow_loose_prompt(metadata):
    response = scored_response()
    response.outputs[0].metadata = metadata
    prediction = build_predictions([response], METRICS)[0]
    assert prediction["instance_metrics"] == {
        "ifeval": {"ifeval": 1.0},
        "prompt_level_loose_acc": {"ifeval": 1.0},
    }
    assert all(m.compute_instance(response) is None for m in METRICS)


def test_no_output_or_score_remains_unavailable():
    response = scored_response()
    response.outputs = []
    response.scores = {}
    assert build_predictions([response], METRICS)[0]["instance_metrics"] == {}


def test_multiple_outputs_follow_first_output_aggregate_semantics():
    response = scored_response()
    response.outputs.append(scored_response("no numbers").outputs[0])
    prediction = build_predictions([response], METRICS)[0]
    assert len(prediction["model_output"]) == 2
    assert prediction["model_output"][1]["scorer_results"]["ifeval"]["strict"] == [False, False]
    for metric in METRICS:
        assert prediction["instance_metrics"][metric.name]["ifeval"] == metric.compute([response])
    augment_prediction_instance_metrics(prediction, METRICS)
    assert prediction["instance_metrics"]["inst_level_strict_acc"]["ifeval"] == 0.5


@pytest.mark.parametrize(
    "result", [{"strict": [], "loose": []}, {"strict": [False], "loose": [False]}]
)
def test_empty_or_failed_outcomes_are_zero_not_missing(result):
    response = scored_response()
    response.outputs[0].metadata["ifeval"] = result
    prediction = build_predictions([response], METRICS)[0]
    assert all(prediction["instance_metrics"][m.name]["ifeval"] == 0 for m in METRICS)


@pytest.mark.parametrize("bad", [None, "false", [1], ["false"], {}])
def test_partial_or_malformed_outcomes_do_not_fall_back_to_loose(bad):
    response = scored_response()
    response.outputs[0].metadata["ifeval"] = {"strict": bad, "loose": [True, False]}
    prediction = build_predictions([response], METRICS)[0]
    assert "prompt_level_strict_acc" not in prediction["instance_metrics"]
    assert "inst_level_strict_acc" not in prediction["instance_metrics"]
    assert prediction["instance_metrics"]["prompt_level_loose_acc"] == {"ifeval": 0}
    assert prediction["instance_metrics"]["inst_level_loose_acc"] == {"ifeval": 0.5}
    augment_prediction_instance_metrics(prediction, METRICS)
    assert "prompt_level_strict_acc" not in prediction["instance_metrics"]


@pytest.mark.parametrize("flat", [False, True])
def test_legacy_mislabeled_exact_values_are_not_trusted(flat):
    values = {"ifeval": 1.0, **{m.name: 1.0 for m in METRICS}}
    prediction = {
        "instance_metrics": values if flat else {k: {"ifeval": v} for k, v in values.items()}
    }
    augment_prediction_instance_metrics(prediction, METRICS)
    assert set(prediction["instance_metrics"]) == {"ifeval", "prompt_level_loose_acc"}
    assert prediction["instance_metrics"]["prompt_level_loose_acc"] == {"ifeval": 1.0}


def test_builder_preserves_errors_and_other_scorer_results_without_metrics():
    response = scored_response()
    result = response.outputs[0].metadata["ifeval"]
    result["errors"] = {"unknown:instruction": "KeyError: unknown"}
    response.outputs[0].metadata["scorer_results"] = {"other": {"value": 3}}
    prediction = build_predictions([response])[0]
    assert prediction["model_output"][0]["scorer_results"] == {
        "other": {"value": 3},
        "ifeval": result,
    }
    assert response.outputs[0].metadata["scorer_results"] == {"other": {"value": 3}}


def test_metric_configuration_identity_is_unchanged():
    for metric in METRICS:
        assert metric.to_dict() == {
            "type": type(metric).__name__,
            "name": metric.name,
            "scorer": "IFEvalScorer",
        }

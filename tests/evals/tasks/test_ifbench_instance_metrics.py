"""Saved IFBench metrics retain strict/loose and instruction semantics."""

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
from olmo_eval.runners.io.builders import build_predictions

METRICS = (
    IFEvalPromptStrictAccuracy(),
    IFEvalPromptLooseAccuracy(),
    IFEvalInstStrictAccuracy(),
    IFEvalInstLooseAccuracy(),
)


def scored_response(text, counts=(2, 3)):
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


def test_saved_metrics_distinguish_strict_loose_and_partial_instruction_success():
    response = scored_response("Extras: 5\nI have 3 apples and 4 oranges.")
    assert response.outputs[0].metadata["ifeval"] == {
        "strict": [False, True],
        "loose": [True, True],
    }
    assert response.scores["ifeval"] == 1.0
    prediction = build_predictions([response], METRICS)[0]
    for metric, value in zip(METRICS, (0.0, 1.0, 0.5, 1.0), strict=True):
        assert metric.compute_instance(response) == value
        assert prediction["instance_metrics"][metric.name]["ifeval"] == value


def test_instruction_aggregate_remains_weighted_by_instruction_count():
    responses = [
        scored_response("Extras: 5\nI have 3 apples and 4 oranges."),
        scored_response("I have 3 apples and 4 oranges.", counts=(2,)),
    ]
    metric = IFEvalInstStrictAccuracy()
    assert [metric.compute_instance(response) for response in responses] == [0.5, 1.0]
    assert metric.compute(responses) == pytest.approx(2 / 3)
    assert IFEvalPromptStrictAccuracy().compute(responses) == 0.5


@pytest.mark.parametrize("metadata", [{}, {"ifeval": {}}, {"ifeval": {"strict": [], "loose": []}}])
def test_missing_instruction_flags_do_not_fabricate_scores(metadata):
    response = scored_response("I have 3 apples and 4 oranges.")
    response.outputs[0].metadata = metadata
    response.scores["ifeval"] = 1.0
    prediction = build_predictions([response], METRICS)[0]
    for metric in METRICS:
        assert metric.compute_instance(response) is None
        if metric.name == "prompt_level_loose_acc":
            assert prediction["instance_metrics"][metric.name]["ifeval"] == 1.0
        else:
            assert metric.name not in prediction["instance_metrics"]


def test_no_output_has_no_instance_metric():
    response = scored_response("I have 3 apples and 4 oranges.")
    response.outputs = []
    assert all(metric.compute_instance(response) is None for metric in METRICS)


def test_scorer_only_legacy_predictions_do_not_fill_all_metrics_with_loose_score():
    prediction = {"instance_metrics": {"ifeval": {"ifeval": 1.0}}, "model_output": []}
    augment_prediction_instance_metrics(prediction, METRICS)
    assert prediction["instance_metrics"] == {
        "ifeval": {"ifeval": 1.0},
        "prompt_level_loose_acc": {"ifeval": 1.0},
    }


def test_existing_exact_values_are_preserved_without_original_response():
    prediction = {"instance_metrics": {"prompt_level_strict_acc": {"ifeval": 0.0}}}
    augment_prediction_instance_metrics(prediction, METRICS)
    assert prediction["instance_metrics"] == {"prompt_level_strict_acc": {"ifeval": 0.0}}


def test_metric_configuration_identity_is_unchanged():
    for metric in METRICS:
        assert metric.to_dict() == {
            "type": type(metric).__name__,
            "name": metric.name,
            "scorer": "IFEvalScorer",
        }

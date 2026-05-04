from __future__ import annotations

from olmo_eval.common.metrics import LogprobPerTokenMCAccuracyMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.runners.io.builders import build_predictions


def test_build_predictions_includes_scoring_errors() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="out",
                extracted_answer="out",
                metadata={
                    "scoring_errors": {
                        "code_exec": {
                            "phase": "execution",
                            "type": "RuntimeError",
                            "message": "boom",
                        }
                    },
                    "score:code_exec": 0.0,
                },
            )
        ],
        scores={"code_exec": 0.0},
    )

    predictions = build_predictions([response])

    assert predictions[0]["instance_metrics"] == {"code_exec": {"code_exec": 0.0}}
    assert predictions[0]["model_output"][0]["scoring_errors"] == {
        "code_exec": {
            "phase": "execution",
            "type": "RuntimeError",
            "message": "boom",
        }
    }
    assert predictions[0]["model_output"][0]["sample_metrics"] == {"code_exec": {"code_exec": 0.0}}


def test_build_predictions_includes_sample_metrics_for_multiple_outputs() -> None:
    response = Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[
            LMOutput(
                text="wrong",
                extracted_answer="B",
                metadata={"score:exact_match": 0.0},
            ),
            LMOutput(
                text="right",
                extracted_answer="A",
                metadata={"score:exact_match": 1.0},
            ),
        ],
        scores={"exact_match": 0.0},
    )

    predictions = build_predictions([response])

    assert predictions[0]["model_output"][0]["sample_metrics"] == {
        "exact_match": {"exact_match": 0.0}
    }
    assert predictions[0]["model_output"][1]["sample_metrics"] == {
        "exact_match": {"exact_match": 1.0}
    }


def test_build_predictions_materializes_exact_mc_accuracy_metric_keys() -> None:
    response = Response(
        instance=Instance(
            question="Q",
            choices=("A", "B"),
            metadata={"gold_idx": 1},
        ),
        request=LMRequest(request_type=RequestType.LOGLIKELIHOOD, prompt="Q"),
        outputs=[
            LMOutput(
                text="A",
                logprobs=[{"token": "A", "logprob": -4.0}],
            ),
            LMOutput(
                text="B",
                logprobs=[{"token": "B", "logprob": -1.0}],
            ),
        ],
        scores={"logprob": -1.0},
    )

    predictions = build_predictions([response], metrics=(LogprobPerTokenMCAccuracyMetric(),))

    assert predictions[0]["instance_metrics"]["logprob"]["logprob"] == -1.0
    assert predictions[0]["instance_metrics"]["accuracy"]["logprob"] == 1.0

from __future__ import annotations

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

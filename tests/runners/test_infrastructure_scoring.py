"""Tests that sandbox infrastructure failures invalidate task results."""

from __future__ import annotations

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.runners.asynq.preparation import compute_task_metrics


class _Task:
    config = TaskConfig(name="test", data_source="test/data")

    def compute_metrics(self, _responses: list[Response]) -> dict[str, dict[str, float]]:
        return {"accuracy": {"code_exec": 0.0}}


def _response(error: dict[str, str]) -> Response:
    return Response(
        instance=Instance(question="question", gold_answer="answer"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="question"),
        outputs=[
            LMOutput(
                text="output",
                metadata={"scoring_errors": {"code_exec": error}},
            )
        ],
    )


def test_infrastructure_scoring_error_marks_task_incomplete() -> None:
    result = compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        [_response({"type": "SandboxInfrastructureError", "infrastructure": "true"})],
        {},
        1,
        1.0,
    )

    assert result.error == "Incomplete: 1 output score(s) failed due to sandbox infrastructure"
    assert result.to_dict()["error"] == result.error


def test_model_scoring_error_does_not_mark_task_incomplete() -> None:
    result = compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        [_response({"type": "AssertionError", "message": "tests failed"})],
        {},
        1,
        1.0,
    )

    assert result.error is None

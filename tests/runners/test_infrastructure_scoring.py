"""Tests that scoring failures invalidate task results instead of counting as zeros."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.runners.asynq.preparation import compute_task_metrics
from olmo_eval.runners.asynq.results import aggregate_results
from tests.evals.tasks.judge_stubs import score_like_the_runner


class _Task:
    config = TaskConfig(name="test", data_source="test/data")

    def compute_metrics(self, _responses: list[Response]) -> dict[str, dict[str, float]]:
        return {"accuracy": {"code_exec": 0.0}}


def _response(*errors: dict[str, dict[str, str]] | None) -> Response:
    """A response with one output per entry, each carrying those scoring errors (or none)."""
    return Response(
        instance=Instance(question="question", gold_answer="answer"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="question"),
        outputs=[
            LMOutput(text="output", metadata={} if e is None else {"scoring_errors": e})
            for e in errors
        ],
    )


def _result(*responses: Response):
    return compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        list(responses),
        {},
        len(responses),
        1.0,
    )


def test_infrastructure_scoring_error_marks_task_incomplete() -> None:
    result = compute_task_metrics(
        "test:variant",
        _Task(),  # type: ignore[arg-type]
        [
            _response(
                {"code_exec": {"type": "SandboxInfrastructureError", "infrastructure": "true"}}
            )
        ],
        {},
        1,
        1.0,
    )

    assert result.error == "Incomplete: 1 output score(s) failed due to sandbox infrastructure"
    assert result.to_dict()["error"] == result.error


def test_an_unexpected_scoring_error_marks_task_incomplete() -> None:
    result = _result(_response({"code_exec": {"type": "ValueError", "message": "bad gold"}}))

    assert result.error == (
        "Incomplete: 1 output(s) could not be scored; first error: ValueError: bad gold"
    )


def test_an_output_lost_to_infrastructure_is_not_also_counted_as_unscored() -> None:
    # A judge that gives up records its own flagged error, and the runner adds an
    # unflagged response-level entry for the exception the task then raised.
    errors = {
        "judge": {"type": "ScoringIncompleteError", "infrastructure": "true"},
        "__response__": {"type": "ScoringIncompleteError", "message": "judge unavailable"},
    }

    result = _result(_response(errors))

    assert result.error == "Incomplete: 1 output score(s) failed due to sandbox infrastructure"


def test_both_kinds_of_failure_are_reported() -> None:
    result = _result(
        _response(
            {"code_exec": {"type": "SandboxTransportError", "infrastructure": "true"}},
            {"code_exec": {"type": "KeyError", "message": "'answer'"}},
            {"code_exec": {"type": "KeyError", "message": "'answer'"}},
            None,
        )
    )

    assert result.error == (
        "Incomplete: 1 output score(s) failed due to sandbox infrastructure "
        "Incomplete: 2 output(s) could not be scored; first error: KeyError: 'answer'"
    )


def test_a_task_without_scoring_errors_is_complete() -> None:
    assert _result(_response(None, {})).error is None


class _RaisingTask(_Task):
    """A task whose scoring raises, as a misconfigured or buggy scorer does."""

    def _get_scorers(self):
        return {"code_exec": object()}

    async def score_responses(self, responses, context=None):
        raise ValueError("Multi-IF output has no per-turn replies")


@pytest.mark.anyio
async def test_a_raising_scorer_withholds_the_tasks_metrics() -> None:
    # The runner records an exception from score_responses as a 0.0 placeholder. The task
    # must come out failed, and its placeholder metrics must not be published.
    scored = await score_like_the_runner(_RaisingTask(), _response(None))

    result = compute_task_metrics("test:variant", _RaisingTask(), [scored], {}, 1, 1.0)  # type: ignore[arg-type]
    provider = SimpleNamespace(alias="m", model="org/m", kind="vllm_server", to_dict=dict)
    aggregated = aggregate_results(
        results={"test:variant": result},
        expanded_tasks=["test:variant"],
        task_specs=["test:variant"],
        provider_config=provider,
        attention_backend=None,
    )

    assert result.error.startswith("Incomplete: 1 output(s) could not be scored")
    assert "no per-turn replies" in result.error
    assert "metrics" not in aggregated["tasks"]["test:variant"]

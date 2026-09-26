"""Tests for per-task generation counts (cap-hit / empty / unclosed think).

A reasoning model whose budget runs out mid-trace scores near zero on that
instance, and nothing errors, so these counts must reach every surface a score
is read from: the task completion line, the summary table, metrics.json, the
stored task row and suite averages.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig, get_task
from olmo_eval.runners.asynq.preparation import compute_task_metrics, finalize_task
from olmo_eval.runners.asynq.results import _report_task_completion, aggregate_results
from olmo_eval.runners.asynq.types import TaskTracker
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.generation_counts import (
    count_generations,
    format_generation_counts,
    sum_generation_counts,
)
from olmo_eval.runners.processing.metrics import build_single_model_metrics, log_summary
from olmo_eval.storage.base import convert_runner_results

SPEC = "ifeval_ood"
CLEAN_METRICS = {"accuracy": {"exact_match": 0.5}}


def _response(
    *outputs: tuple[str, str | None],
    request_type: RequestType = RequestType.CHAT,
) -> Response:
    """A response whose outputs are (text, finish_reason) pairs."""
    return Response(
        instance=Instance(question="q", gold_answer="a"),
        request=LMRequest(request_type=request_type, prompt="q"),
        outputs=[
            LMOutput(
                text=text,
                metadata={} if finish_reason is None else {"finish_reason": finish_reason},
            )
            for text, finish_reason in outputs
        ],
    )


def _stripped(responses: list[Response]) -> list[Response]:
    """Strip thinking traces with a real reasoning-regime task, as the runner does."""
    get_task(SPEC).strip_thinking_traces(responses)
    return responses


def _reasoning_batch() -> list[Response]:
    """One of each outcome, in the shapes a thinking model produces."""
    return _stripped(
        [
            # Clean: trace closed, answer after it.
            _response(("<think>plan</think>\nFinal answer.", "stop")),
            # Budget ran out mid-trace; the template opened the trace, so the
            # output carries no <think> of its own.
            _response(("step 1, step 2, step 3", "length")),
            # The model opened a trace itself and was cut off inside it.
            _response(("<think>still thinking", "length")),
            # Trace closed, nothing after it.
            _response(("<think>plan</think>\n  ", "stop")),
        ]
    )


class _StubTask:
    config = TaskConfig(name=SPEC, data_source="test/data", metrics=(AccuracyMetric(),))

    def strip_thinking_traces(self, responses: list[Response]) -> None:
        get_task(SPEC).strip_thinking_traces(responses)

    async def score_responses(self, responses: list[Response]) -> list[Response]:
        return responses

    def compute_metrics(self, responses: list[Response]) -> dict[str, dict[str, float]]:
        return CLEAN_METRICS


def _task_result(responses: list[Response]):
    return compute_task_metrics(
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        scored_responses=responses,
        failed_instances={},
        total_instances=len(responses),
        duration_seconds=1.0,
    )


def _aggregate(result, spec: str = SPEC) -> dict[str, Any]:
    provider_config = SimpleNamespace(
        alias="m", model="org/m", kind="vllm_server", to_dict=lambda: {}
    )
    return aggregate_results(
        results={spec: result},
        expanded_tasks=[spec],
        task_specs=[spec],
        provider_config=provider_config,
        attention_backend=None,
    )


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_counts_each_outcome() -> None:
    counts = count_generations(_reasoning_batch())

    assert counts == {
        "generations": 4,
        "cap_hit": 2,
        "empty": 1,
        "unclosed_think": 1,
        "finish_reason_unknown": 0,
    }


def test_closed_trace_opened_by_the_template_is_not_unclosed() -> None:
    """With the opener in the prompt, a closed trace shows only </think>."""
    counts = count_generations(_stripped([_response(("reasoning</think>\nAnswer.", "stop"))]))

    assert counts is not None
    assert counts["unclosed_think"] == 0
    assert counts["empty"] == 0


def test_missing_finish_reason_is_counted_as_unknown_not_as_complete() -> None:
    counts = count_generations([_response(("answer", None), ("", None))])

    assert counts is not None
    assert counts["finish_reason_unknown"] == 2
    assert counts["cap_hit"] == 0
    assert counts["empty"] == 1


def test_every_sample_of_a_multi_sample_request_is_counted() -> None:
    counts = count_generations([_response(("a", "stop"), ("b", "length"), ("c", "length"))])

    assert counts is not None
    assert counts["generations"] == 3
    assert counts["cap_hit"] == 2


def test_loglikelihood_tasks_have_no_counts() -> None:
    responses = [_response((" yes", None), request_type=RequestType.LOGLIKELIHOOD)]

    assert count_generations(responses) is None


def test_format_lists_every_count_and_flags_unknown_only_when_present() -> None:
    assert format_generation_counts(count_generations(_reasoning_batch())) == (
        "cap-hit 2/4, empty 1, unclosed-think 1"
    )
    assert "finish-reason unknown 1" in format_generation_counts(
        count_generations([_response(("answer", None))])
    )
    assert format_generation_counts(None) == "-"


# ---------------------------------------------------------------------------
# Every surface a score is read from
# ---------------------------------------------------------------------------


def test_compute_task_metrics_attaches_counts() -> None:
    result = _task_result(_reasoning_batch())

    assert result.generation_counts is not None
    assert result.generation_counts["cap_hit"] == 2
    assert result.to_dict()["generation_counts"] == result.generation_counts


def test_finalize_task_attaches_counts_after_stripping() -> None:
    """finalize_task strips traces itself, so a closed empty answer counts as empty."""
    raw = [
        _response(("<think>plan</think>\n  ", "stop")),
        _response(("step 1, step 2", "length")),
    ]
    tracker = TaskTracker(
        model_name="m",
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        total_instances=len(raw),
        responses=dict(enumerate(raw)),
    )

    result = asyncio.run(finalize_task(tracker))

    assert result.generation_counts is not None
    assert result.generation_counts["empty"] == 1
    assert result.generation_counts["cap_hit"] == 1


def test_task_completion_line_reports_counts(caplog) -> None:
    caplog.set_level(logging.INFO)

    _report_task_completion("m", _task_result(_reasoning_batch()))

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "4/4 instances saved; cap-hit 2/4, empty 1, unclosed-think 1" in logged


def test_summary_table_reports_counts_beside_the_score(capsys) -> None:
    log_summary(_aggregate(_task_result(_reasoning_batch())))

    out = capsys.readouterr().out
    assert "Generations" in out
    assert "cap-hit 2/4" in out
    assert "0.5000" in out


def test_metrics_json_carries_counts() -> None:
    results = _aggregate(_task_result(_reasoning_batch()))

    entry = build_single_model_metrics(results).to_dict()["tasks"][0]

    assert entry["generation_counts"]["cap_hit"] == 2
    assert entry["generation_counts"]["generations"] == 4


def test_metrics_json_omits_counts_for_loglikelihood_tasks() -> None:
    responses = [_response((" yes", None), request_type=RequestType.LOGLIKELIHOOD)]
    results = _aggregate(_task_result(responses))

    entry = build_single_model_metrics(results).to_dict()["tasks"][0]

    assert "generation_counts" not in entry


def test_stored_task_row_carries_counts() -> None:
    results = _aggregate(_task_result(_reasoning_batch()))
    results["tasks"][SPEC]["task_hash"] = "deadbeef"

    row = convert_runner_results(results, experiment_id="exp-1").tasks[0]

    assert row.generation_counts is not None
    assert row.generation_counts["cap_hit"] == 2


# ---------------------------------------------------------------------------
# Suite averages
# ---------------------------------------------------------------------------


def _task_data(cap_hit: int, generations: int) -> dict[str, Any]:
    return {
        "metrics": CLEAN_METRICS,
        "primary_metric": "accuracy:exact_match",
        "num_instances": generations,
        "generation_counts": {
            "generations": generations,
            "cap_hit": cap_hit,
            "empty": 0,
            "unclosed_think": 0,
            "finish_reason_unknown": 0,
        },
    }


def test_average_suite_sums_member_counts() -> None:
    task_results = {
        "ifeval_mt_wildchat_unused_withRewrite": _task_data(cap_hit=1, generations=10),
        "ifeval_mt_ood_wildchat_unused_withRewrite": _task_data(cap_hit=2, generations=10),
        "ifeval_ood": _task_data(cap_hit=291, generations=300),
    }

    suite = compute_suite_aggregations(["ifbench"], task_results)["ifbench"]

    assert suite["generation_counts"]["cap_hit"] == 294
    assert suite["generation_counts"]["generations"] == 320


def test_average_of_averages_suite_sums_member_counts() -> None:
    task_results = {
        "aime_2024:pass_at_32": _task_data(cap_hit=5, generations=960),
        "aime_2025:pass_at_32": _task_data(cap_hit=7, generations=960),
    }

    suite = compute_suite_aggregations(["aime_2022_to_2026"], task_results)["aime_2022_to_2026"]

    assert suite["generation_counts"]["cap_hit"] == 12
    assert suite["generation_counts"]["generations"] == 1920


def test_sum_is_none_when_no_member_has_counts() -> None:
    assert sum_generation_counts({"a": {"metrics": CLEAN_METRICS}}, ["a", "missing"]) is None

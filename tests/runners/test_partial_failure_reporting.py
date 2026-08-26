"""Regression tests for partial-failure reporting (issue #310 / U0005).

Seeded from a reproduction of 27B attempt-1 (experiment 01M0VW7G0WRRN1452DXCWMW8W6):
a healthy vLLM server rejected 47 of 63 requests with a per-request 400 (context
length). The harness logged "Processed 63/63", saved 16 predictions, printed a green
Success row and exited 0, because compute_task_metrics built an error summary from the
failed instances, logged it, and then dropped it on the floor.

The tests below pin the three halves of the fix:
  1. the error summary is attached to the TaskResult,
  2. saved-vs-processed accounting reaches the summary table and metrics.json,
  3. a hard-failure rate above the configured threshold fails the task and the run.
"""

from __future__ import annotations

import asyncio
import logging
import queue
from types import SimpleNamespace

import pytest

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.harness.config import HarnessConfig
from olmo_eval.inference.errors import classify_terminal_provider_error
from olmo_eval.runners.asynq.preparation import compute_task_metrics
from olmo_eval.runners.asynq.processing import process_chat_request
from olmo_eval.runners.asynq.results import (
    _report_task_completion,
    aggregate_results,
    check_hard_failure_gate,
)
from olmo_eval.runners.asynq.types import QueueItem, ResultItem
from olmo_eval.runners.common.constants import HardFailureRateExceeded
from olmo_eval.runners.common.types import DEFAULT_MAX_HARD_FAILURE_RATE, TaskResult
from olmo_eval.runners.processing.metrics import build_single_model_metrics, log_summary

SPEC = "deepscholar_bench:probe"
TOTAL_INSTANCES = 63
SAVED_INSTANCES = 16
HARD_FAILURES = TOTAL_INSTANCES - SAVED_INSTANCES  # 47


# ---------------------------------------------------------------------------
# Stubs reproducing the attempt-1 shape
# ---------------------------------------------------------------------------


def _context_length_400() -> Exception:
    """A per-request vLLM 400, as seen in 27B attempt-1.

    Deliberately NOT an EngineDeadError: the provider is healthy, it is these
    particular requests that are rejected.
    """
    error_type = type(
        "VLLMValidationError",
        (Exception,),
        {"__module__": "vllm.entrypoints.openai.protocol"},
    )
    return error_type(
        "This model's maximum context length is 40960 tokens. "
        "However, you requested 52731 tokens. (HTTP 400)"
    )


def _queue_item(instance_idx: int) -> QueueItem:
    return QueueItem(
        model_name="qwen3-8b-27b",
        task_id=SPEC,
        instance_idx=instance_idx,
        instance=Instance(question=f"q{instance_idx}", gold_answer="a"),
        request=LMRequest(request_type=RequestType.CHAT, prompt=f"q{instance_idx}"),
    )


def _stub_harness(
    hard_failing: set[int],
    soft_failing: set[int] | None = None,
) -> SimpleNamespace:
    """A harness whose provider is healthy but 400s a fixed subset of requests.

    Instances in `soft_failing` come back with an error AND a usable output, the
    MaxTurnsExceeded shape that must keep scoring.
    """
    soft = soft_failing or set()

    async def run(request, sampling_params, trace_metadata=None):
        idx = trace_metadata["instance_idx"]
        if idx in hard_failing:
            raise _context_length_400()
        return SimpleNamespace(
            final_output=LMOutput(text=f"answer {idx}", metadata={}),
            trajectory=None,
            error="MaxTurnsExceeded" if idx in soft else None,
            max_turns_reached=idx in soft,
            total_tool_calls=0,
            num_turns=1,
        )

    return SimpleNamespace(
        provider=SimpleNamespace(describe_request=lambda *a, **k: None),
        _apply_config=lambda request: request,
        run=run,
    )


class _StubTask:
    """Minimal task that reports a healthy-looking metric."""

    config = TaskConfig(name="deepscholar_bench", data_source="test/data")

    def compute_metrics(self, responses: list[Response]) -> dict[str, dict[str, float]]:
        return {"organization": {"mean": 0.72}}


def _scored_response(idx: int) -> Response:
    return Response(
        instance=Instance(question=f"q{idx}", gold_answer="a"),
        request=LMRequest(request_type=RequestType.CHAT, prompt=f"q{idx}"),
        outputs=[LMOutput(text=f"answer {idx}", metadata={})],
    )


def _run_inference(
    total: int,
    hard_failing: set[int],
    soft_failing: set[int] | None = None,
) -> list[ResultItem]:
    """Drive the real per-instance chat path and collect what the workers emit."""
    harness = _stub_harness(hard_failing, soft_failing)
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    async def drive() -> None:
        for idx in range(total):
            await process_chat_request(
                _queue_item(idx),
                harness,  # type: ignore[arg-type]
                result_queue,  # type: ignore[arg-type]
            )

    asyncio.run(drive())

    items: list[ResultItem] = []
    while not result_queue.empty():
        items.append(result_queue.get())
    return items


def _task_result_from(items: list[ResultItem], **kwargs) -> TaskResult:
    """Fold worker results into a TaskResult exactly as process_results would.

    process_results routes an error-with-no-outputs to tracker.add_failure() and
    scores everything else, including soft failures that carry an output.
    """
    hard_failures = {item.instance_idx: (item.error or "") for item in items if _is_hard(item)}
    scored = [_scored_response(item.instance_idx) for item in items if not _is_hard(item)]
    return compute_task_metrics(
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        scored_responses=scored,
        failed_instances=hard_failures,
        total_instances=len(items),
        duration_seconds=1234.0,
        **kwargs,
    )


def _is_hard(item: ResultItem) -> bool:
    """A hard failure is an error with no outputs at all."""
    return bool(item.error) and not item.outputs


def _synthetic_result(
    total: int,
    failures: int,
    **kwargs,
) -> TaskResult:
    """A TaskResult for `failures` of `total` instances hard-failing."""
    return compute_task_metrics(
        spec=SPEC,
        task=_StubTask(),  # type: ignore[arg-type]
        scored_responses=[_scored_response(i) for i in range(total - failures)],
        failed_instances={i: "boom" for i in range(total - failures, total)},
        total_instances=total,
        duration_seconds=1.0,
        **kwargs,
    )


def _aggregate(result: TaskResult) -> dict:
    provider_config = SimpleNamespace(
        alias="qwen3-8b-27b",
        model="Qwen/Qwen3-8B",
        kind="vllm_server",
        to_dict=lambda: {},
    )
    return aggregate_results(
        results={SPEC: result},
        expanded_tasks=[SPEC],
        task_specs=[SPEC],
        provider_config=provider_config,
        attention_backend=None,
    )


# ---------------------------------------------------------------------------
# The premise: a per-request 400 is not terminal, so it lands as a hard failure
# ---------------------------------------------------------------------------


def test_per_request_400_is_not_a_terminal_provider_error() -> None:
    assert classify_terminal_provider_error(_context_length_400()) is None


def test_hard_and_soft_failures_stay_distinct() -> None:
    items = _run_inference(
        TOTAL_INSTANCES,
        hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES)),
        soft_failing={0, 1},
    )
    hard = [item for item in items if _is_hard(item)]
    soft = [item for item in items if item.error and item.outputs]

    assert len(items) == TOTAL_INSTANCES
    assert len(hard) == HARD_FAILURES
    assert len(soft) == 2


# ---------------------------------------------------------------------------
# 47/63 hard failures: the attempt-1 case
# ---------------------------------------------------------------------------


def test_partial_hard_failure_marks_task_failed() -> None:
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES))
    )
    result = _task_result_from(items)

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    assert "Hard failure rate" in result.error
    # The summary compute_task_metrics used to drop is now attached.
    assert result.error_summary is not None
    assert "47 instances failed" in result.error_summary
    assert "maximum context length" in result.error_summary

    assert result.num_instances == SAVED_INSTANCES
    assert result.instances_processed == TOTAL_INSTANCES
    assert result.instances_failed == HARD_FAILURES
    assert result.hard_failure_rate == pytest.approx(HARD_FAILURES / TOTAL_INSTANCES)


def test_partial_hard_failure_fails_the_run() -> None:
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES))
    )
    result = _task_result_from(items)

    with pytest.raises(HardFailureRateExceeded) as excinfo:
        check_hard_failure_gate({SPEC: result})
    assert SPEC in str(excinfo.value)


def test_partial_hard_failure_is_recorded_in_the_results_dict() -> None:
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(range(SAVED_INSTANCES, TOTAL_INSTANCES))
    )
    results = _aggregate(_task_result_from(items))
    task_entry = results["tasks"][SPEC]

    assert results["errors"], "a failed task must be recorded at run level"
    assert "error" in task_entry
    assert task_entry["instances_saved"] == SAVED_INSTANCES
    assert task_entry["instances_processed"] == TOTAL_INSTANCES
    assert task_entry["instances_failed"] == HARD_FAILURES
    assert "47 instances failed" in task_entry["error_summary"]


# ---------------------------------------------------------------------------
# 100% hard failures: issue #310's BadRequestError case
# ---------------------------------------------------------------------------


def test_total_failure_marks_task_failed_and_fails_the_run() -> None:
    items = _run_inference(TOTAL_INSTANCES, hard_failing=set(range(TOTAL_INSTANCES)))
    result = _task_result_from(items)

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    assert "Hard failure rate" in result.error
    assert result.error_summary is not None
    assert "63 instances failed" in result.error_summary
    assert result.num_instances == 0
    assert result.instances_processed == TOTAL_INSTANCES
    assert result.instances_failed == TOTAL_INSTANCES
    assert result.hard_failure_rate == pytest.approx(1.0)

    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


# ---------------------------------------------------------------------------
# 0% hard failures: a clean run stays green and exits 0
# ---------------------------------------------------------------------------


def test_clean_run_stays_green() -> None:
    items = _run_inference(TOTAL_INSTANCES, hard_failing=set())
    result = _task_result_from(items)

    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    assert result.error_summary is None
    assert result.num_instances == TOTAL_INSTANCES
    assert result.instances_processed == TOTAL_INSTANCES
    assert result.instances_failed == 0
    assert result.hard_failure_rate == pytest.approx(0.0)

    check_hard_failure_gate({SPEC: result})  # must not raise

    results = _aggregate(result)
    assert results["errors"] == []
    assert results["tasks"][SPEC]["metrics"] == {"organization": {"mean": 0.72}}


def test_soft_failures_alone_keep_the_task_green() -> None:
    """MaxTurnsExceeded with a fallback answer still scores (preserved from PR#300)."""
    items = _run_inference(
        TOTAL_INSTANCES, hard_failing=set(), soft_failing=set(range(TOTAL_INSTANCES))
    )
    result = _task_result_from(items)

    assert result.instances_failed == 0
    assert result.num_instances == TOTAL_INSTANCES
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    check_hard_failure_gate({SPEC: result})


# ---------------------------------------------------------------------------
# Threshold boundary and configurability
# ---------------------------------------------------------------------------


def test_default_threshold_is_five_percent() -> None:
    assert DEFAULT_MAX_HARD_FAILURE_RATE == 0.05


def test_rate_exactly_at_threshold_passes() -> None:
    result = _synthetic_result(total=20, failures=1)  # exactly 0.05

    assert result.hard_failure_rate == pytest.approx(0.05)
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    # The summary is still attached even though the task passes.
    assert result.error_summary == "Instance 19 failed: boom"
    check_hard_failure_gate({SPEC: result})


def test_rate_just_under_threshold_passes() -> None:
    result = _synthetic_result(total=63, failures=3)  # 0.0476

    assert result.hard_failure_rate == pytest.approx(3 / 63)
    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    assert result.error_summary is not None
    check_hard_failure_gate({SPEC: result})


def test_rate_just_over_threshold_fails() -> None:
    result = _synthetic_result(total=63, failures=4)  # 0.0635

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


def test_threshold_can_be_raised_to_tolerate_failures() -> None:
    result = _synthetic_result(total=63, failures=47, max_hard_failure_rate=0.9)

    assert result.hard_failure_rate_exceeded is False
    assert result.error is None
    # Tolerated, but never silent: the summary and the accounting survive.
    assert result.error_summary is not None
    assert result.instances_failed == 47
    assert result.instances_processed == 63
    check_hard_failure_gate({SPEC: result})


def test_threshold_can_be_lowered_to_zero_tolerance() -> None:
    result = _synthetic_result(total=63, failures=1, max_hard_failure_rate=0.0)

    assert result.hard_failure_rate_exceeded is True
    assert result.error is not None
    with pytest.raises(HardFailureRateExceeded):
        check_hard_failure_gate({SPEC: result})


def test_harness_config_round_trips_the_threshold() -> None:
    """The CLI sets this via `-o max_hard_failure_rate=...` after --harness."""
    config = HarnessConfig(name="test", max_hard_failure_rate=0.25)
    assert config.to_dict()["max_hard_failure_rate"] == 0.25
    assert HarnessConfig.from_dict(config.to_dict()).max_hard_failure_rate == 0.25

    default_config = HarnessConfig(name="test")
    assert default_config.max_hard_failure_rate is None
    assert "max_hard_failure_rate" not in default_config.to_dict()


# ---------------------------------------------------------------------------
# What the operator sees
# ---------------------------------------------------------------------------


def test_task_completion_line_reports_saved_vs_processed(caplog) -> None:
    caplog.set_level(logging.INFO)
    result = _synthetic_result(total=63, failures=47)

    _report_task_completion("qwen3-8b-27b", result)

    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "16/63 instances saved" in logged
    assert "47 hard-failed" in logged


def test_summary_table_reports_saved_vs_processed(capsys) -> None:
    results = _aggregate(_synthetic_result(total=63, failures=47))

    log_summary(results)

    out = capsys.readouterr().out
    assert "16/63" in out
    assert "Failed" in out


def test_metrics_json_carries_the_accounting() -> None:
    results = _aggregate(_synthetic_result(total=63, failures=47))

    metrics_output = build_single_model_metrics(results).to_dict()
    entry = metrics_output["tasks"][0]

    assert entry["instances_saved"] == 16
    assert entry["instances_processed"] == 63
    assert entry["instances_failed"] == 47
    assert "47 instances failed" in entry["error_summary"]
    assert metrics_output["errors"], "the failed task must appear in metrics.json errors"


def test_metrics_json_accounting_on_a_clean_run() -> None:
    results = _aggregate(_synthetic_result(total=63, failures=0))

    entry = build_single_model_metrics(results).to_dict()["tasks"][0]

    assert entry["instances_saved"] == 63
    assert entry["instances_processed"] == 63
    assert entry["instances_failed"] == 0
    assert "error_summary" not in entry

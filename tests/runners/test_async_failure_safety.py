"""Regression tests for fatal async inference failures."""

from __future__ import annotations

import asyncio
import logging
import queue

import pytest

from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.inference.errors import TerminalProviderError
from olmo_eval.runners.asynq.batching import BatchConfig, StreamingStrategy
from olmo_eval.runners.asynq.monitoring import check_workers_alive
from olmo_eval.runners.asynq.processing import process_batch, process_chat_request
from olmo_eval.runners.asynq.results import (
    _build_pending_instance_keys,
    _consume_pending_instance,
    process_results,
)
from olmo_eval.runners.asynq.types import WORKER_FATAL, QueueItem, ResultItem, TaskTracker


def _engine_dead_error() -> Exception:
    error_type = type(
        "EngineDeadError",
        (Exception,),
        {"__module__": "vllm.v1.engine.exceptions"},
    )
    return error_type("EngineCore encountered an issue")


def _queue_item(instance_idx: int) -> QueueItem:
    return QueueItem(
        model_name="model",
        task_id="task",
        instance_idx=instance_idx,
        instance=Instance(question=f"question {instance_idx}", gold_answer="answer"),
        request=LMRequest(
            request_type=RequestType.COMPLETION,
            prompt=f"question {instance_idx}",
        ),
    )


class _Provider:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def describe_request(self, _request: LMRequest, _params: object) -> None:
        return None

    async def agenerate(self, _requests: list[LMRequest], _params: object) -> None:
        raise self.error


class _Harness:
    def __init__(self, error: Exception) -> None:
        self.provider = _Provider(error)

    def _apply_config(self, request: LMRequest) -> LMRequest:
        return request

    def flush_metrics(self, _batch_hash: str) -> None:
        raise AssertionError("metrics should not flush after a failed provider call")


class _ChatHarness(_Harness):
    async def run(self, *_args: object, **_kwargs: object) -> None:
        raise self.provider.error


def test_terminal_batch_error_propagates_without_instance_failures() -> None:
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    with pytest.raises(TerminalProviderError, match="EngineDeadError"):
        asyncio.run(
            process_batch(
                [_queue_item(0), _queue_item(1)],
                _Harness(_engine_dead_error()),  # type: ignore[arg-type]
                result_queue,  # type: ignore[arg-type]
            )
        )

    assert result_queue.empty()


def test_terminal_chat_error_propagates_without_instance_failure() -> None:
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    with pytest.raises(TerminalProviderError, match="EngineDeadError"):
        asyncio.run(
            process_chat_request(
                _queue_item(0),
                _ChatHarness(_engine_dead_error()),  # type: ignore[arg-type]
                result_queue,  # type: ignore[arg-type]
            )
        )

    assert result_queue.empty()


def test_streaming_strategy_propagates_fast_failure_and_cancels_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Reporter:
        def progress_callback(self, _label: str):
            return lambda *_args, **_kwargs: None

    async def run() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def fail_or_wait(items: list[QueueItem], *_args: object, **_kwargs: object) -> None:
            if items[0].instance_idx == 0:
                await sibling_started.wait()
                raise TerminalProviderError("vLLM", "EngineDeadError", "engine died")

            sibling_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        monkeypatch.setattr("olmo_eval.runners.asynq.processing.process_items", fail_or_wait)
        item_queue: queue.Queue[QueueItem | None] = queue.Queue()
        result_queue: queue.Queue[ResultItem] = queue.Queue()
        item_queue.put(_queue_item(0))
        item_queue.put(_queue_item(1))

        with pytest.raises(TerminalProviderError, match="engine died"):
            await StreamingStrategy(BatchConfig.streaming()).run(
                item_queue=item_queue,  # type: ignore[arg-type]
                harness=object(),  # type: ignore[arg-type]
                result_queue=result_queue,  # type: ignore[arg-type]
                max_concurrency=2,
                worker_logger=logging.getLogger(__name__),
                total_instances=2,
            )

        assert sibling_cancelled.is_set()
        assert result_queue.empty()

    monkeypatch.setattr("olmo_eval.common.beaker_status.BeakerStatusReporter", _Reporter)
    asyncio.run(run())


def test_recoverable_batch_error_still_reports_each_instance() -> None:
    result_queue: queue.Queue[ResultItem] = queue.Queue()

    asyncio.run(
        process_batch(
            [_queue_item(0), _queue_item(1)],
            _Harness(RuntimeError("request failed")),  # type: ignore[arg-type]
            result_queue,  # type: ignore[arg-type]
        )
    )

    results = [result_queue.get_nowait(), result_queue.get_nowait()]
    assert [result.instance_idx for result in results] == [0, 1]
    assert all(result.error and "request failed" in result.error for result in results)


def test_pending_instance_tracking_rejects_duplicates() -> None:
    trackers = {
        "task": TaskTracker(
            model_name="model",
            spec="task",
            task=None,
            total_instances=2,
        )
    }
    pending = _build_pending_instance_keys(trackers)
    result = ResultItem(
        model_name="model",
        task_id="task",
        instance_idx=0,
        instance=None,
        request=None,
        outputs=[],
    )

    _consume_pending_instance(pending, result)

    assert pending == {("model", "task", 1)}
    with pytest.raises(RuntimeError, match="duplicate or unexpected"):
        _consume_pending_instance(pending, result)


def test_worker_fatal_reports_exact_pending_instance_identities() -> None:
    result_queue: queue.Queue[ResultItem] = queue.Queue()
    result_queue.put(
        ResultItem(
            model_name="model",
            task_id=WORKER_FATAL,
            instance_idx=-1,
            instance=None,
            request=None,
            outputs=[],
            error="terminal provider failure",
        )
    )
    trackers = {
        "task": TaskTracker(
            model_name="model",
            spec="task",
            task=None,
            total_instances=2,
        )
    }

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(
            process_results(
                trackers=trackers,
                result_queue=result_queue,  # type: ignore[arg-type]
                workers=[],
                scoring_context=None,  # type: ignore[arg-type]
                scoring_concurrency=1,
                total_tasks=1,
                total_instances=2,
                model_name="model",
                save_predictions=False,
                write_predictions_fn=lambda *_args: None,
                save_requests=False,
                write_requests_fn=lambda *_args: None,
            )
        )

    message = str(exc_info.value)
    assert "2 inference result(s) pending" in message
    assert "model:task[0]" in message
    assert "model:task[1]" in message


class _Process:
    def __init__(self, *, alive: bool, exitcode: int | None) -> None:
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


def test_single_failed_worker_is_fatal_while_other_workers_are_alive() -> None:
    workers = [_Process(alive=False, exitcode=1), _Process(alive=True, exitcode=None)]

    with pytest.raises(RuntimeError, match="worker 0 exited with code 1"):
        check_workers_alive(
            workers,  # type: ignore[arg-type]
            queue.Queue(),  # type: ignore[arg-type]
        )

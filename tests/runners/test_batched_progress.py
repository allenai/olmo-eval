"""Tests for BatchedStrategy progress reporting to Beaker."""

from __future__ import annotations

import asyncio
import logging
import queue
from typing import Any

import pytest

from olmo_eval.common.types import Instance, LMRequest, RequestType
from olmo_eval.runners.asynq.batching import BatchConfig, BatchedStrategy
from olmo_eval.runners.asynq.types import QueueItem, ResultItem


def _queue_item(instance_idx: int) -> QueueItem:
    return QueueItem(
        model_name="model",
        task_id="task",
        instance_idx=instance_idx,
        instance=Instance(question=f"question {instance_idx}", gold_answer="answer"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=f"question {instance_idx}"),
    )


class _Reporter:
    labels: list[str]
    calls: list[tuple[int, int, bool]]

    def progress_callback(self, label: str):
        _Reporter.labels.append(label)

        def _cb(count: int, total: int, *, force: bool = False) -> None:
            _Reporter.calls.append((count, total, force))

        return _cb


@pytest.fixture
def reporter(monkeypatch: pytest.MonkeyPatch) -> type[_Reporter]:
    _Reporter.labels = []
    _Reporter.calls = []
    monkeypatch.setattr("olmo_eval.common.beaker_status.BeakerStatusReporter", _Reporter)
    return _Reporter


def _run(
    monkeypatch: pytest.MonkeyPatch,
    items: list[QueueItem | None],
    *,
    chunk_size: int,
    total_instances: int,
    num_workers: int = 1,
    process_items: Any = None,
) -> list[int]:
    batch_sizes: list[int] = []

    async def record(batch: list[QueueItem], *_args: object, **_kwargs: object) -> None:
        batch_sizes.append(len(batch))

    monkeypatch.setattr("olmo_eval.runners.asynq.processing.process_items", process_items or record)
    item_queue: queue.Queue[QueueItem | None] = queue.Queue()
    for item in items:
        item_queue.put(item)

    config = BatchConfig(chunk_size=chunk_size, chunk_timeout=0.05)
    asyncio.run(
        BatchedStrategy(config).run(
            item_queue=item_queue,  # type: ignore[arg-type]
            harness=object(),  # type: ignore[arg-type]
            result_queue=queue.Queue[ResultItem](),  # type: ignore[arg-type]
            max_concurrency=1,
            worker_logger=logging.getLogger(__name__),
            total_instances=total_instances,
            num_workers=num_workers,
        )
    )
    return batch_sizes


def test_reports_cumulative_progress_across_batches(
    monkeypatch: pytest.MonkeyPatch, reporter: type[_Reporter]
) -> None:
    items: list[QueueItem | None] = [_queue_item(i) for i in range(5)] + [None]

    batch_sizes = _run(monkeypatch, items, chunk_size=2, total_instances=5)

    assert batch_sizes == [2, 2, 1]
    assert reporter.labels == ["Processed"]
    assert reporter.calls == [(2, 5, False), (4, 5, False), (5, 5, False), (5, 5, True)]


def test_forces_final_report_when_shutdown_arrives_alone(
    monkeypatch: pytest.MonkeyPatch, reporter: type[_Reporter]
) -> None:
    items: list[QueueItem | None] = [_queue_item(i) for i in range(4)] + [None]

    batch_sizes = _run(monkeypatch, items, chunk_size=2, total_instances=4)

    assert batch_sizes == [2, 2]
    assert reporter.calls == [(2, 4, False), (4, 4, False), (4, 4, True)]


def test_total_is_this_workers_share(
    monkeypatch: pytest.MonkeyPatch, reporter: type[_Reporter]
) -> None:
    items: list[QueueItem | None] = [_queue_item(i) for i in range(3)] + [None]

    _run(monkeypatch, items, chunk_size=64, total_instances=6, num_workers=2)

    assert reporter.calls == [(3, 3, False), (3, 3, True)]


def test_forces_final_report_of_completed_batches_on_failure(
    monkeypatch: pytest.MonkeyPatch, reporter: type[_Reporter]
) -> None:
    items: list[QueueItem | None] = [_queue_item(i) for i in range(4)] + [None]
    calls = 0

    async def fail_second(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("provider died")

    with pytest.raises(RuntimeError, match="provider died"):
        _run(monkeypatch, items, chunk_size=2, total_instances=4, process_items=fail_second)

    assert reporter.calls == [(2, 4, False), (2, 4, True)]

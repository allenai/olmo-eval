"""Tests for async inference worker coordination and batch logging."""

from __future__ import annotations

from inspect import signature
from typing import Any

from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.inference.provider_manager import ProviderManager


class _FakeProcess:
    def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
        self.target = target
        self.args = args
        self.started = False

    def start(self) -> None:
        self.started = True


class _FakeContext:
    def __init__(self) -> None:
        self.processes: list[_FakeProcess] = []

    def Process(self, target: Any, args: tuple[Any, ...]) -> _FakeProcess:
        process = _FakeProcess(target, args)
        self.processes.append(process)
        return process


def test_provider_manager_passes_start_event_to_each_worker() -> None:
    ctx = _FakeContext()
    init_queue = object()
    start_event = object()
    manager = ProviderManager(
        harness_config=HarnessConfig(
            name="test",
            provider=ProviderConfig(model="test-model"),
        ),
        num_inference_workers=2,
        gpu_ids=[0, 1],
        item_queue=object(),
        result_queue=object(),
        output_dir="/tmp/results",
    )

    processes = manager.start(
        ctx,
        total_instances=4,
        init_queue=init_queue,
        start_event=start_event,
    )

    assert processes == ctx.processes
    assert all(process.started for process in processes)
    for process in processes:
        worker_args = dict(zip(signature(process.target).parameters, process.args, strict=False))
        assert worker_args["init_queue"] is init_queue
        assert worker_args["num_workers"] == 2
        assert worker_args["start_event"] is start_event

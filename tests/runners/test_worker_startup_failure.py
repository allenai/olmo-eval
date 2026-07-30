"""A provider that cannot start must fail the run, not produce an empty one.

When a managed vLLM server misses its startup budget, ``VLLMServerProcess.start``
raises inside the inference worker, i.e. inside a subprocess whose exception the
runner cannot see directly. The failure has to travel: worker exception ->
``WORKER_FATAL`` marker on the result queue -> runner raises -> CLI exits
non-zero. Every hop is pinned here, because if any of them stops working the run
ends up reporting zero instances and zero failures, which reads exactly like a
successful run that produced no data.
"""

from __future__ import annotations

import queue

import pytest

from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq.monitoring import check_workers_alive, wait_for_init_times
from olmo_eval.runners.asynq.types import WORKER_FATAL
from olmo_eval.runners.asynq.workers import inference_worker

# Verbatim shape of the error VLLMServerProcess.start raises on timeout, so the
# assertions below track the real message rather than a paraphrase.
STARTUP_TIMEOUT_ERROR = (
    "vLLM server failed to start for model Qwen/Qwen3.5-35B-A3B within 300.0s. "
    "Error: <urlopen error [Errno 111] Connection refused>"
)


class CannotStartProvider:
    """Stands in for a provider whose managed server never becomes healthy.

    Resolved by dotted path through the ``python`` provider kind, which is how a
    launch points at a provider class; the constructor is where
    ``VLLMServerProcess.start`` is called and therefore where a startup timeout
    surfaces.
    """

    def __init__(self, model_name: str, **kwargs: object) -> None:
        raise RuntimeError(STARTUP_TIMEOUT_ERROR)


class RecordingQueue(queue.Queue):
    """Synchronous stand-in for ``mp.Queue``.

    Only the transport is faked. Using a real ``mp.Queue`` here would make the
    test racy: a put is flushed by a background feeder thread, so an immediately
    following ``get_nowait`` can still raise ``Empty``.
    """

    def cancel_join_thread(self) -> None:
        return None


class ExitedWorker:
    """A worker process that has already died with a non-zero status."""

    exitcode = 1

    def is_alive(self) -> bool:
        return False

    def terminate(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


def _harness_config() -> HarnessConfig:
    return HarnessConfig(
        name="startup-failure",
        provider=ProviderConfig(
            kind="python",
            model="Qwen/Qwen3.5-35B-A3B",
            kwargs={"class": f"{__name__}.{CannotStartProvider.__name__}"},
        ),
    )


def test_worker_reports_a_provider_that_cannot_start_as_fatal() -> None:
    """The worker must exit non-zero and leave the cause on the result queue."""
    result_queue = RecordingQueue()

    with pytest.raises(SystemExit) as exit_info:
        inference_worker(
            worker_id="startup-failure-0",
            gpu_ids=[],
            item_queue=RecordingQueue(),
            result_queue=result_queue,
            harness_config_dict=_harness_config().to_dict(),
            total_instances=1,
            init_queue=RecordingQueue(),
        )

    assert exit_info.value.code == 1

    item = result_queue.get_nowait()
    assert item.task_id == WORKER_FATAL
    assert STARTUP_TIMEOUT_ERROR in item.error
    # No output was produced, so nothing can be mistaken for a scored instance.
    assert item.outputs == []


def test_runner_turns_that_fatal_marker_into_an_exception() -> None:
    """The runner's init wait must raise, and must carry the cause with it."""
    result_queue = RecordingQueue()
    result_queue.put(
        _fatal_item(f"Worker process crashed: {STARTUP_TIMEOUT_ERROR}"),
    )

    with pytest.raises(RuntimeError) as exc_info:
        wait_for_init_times(
            RecordingQueue(),
            num_workers=1,
            workers=[ExitedWorker()],
            result_queue=result_queue,
            timeout=5,
        )

    # The reason the run died has to reach the top-level message; "a worker
    # died" on its own is what sends people looking in the wrong place.
    assert STARTUP_TIMEOUT_ERROR in str(exc_info.value)


def test_a_dead_worker_still_fails_the_run_without_a_fatal_marker() -> None:
    """A worker that dies without reporting must not pass for a healthy one.

    ``mp.Queue`` puts are flushed asynchronously, so the marker can still be in
    flight when the runner polls. The fallback message names no cause -- that is
    a known weakness of this path, pinned here so it stays visible -- but it must
    still raise rather than let the run continue.
    """
    with pytest.raises(RuntimeError, match="All workers died unexpectedly"):
        check_workers_alive([ExitedWorker()], RecordingQueue())


def _fatal_item(error: str):
    from olmo_eval.runners.asynq.types import ResultItem

    return ResultItem(
        model_name="Qwen/Qwen3.5-35B-A3B",
        task_id=WORKER_FATAL,
        instance_idx=-1,
        instance=None,
        request=None,
        outputs=[],
        error=error,
    )

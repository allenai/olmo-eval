"""Per-task scoring times: judge and sandbox calls, and the loop's own scoring work."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from olmo_eval.common.execution import ProcessPoolManager, ProcessScoringPoolConfig
from olmo_eval.common.execution.environment import ScoringContext
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers import ProcessScorer
from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.scoring_time import ScoringTimes, time_child_tasks, timed, timed_scoring
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig


class _FakeTime:
    """A settable stand-in for time.perf_counter."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def now(monkeypatch: pytest.MonkeyPatch) -> _FakeTime:
    fake = _FakeTime()
    monkeypatch.setattr("olmo_eval.common.scoring_time._now", fake)
    return fake


async def _elsewhere(now: _FakeTime, seconds: float) -> None:
    """Wait while another task advances the clock, as a remote call would."""
    done = asyncio.get_running_loop().create_future()

    async def finish() -> None:
        now.t += seconds
        done.set_result(None)

    pending = asyncio.ensure_future(finish())
    await done
    await pending


def test_steps_count_as_compute_and_awaits_do_not(now: _FakeTime) -> None:
    async def score() -> str:
        now.t += 0.5  # extracting the answer
        await _elsewhere(now, 10.0)
        now.t += 0.25  # applying a rule-based scorer
        return "scored"

    times = ScoringTimes()
    assert asyncio.run(timed_scoring(score(), times)) == "scored"
    assert times == ScoringTimes(compute_seconds=pytest.approx(0.75))


def test_calls_add_their_full_durations_even_when_concurrent(now: _FakeTime) -> None:
    async def call(component: str, seconds: float) -> None:
        with timed(component):  # type: ignore[arg-type]
            await _elsewhere(now, seconds)

    async def score() -> None:
        # Judge calls run as their own tasks, as asyncio.gather schedules them.
        await asyncio.gather(call("judge", 2.0), call("judge", 3.0), call("sandbox", 1.0))

    times = ScoringTimes()
    asyncio.run(timed_scoring(score(), times))
    # Each call is timed from its start, while the others' time also passes.
    assert times.judge_seconds >= 2.0 + 3.0
    assert times.sandbox_seconds >= 1.0
    assert times.compute_seconds == 0.0


def test_calls_outside_scoring_are_not_recorded(now: _FakeTime) -> None:
    async def outside() -> None:
        with timed("judge"):
            await _elsewhere(now, 4.0)

    times = ScoringTimes()

    async def run() -> None:
        await timed_scoring(asyncio.sleep(0), times)
        await outside()

    asyncio.run(run())
    assert times == ScoringTimes()


def test_concurrent_instances_record_to_their_own_task(now: _FakeTime) -> None:
    async def judged() -> None:
        with timed("judge"):
            await _elsewhere(now, 5.0)

    async def computed() -> None:
        now.t += 2.0
        await asyncio.sleep(0)

    judge_task, cpu_task = ScoringTimes(), ScoringTimes()

    async def run() -> None:
        await asyncio.gather(
            timed_scoring(judged(), judge_task), timed_scoring(computed(), cpu_task)
        )

    asyncio.run(run())
    assert cpu_task == ScoringTimes(compute_seconds=pytest.approx(2.0))
    assert judge_task.compute_seconds == 0.0
    assert judge_task.judge_seconds >= 5.0


def test_a_failing_scorer_keeps_its_time_and_raises(now: _FakeTime) -> None:
    async def score() -> None:
        now.t += 1.5
        await asyncio.sleep(0)
        raise ValueError("unparseable answer")

    times = ScoringTimes()
    with pytest.raises(ValueError, match="unparseable"):
        asyncio.run(timed_scoring(score(), times))
    assert times.compute_seconds == pytest.approx(1.5)


def test_cancellation_reaches_the_scoring_coroutine(now: _FakeTime) -> None:
    cleaned_up = []

    async def score() -> None:
        try:
            await asyncio.sleep(3600)
        finally:
            cleaned_up.append(True)

    async def run() -> None:
        task = asyncio.create_task(timed_scoring(score(), ScoringTimes()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert cleaned_up == [True]


_CLOCK = _FakeTime()


@dataclass(frozen=True, slots=True)
class _SlowContextScorer(ContextScorer):
    """A context scorer that spends 0.2 s of CPU per output on the fake clock."""

    name: str = "slow_context"

    async def ascore_with_context(self, instance, output, context) -> float:
        _CLOCK.t += 0.2
        await asyncio.sleep(0)
        return 1.0


@dataclass(frozen=True, slots=True)
class _SleepingProcessScorer(ProcessScorer):
    """A process scorer that takes 0.05 s of real time per output."""

    name: str = "sleeping_process"

    def process_score(self, instance: Instance, output: LMOutput) -> float:
        time.sleep(0.05)
        return 1.0


class _Task(Task):
    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="Q", gold_answer="A")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)


def _response(outputs: int) -> Response:
    return Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[LMOutput(text="A") for _ in range(outputs)],
    )


def test_scorers_run_as_child_tasks_count_as_compute(monkeypatch: pytest.MonkeyPatch) -> None:
    _CLOCK.t = 0.0
    monkeypatch.setattr("olmo_eval.common.scoring_time._now", _CLOCK)
    task = _Task(
        TaskConfig(
            name="t",
            data_source="test/dataset",
            metrics=(AccuracyMetric(scorer=_SlowContextScorer),),
        )
    )
    times = ScoringTimes()

    async def run() -> None:
        restore = time_child_tasks()
        try:
            # Task.score_responses gathers context scorers, one child task per output.
            await timed_scoring(task.score_responses([_response(3)], ScoringContext()), times)
        finally:
            restore()
        assert asyncio.get_running_loop().get_task_factory() is None

    asyncio.run(run())
    assert times.compute_seconds == pytest.approx(3 * 0.2)


def test_process_pool_scoring_counts_as_compute() -> None:
    metric = AccuracyMetric(scorer=_SleepingProcessScorer)
    task = _Task(TaskConfig(name="t", data_source="test/dataset", metrics=(metric,)))
    manager = ProcessPoolManager({"cpu": ProcessScoringPoolConfig(workers=2)})
    context = ScoringContext(process_pool_manager=manager)
    times = ScoringTimes()
    try:
        asyncio.run(timed_scoring(task.score_responses([_response(2)], context), times))
    finally:
        manager.shutdown()
    # Time in the worker counts; the scoring loop's own steps add a little more.
    assert times.compute_seconds >= 2 * 0.05

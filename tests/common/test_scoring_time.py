"""Per-task scoring times: judge and sandbox calls, and the loop's own scoring work."""

from __future__ import annotations

import asyncio

import pytest

from olmo_eval.common.scoring_time import ScoringTimes, timed, timed_scoring


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


def test_compute_time_can_be_added_outside_an_instance(now: _FakeTime) -> None:
    times = ScoringTimes()
    with timed("compute", times):
        now.t += 0.5  # computing the task's metrics
    assert times.compute_seconds == pytest.approx(0.5)

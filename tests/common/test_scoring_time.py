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
from olmo_eval.common.scoring_time import (
    ScoringKind,
    ScoringTimes,
    install_child_task_timing,
    measure_instance_scoring,
    measure_scoring,
)
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task, TaskConfig
from tests.conftest import ScoringClock


def test_steps_count_as_compute_and_awaits_do_not(scoring_clock: ScoringClock) -> None:
    async def score() -> str:
        scoring_clock.t += 0.5  # extracting the answer
        await scoring_clock.wait_elsewhere(10.0)
        scoring_clock.t += 0.25  # applying a rule-based scorer
        return "scored"

    totals = ScoringTimes()
    assert asyncio.run(measure_instance_scoring(score(), totals)) == "scored"
    assert totals == ScoringTimes(compute=pytest.approx(0.75))


def test_calls_add_their_full_durations_even_when_concurrent(scoring_clock: ScoringClock) -> None:
    async def call(kind: ScoringKind, seconds: float) -> None:
        with measure_scoring(kind):
            await scoring_clock.wait_elsewhere(seconds)

    async def score() -> None:
        # Judge calls run as their own tasks, as asyncio.gather schedules them.
        await asyncio.gather(call("judge", 2.0), call("judge", 3.0), call("sandbox", 1.0))

    totals = ScoringTimes()
    asyncio.run(measure_instance_scoring(score(), totals))
    # Each call is timed from its start, while the others' time also passes.
    assert totals.judge >= 2.0 + 3.0
    assert totals.sandbox >= 1.0
    assert totals.compute == 0.0


def test_calls_outside_scoring_are_not_recorded(scoring_clock: ScoringClock) -> None:
    async def outside() -> None:
        with measure_scoring("judge"):
            await scoring_clock.wait_elsewhere(4.0)

    totals = ScoringTimes()

    async def run() -> None:
        await measure_instance_scoring(asyncio.sleep(0), totals)
        await outside()

    asyncio.run(run())
    assert totals == ScoringTimes()


def test_concurrent_instances_record_to_their_own_task(scoring_clock: ScoringClock) -> None:
    async def judged() -> None:
        with measure_scoring("judge"):
            await scoring_clock.wait_elsewhere(5.0)

    async def computed() -> None:
        scoring_clock.t += 2.0
        await asyncio.sleep(0)

    judged_totals, computed_totals = ScoringTimes(), ScoringTimes()

    async def run() -> None:
        await asyncio.gather(
            measure_instance_scoring(judged(), judged_totals),
            measure_instance_scoring(computed(), computed_totals),
        )

    asyncio.run(run())
    assert computed_totals == ScoringTimes(compute=pytest.approx(2.0))
    assert judged_totals.compute == 0.0
    assert judged_totals.judge >= 5.0


def test_a_failing_scorer_keeps_its_time_and_raises(scoring_clock: ScoringClock) -> None:
    async def score() -> None:
        scoring_clock.t += 1.5
        await asyncio.sleep(0)
        raise ValueError("unparseable answer")

    totals = ScoringTimes()
    with pytest.raises(ValueError, match="unparseable"):
        asyncio.run(measure_instance_scoring(score(), totals))
    assert totals.compute == pytest.approx(1.5)


def test_cancellation_reaches_the_scoring_coroutine(scoring_clock: ScoringClock) -> None:
    cleaned_up = []

    async def score() -> None:
        try:
            await asyncio.sleep(3600)
        finally:
            cleaned_up.append(True)

    async def run() -> None:
        task = asyncio.create_task(measure_instance_scoring(score(), ScoringTimes()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert cleaned_up == [True]


_scorer_clock = ScoringClock()  # the scoring_clock fixture replaces it for each test


@dataclass(frozen=True, slots=True)
class _SlowContextScorer(ContextScorer):
    """A context scorer that spends 0.2 s of CPU per output on the fake clock."""

    name: str = "slow_context"

    async def ascore_with_context(self, instance, output, context) -> float:
        _scorer_clock.t += 0.2
        await asyncio.sleep(0)
        return 1.0


@dataclass(frozen=True, slots=True)
class _SleepingProcessScorer(ProcessScorer):
    """A process scorer that takes 0.05 s of real time per output, after a 0.5 s warmup."""

    name: str = "sleeping_process"

    def warmup(self) -> None:
        time.sleep(0.5)

    def process_score(self, instance: Instance, output: LMOutput) -> float:
        time.sleep(0.05)
        return 1.0


class _Task(Task):
    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="Q", gold_answer="A")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)


def _response(num_outputs: int) -> Response:
    return Response(
        instance=Instance(question="Q", gold_answer="A"),
        request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
        outputs=[LMOutput(text="A") for _ in range(num_outputs)],
    )


def test_scorers_run_as_child_tasks_count_as_compute(
    scoring_clock: ScoringClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(f"{__name__}._scorer_clock", scoring_clock)
    task = _Task(
        TaskConfig(
            name="t",
            data_source="test/dataset",
            metrics=(AccuracyMetric(scorer=_SlowContextScorer),),
        )
    )
    totals = ScoringTimes()

    async def run() -> None:
        uninstall = install_child_task_timing()
        try:
            # Task.score_responses gathers context scorers, one child task per output.
            await measure_instance_scoring(
                task.score_responses([_response(3)], ScoringContext()), totals
            )
        finally:
            uninstall()
        assert asyncio.get_running_loop().get_task_factory() is None

    asyncio.run(run())
    assert totals.compute == pytest.approx(3 * 0.2)


def test_process_pool_scoring_counts_as_compute() -> None:
    metric = AccuracyMetric(scorer=_SleepingProcessScorer)
    task = _Task(TaskConfig(name="t", data_source="test/dataset", metrics=(metric,)))
    manager = ProcessPoolManager({"cpu": ProcessScoringPoolConfig(workers=2)})
    context = ScoringContext(process_pool_manager=manager)
    totals = ScoringTimes()
    try:
        asyncio.run(measure_instance_scoring(task.score_responses([_response(2)], context), totals))
    finally:
        manager.shutdown()
    # Time in the worker counts, but not its warmup; the loop's own steps add a little.
    assert 2 * 0.05 <= totals.compute < 0.4


def test_child_task_timing_leaves_an_existing_task_factory_alone() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        loop.set_task_factory(asyncio.eager_task_factory)
        uninstall = install_child_task_timing()
        uninstall()
        assert loop.get_task_factory() is asyncio.eager_task_factory
        loop.set_task_factory(None)

    asyncio.run(run())

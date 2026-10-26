"""Time spent in each part of scoring, summed per task.

The async runner scores each instance through ``timed_scoring``, which makes the task's
``ScoringTimes`` current for that instance. Judge and sandbox call sites wrap each call
in ``timed("judge")`` or ``timed("sandbox")``, which adds the call's duration to the
current task. Calls made outside scoring, such as an agent rollout's tool calls in an
inference worker, are not recorded.

Each piece is timed on its own and the durations are summed, so concurrent calls each
count in full: a task's judge time can exceed the wall time its scoring took. Compute
time is the event loop's time running the instance's scoring code between awaits, so
it never overlaps across instances.
"""

from __future__ import annotations

import time
from collections.abc import Coroutine, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

_now = time.perf_counter


@dataclass
class ScoringTimes:
    """Summed seconds of one task's scoring, by component."""

    judge_seconds: float = 0.0  # judge model calls, each from request to reply
    sandbox_seconds: float = 0.0  # sandbox operations, each from lease to result
    compute_seconds: float = 0.0  # scoring code run by the event loop, and metrics


_current: ContextVar[ScoringTimes | None] = ContextVar("scoring_times", default=None)


@contextmanager
def timed(
    component: Literal["judge", "sandbox", "compute"], times: ScoringTimes | None = None
) -> Iterator[None]:
    """Add the block's duration to ``times``, or the current task's, if any."""
    times = times or _current.get()
    if times is None:
        yield
        return
    attr = f"{component}_seconds"
    start = _now()
    try:
        yield
    finally:
        setattr(times, attr, getattr(times, attr) + _now() - start)


async def timed_scoring[T](coro: Coroutine[Any, Any, T], times: ScoringTimes) -> T:
    """Await ``coro`` with ``times`` current, adding its synchronous steps to compute time.

    Tasks that ``coro`` starts (``asyncio.gather``, ``create_task``) inherit ``times``,
    so their judge and sandbox calls are recorded, but their own steps are not timed.
    """
    token = _current.set(times)
    try:
        return await _StepTimed(coro, times)
    finally:
        _current.reset(token)


class _StepTimed[T]:
    """Drives a coroutine step by step, timing each step it runs."""

    def __init__(self, coro: Coroutine[Any, Any, T], times: ScoringTimes) -> None:
        self._coro = coro
        self._times = times

    def __await__(self) -> Generator[Any, Any, T]:
        value: Any = None
        error: BaseException | None = None
        while True:
            start = _now()
            try:
                yielded = self._coro.send(value) if error is None else self._coro.throw(error)
            except StopIteration as stop:
                return stop.value
            finally:
                self._times.compute_seconds += _now() - start
            try:
                value, error = (yield yielded), None
            except BaseException as exc:  # forwarded into the coroutine, e.g. cancellation
                value, error = None, exc


__all__ = ["ScoringTimes", "timed", "timed_scoring"]

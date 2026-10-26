"""Time spent in each part of scoring, summed per task.

The async runner scores each instance through ``measure_instance_scoring``, and calls
``install_child_task_timing`` first so that tasks the scorers start are timed too.

- Judge and sandbox time runs from when a call is made until it returns. It includes the
  client's own retries and excludes waits for a scorer-level judge or sandbox slot.
- Compute time is the scoring code running on the event loop, in tasks it starts, in a
  thread, or in the process pool. Pool worker warmup is excluded.

Each value is a sum of separately timed pieces, so it can exceed the wall time of scoring.
Calls outside scoring are not recorded, such as agent rollouts in inference workers and
the judges of external benchmarks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

ScoringKind = Literal["judge", "sandbox", "compute"]

_now = time.perf_counter


@dataclass
class ScoringTimes:
    """Summed seconds of one task's scoring, by kind."""

    judge: float = 0.0
    sandbox: float = 0.0
    compute: float = 0.0

    def add(self, kind: ScoringKind, seconds: float) -> None:
        setattr(self, kind, getattr(self, kind) + seconds)

    @contextmanager
    def measure(self, kind: ScoringKind) -> Iterator[None]:
        """Add the block's duration to ``kind``."""
        start = _now()
        try:
            yield
        finally:
            self.add(kind, _now() - start)


_current_totals: ContextVar[ScoringTimes | None] = ContextVar("scoring_times", default=None)


def add_scoring_seconds(kind: ScoringKind, seconds: float) -> None:
    """Add ``seconds`` to the current task's ``kind`` time, if any."""
    totals = _current_totals.get()
    if totals is not None:
        totals.add(kind, seconds)


@contextmanager
def measure_scoring(kind: ScoringKind) -> Iterator[None]:
    """Add the block's duration to the current task's ``kind`` time, if any."""
    totals = _current_totals.get()
    if totals is None:
        yield
        return
    with totals.measure(kind):
        yield


async def measure_instance_scoring[T](coro: Coroutine[Any, Any, T], totals: ScoringTimes) -> T:
    """Await ``coro`` with ``totals`` current, adding its synchronous steps to compute time."""
    token = _current_totals.set(totals)
    try:
        return await _ComputeTimedCoroutine(coro, totals)
    finally:
        _current_totals.reset(token)


def install_child_task_timing() -> Callable[[], None]:
    """Step-time every task started while a task's ``ScoringTimes`` is current.

    Installs a task factory on the running loop and returns a function that removes it.
    Steps of different tasks never overlap, so none is counted twice. A loop that already
    has a task factory, which might run a child's first step inside its parent's, is left
    alone, and child tasks then go untimed.
    """
    loop = asyncio.get_running_loop()
    if loop.get_task_factory() is not None:
        return lambda: None

    def factory(loop: asyncio.AbstractEventLoop, coro: Any, **kwargs: Any) -> asyncio.Task:
        context = kwargs.get("context")
        totals = context.get(_current_totals) if context is not None else _current_totals.get()
        if totals is not None:
            coro = _time_compute(coro, totals)
        # On Python 3.13, create_task names an unnamed factory-made task "None";
        # nothing here can prevent that.
        return asyncio.Task(coro, loop=loop, **kwargs)

    loop.set_task_factory(factory)
    return lambda: loop.set_task_factory(None)


async def _time_compute[T](coro: Coroutine[Any, Any, T], totals: ScoringTimes) -> T:
    """Wrap `_ComputeTimedCoroutine` in a coroutine, which `asyncio.Task` requires."""
    return await _ComputeTimedCoroutine(coro, totals)


class _ComputeTimedCoroutine[T]:
    """Drives a coroutine step by step, adding each step's duration to compute time."""

    def __init__(self, coro: Coroutine[Any, Any, T], totals: ScoringTimes) -> None:
        self._coro = coro
        self._totals = totals

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
                self._totals.add("compute", _now() - start)
            try:
                value, error = (yield yielded), None
            except BaseException as exc:  # forwarded into the coroutine, e.g. cancellation
                value, error = None, exc


__all__ = [
    "ScoringKind",
    "ScoringTimes",
    "add_scoring_seconds",
    "install_child_task_timing",
    "measure_instance_scoring",
    "measure_scoring",
]

"""Time spent in each part of scoring, summed per task.

The async runner scores each instance through ``timed_scoring``, which makes the task's
``ScoringTimes`` current for that instance, and installs ``time_child_tasks`` so that
tasks the scorers start (``asyncio.gather``, ``create_task``) are timed too.

- Judge and sandbox call sites wrap each call in ``timed("judge")`` or
  ``timed("sandbox")``. A call is timed from when it is made until it returns, so it
  includes the client's own retries and connection waits, but not waits for a
  scorer-level judge or sandbox slot.
- Compute is the event loop's time running the instance's scoring code between awaits,
  in every task it starts, plus scoring work run in a process pool or a thread. A pool
  worker's start-up is left out by ``ProcessScorer.warmup``, though a library's
  first-call costs for inputs the warmup didn't cover still count.

Each piece is timed on its own and the durations are summed, so concurrent calls each
count in full: a task's judge time can exceed the wall time its scoring took. A call
also lasts longer while the event loop is busy with other instances' compute. The
client-side work of a judge or sandbox call counts both in that call's time and as
compute, and a background task a library starts while scoring an instance counts as
that instance's compute. Calls made outside scoring, such as an agent rollout's tool
calls in an inference worker, or an external benchmark's own judge, are not recorded.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine, Generator, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

Component = Literal["judge", "sandbox", "compute"]

_now = time.perf_counter


@dataclass
class ScoringTimes:
    """Summed seconds of one task's scoring, by component."""

    judge_seconds: float = 0.0
    sandbox_seconds: float = 0.0
    compute_seconds: float = 0.0

    def add(self, component: Component, seconds: float) -> None:
        attr = f"{component}_seconds"
        setattr(self, attr, getattr(self, attr) + seconds)

    @contextmanager
    def timing(self, component: Component) -> Iterator[None]:
        """Add the block's duration to ``component``."""
        start = _now()
        try:
            yield
        finally:
            self.add(component, _now() - start)


_current: ContextVar[ScoringTimes | None] = ContextVar("scoring_times", default=None)


def add_time(component: Component, seconds: float) -> None:
    """Add ``seconds`` to the current task's ``component`` time, if any."""
    times = _current.get()
    if times is not None:
        times.add(component, seconds)


@contextmanager
def timed(component: Component) -> Iterator[None]:
    """Add the block's duration to the current task's ``component`` time, if any."""
    times = _current.get()
    if times is None:
        yield
        return
    with times.timing(component):
        yield


async def timed_scoring[T](coro: Coroutine[Any, Any, T], times: ScoringTimes) -> T:
    """Await ``coro`` with ``times`` current, adding its synchronous steps to compute time."""
    token = _current.set(times)
    try:
        return await _StepTimed(coro, times)
    finally:
        _current.reset(token)


def time_child_tasks() -> Callable[[], None]:
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
        times = context.get(_current) if context is not None else _current.get()
        if times is not None:
            coro = _step_timed(coro, times)
        # Python 3.13's create_task names a factory-made task "None" when given no name.
        return asyncio.Task(coro, loop=loop, **kwargs)

    loop.set_task_factory(factory)
    return lambda: loop.set_task_factory(None)


async def _step_timed[T](coro: Coroutine[Any, Any, T], times: ScoringTimes) -> T:
    return await _StepTimed(coro, times)


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


__all__ = ["ScoringTimes", "add_time", "time_child_tasks", "timed", "timed_scoring"]

"""Run-wide shutdown request, set when SIGTERM arrives (for example, Beaker preemption).

The CLI's SIGTERM handler sets the request before raising :class:`Terminated`, so code
that is still running while the run unwinds (scoring tasks, progress records) can see
that its results arrived during shutdown and must not be kept as final.
"""

from __future__ import annotations

import threading

_requested = threading.Event()


class Terminated(BaseException):
    """SIGTERM arrived while a run was in progress (for example, Beaker preemption)."""


def request_shutdown() -> None:
    _requested.set()


def clear_shutdown() -> None:
    _requested.clear()


def shutdown_requested() -> bool:
    return _requested.is_set()

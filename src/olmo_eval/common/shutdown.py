"""Run-wide shutdown request, set when SIGTERM arrives (for example, Beaker preemption).

The CLI's SIGTERM handler sets the request before raising :class:`Terminated`, so code
that is still running while the run unwinds (scoring tasks, progress records) can see
that its results arrived during shutdown and must not be kept as final.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import time

_requested = threading.Event()

# How long a run that has ended may take to exit. Exiting joins every non-daemon thread
# and child process; one that never ends (a provider worker or engine stuck in startup)
# would otherwise keep the process, and the Beaker job holding its GPUs, alive forever.
EXIT_GRACE_SECONDS = 120.0


class Terminated(BaseException):
    """SIGTERM arrived while a run was in progress (for example, Beaker preemption)."""


def request_shutdown() -> None:
    _requested.set()


def clear_shutdown() -> None:
    _requested.clear()


def shutdown_requested() -> bool:
    return _requested.is_set()


def exit_blockers() -> list[str]:
    """Threads and child processes that Python waits for before the process can exit."""
    current = threading.current_thread()
    blockers = [
        f"thread {t.name!r}"
        for t in threading.enumerate()
        if not t.daemon and t is not current and t is not threading.main_thread()
    ]
    try:
        children = multiprocessing.active_children()
    except Exception:  # noqa: BLE001 - reporting only; the exit must not depend on it
        children = []
    blockers += [f"process {p.name!r} (pid {p.pid})" for p in children]
    return blockers


def force_exit_after(code: int, grace: float = EXIT_GRACE_SECONDS) -> None:
    """Make sure the process exits with `code`, even if a thread or child blocks the exit.

    Call it just before raising SystemExit(code). A daemon thread waits `grace` seconds;
    if the interpreter is then still shutting down, it reports what the exit is waiting
    for, kills the child processes left (they would otherwise keep running, holding their
    GPUs), and ends the process with os._exit(code). A process that exits normally never
    gets there, and neither does a caller that catches the SystemExit (a test invoking
    the command in-process): its main thread is still running, so nothing is forced.
    """

    def watchdog() -> None:
        time.sleep(grace)
        # Python stops the main thread before it joins the other threads and children.
        if threading.main_thread().is_alive():
            return
        try:
            blockers = ", ".join(exit_blockers()) or "nothing found"
            sys.stdout.flush()
            print(
                f"Process still alive {grace:.0f}s after exit {code}; forcing it. "
                f"Waiting on: {blockers}",
                file=sys.stderr,
                flush=True,
            )
            for child in multiprocessing.active_children():
                child.kill()
        finally:
            os._exit(code)

    threading.Thread(target=watchdog, name="exit-watchdog", daemon=True).start()

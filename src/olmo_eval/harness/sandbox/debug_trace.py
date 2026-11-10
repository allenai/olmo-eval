"""Debug-only tracing of sandbox calls and event-loop stalls.

Enabled by OLMO_EVAL_SANDBOX_TRACE=<directory>; does nothing when it is unset. Writes, per
process:

- ``calls-<pid>.jsonl``: one line per sandbox transport attempt (wall-clock start, elapsed,
  sandbox, operation, attempt, outcome and exception type).
- ``lag-<pid>.jsonl``: one line whenever the event loop wakes a 100 ms sleep more than
  200 ms late.
- ``stalls-<pid>.jsonl``: when the loop misses its heartbeat for 500 ms, the loop thread's
  stack at that moment (once per stall), to name the code that blocks it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
import traceback
from typing import Any

TRACE_ENV = "OLMO_EVAL_SANDBOX_TRACE"

_TICK = 0.1
_LAG_THRESHOLD = 0.2
_STALL_THRESHOLD = 0.5

_lock = threading.Lock()
_files: dict[str, Any] = {}
_monitored_loops: set[int] = set()


def enabled() -> bool:
    return bool(os.environ.get(TRACE_ENV))


def _write(kind: str, record: dict[str, Any]) -> None:
    directory = os.environ.get(TRACE_ENV)
    if not directory:
        return
    with _lock:
        fh = _files.get(kind)
        if fh is None:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(directory, f"{kind}-{os.getpid()}.jsonl")
            fh = open(path, "a", buffering=1)  # noqa: SIM115 (kept open for the process)
            _files[kind] = fh
        fh.write(json.dumps(record) + "\n")


def record_call(**record: Any) -> None:
    """Record one sandbox transport attempt."""
    if enabled():
        _write("calls", record)


def start_loop_monitor() -> None:
    """Start the lag monitor and stall watchdog for the running loop (once per loop)."""
    if not enabled():
        return
    loop = asyncio.get_running_loop()
    if id(loop) in _monitored_loops:
        return
    _monitored_loops.add(id(loop))
    loop_thread = threading.get_ident()
    heartbeat = [time.monotonic()]

    async def tick() -> None:
        while True:
            start = time.monotonic()
            await asyncio.sleep(_TICK)
            now = time.monotonic()
            heartbeat[0] = now
            lag = now - start - _TICK
            if lag > _LAG_THRESHOLD:
                _write("lag", {"ts": time.time(), "lag": round(lag, 3)})

    def watchdog() -> None:
        reported = None
        while True:
            time.sleep(_TICK)
            beat = heartbeat[0]
            if time.monotonic() - beat < _STALL_THRESHOLD or reported == beat:
                continue
            reported = beat
            frame = sys._current_frames().get(loop_thread)
            stack = traceback.format_stack(frame)[-25:] if frame is not None else []
            _write(
                "stalls",
                {
                    "ts": time.time(),
                    "since_beat": round(time.monotonic() - beat, 3),
                    "stack": [s.strip() for s in stack],
                },
            )

    loop.create_task(tick())
    threading.Thread(target=watchdog, name="sandbox-trace-watchdog", daemon=True).start()

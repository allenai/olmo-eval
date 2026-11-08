"""force_exit_after: a run that has ended exits, even when a thread or child would block it."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

_STUCK_THREAD = """
import threading
from olmo_eval.common.shutdown import force_exit_after

threading.Thread(target=threading.Event().wait, name="stuck").start()
force_exit_after(3, grace={grace})
raise SystemExit(3)
"""

_STUCK_CHILD = """
import multiprocessing as mp
import time
from olmo_eval.common.shutdown import force_exit_after


def hang():
    time.sleep(3600)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    ctx.Process(target=hang, name="stuck-worker").start()
    force_exit_after(1, grace={grace})
    raise SystemExit(1)
"""

_CAUGHT = """
import time
from olmo_eval.common.shutdown import force_exit_after

try:
    force_exit_after(1, grace=0.5)
    raise SystemExit(1)
except SystemExit:
    pass
time.sleep(2)
print("still running")
"""

_CLEAN = """
from olmo_eval.common.shutdown import force_exit_after

force_exit_after(2, grace=60)
raise SystemExit(2)
"""


def _run(source: str, tmp_path) -> tuple[subprocess.CompletedProcess[str], float]:
    script = tmp_path / "script.py"
    script.write_text(textwrap.dedent(source))
    start = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    return result, time.monotonic() - start


def test_a_stuck_thread_cannot_keep_the_process_alive(tmp_path) -> None:
    result, _ = _run(_STUCK_THREAD.format(grace=0.5), tmp_path)

    assert result.returncode == 3
    assert "forcing it" in result.stderr
    assert "thread 'stuck'" in result.stderr


def test_a_stuck_child_process_cannot_keep_the_process_alive(tmp_path) -> None:
    result, _ = _run(_STUCK_CHILD.format(grace=1.0), tmp_path)

    assert result.returncode == 1
    assert "process 'stuck-worker'" in result.stderr


def test_a_clean_exit_is_not_delayed_or_reported(tmp_path) -> None:
    result, elapsed = _run(_CLEAN, tmp_path)

    assert result.returncode == 2
    assert "forcing it" not in result.stderr
    assert elapsed < 30


def test_a_caller_that_catches_the_exit_is_left_running(tmp_path) -> None:
    result, _ = _run(_CAUGHT, tmp_path)

    assert result.returncode == 0
    assert "still running" in result.stdout
    assert "forcing it" not in result.stderr

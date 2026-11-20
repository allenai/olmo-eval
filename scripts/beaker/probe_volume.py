"""Exercise POSIX operations needed by resume storage on a disposable volume path.

Run in a Beaker task with a read-write native volume mounted at /eval-cache.
The optional directory argument allows separate jobs to check the same payload.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    path = root / "payload"
    payload = b"olmo-eval-native-volume-probe\n" * 32768
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    assert path.read_bytes() == payload
    lock = root / "probe.lock"
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = (
            "import fcntl,sys; f=open(sys.argv[1], 'a+'); "
            "fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)"
        )
        denied = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True)
        assert denied.returncode != 0 and b"BlockingIOError" in denied.stderr
    subprocess.run([sys.executable, "-c", code, str(lock)], check=True)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code + "; print('locked', flush=True); import time; time.sleep(60)",
            str(lock),
        ],
        stdout=subprocess.PIPE,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == b"locked"
    holder.kill()
    holder.wait(timeout=10)
    subprocess.run([sys.executable, "-c", code, str(lock)], check=True)
    print(
        json.dumps(
            {
                "host": socket.gethostname(),
                "node": os.environ.get("BEAKER_NODE_HOSTNAME"),
                "job": os.environ.get("BEAKER_JOB_ID"),
                "root": str(root),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started,
                "fsync_rename_integrity_lock_and_kill_release": "passed",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

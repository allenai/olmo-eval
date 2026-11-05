"""Own a shared WEKA run directory across Beaker replacement jobs.

The outer advisory lock covers the whole run, including its manifest and upload. The
child inherits its descriptor so killing only this supervisor cannot let another writer
start while the child is alive. Lock files are never unlinked while running.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.constants.infrastructure import SHARED_RUN_ENV

if TYPE_CHECKING:
    from collections.abc import Sequence

RETENTION_SECONDS = 7 * 24 * 3600
REFRESH_SECONDS = 3600


def write_atomic(path: Path, value: Any) -> None:
    """Publish JSON durably on the same filesystem."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def refresh_files(directory: Path) -> None:
    """Extend the prune lifetime of every recovery file, without following links."""
    for root, directories, names in os.walk(directory, followlinks=False):
        directories[:] = [d for d in directories if not (Path(root) / d).is_symlink()]
        for name in names:
            path = Path(root) / name
            if not path.is_symlink():
                with contextlib.suppress(FileNotFoundError):
                    os.utime(path, follow_symlinks=False)


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_shared(
    root: Path,
    run_uuid: str,
    results: Path,
    command: Sequence[str],
    *,
    refresh_seconds: float = REFRESH_SECONDS,
    volume: str | None = None,
) -> int:
    """Run once or resume under exclusive ownership; return the child's exit status."""
    if str(uuid.UUID(run_uuid)) != run_uuid:
        raise ValueError("run UUID must be a canonical UUID")
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Resume root {root} is not mounted; refusing local fallback")
    root = root / "tmp-7d"
    root.mkdir(mode=0o755, exist_ok=True)
    directory = root / run_uuid
    directory.mkdir(mode=0o700, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    receipt_path = results / "weka-resume.json"
    session = directory / "session.json"
    output = directory / "output"
    with (directory / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another attempt owns WEKA run {run_uuid} ({directory})") from e
        if session.exists():
            if time.time() - session.stat().st_mtime >= RETENTION_SECONDS:
                raise RuntimeError("WEKA recovery state expired; submit a new evaluation")
            session_data = json.loads(session.read_text())
            if session_data["command"] != list(command):
                raise RuntimeError("WEKA run command changed; submit a new evaluation")
        elif receipt_path.exists() or output.exists():
            raise RuntimeError("WEKA recovery session is missing; submit a new evaluation")
        else:
            session_data = {"uuid": run_uuid, "command": list(command)}
            write_atomic(session, session_data)
        output.mkdir(mode=0o700, exist_ok=True)
        receipt = {
            "uuid": run_uuid,
            "output_dir": str(output),
            "status": "running",
            "volume": volume,
        }
        complete = directory / "complete.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            for name, digest in saved["artifacts"].items():
                path = output / name
                if not path.is_file() or _checksum(path) != digest:
                    raise RuntimeError(f"Completed WEKA run is missing or changed: {path}")
            write_atomic(
                receipt_path,
                {**receipt, "status": "complete", "run_id": saved.get("run_id"), "exit_code": 0},
            )
            return 0
        if session_data.get("completed"):
            raise RuntimeError("WEKA completion record is missing; submit a new evaluation")
        # This directory belongs to one submission. The outer lock proves the old
        # writer is gone, including on another node, so its PID lock can be reclaimed.
        for stale in (output / "progress").glob("*/.lock"):
            stale.unlink()
        refresh_files(directory)
        write_atomic(receipt_path, receipt)
        print(f"WEKA run {run_uuid}: {output}", flush=True)
        stopped = threading.Event()
        errors: list[OSError] = []
        child = subprocess.Popen(
            [*command, "-O", str(output)],
            pass_fds=(lock.fileno(),),
            start_new_session=True,
            env={**os.environ, SHARED_RUN_ENV: str(directory)},
        )

        def forward(signum: int, _frame: Any) -> None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signum)

        def keep_alive() -> None:
            while not stopped.wait(refresh_seconds):
                try:
                    refresh_files(directory)
                except OSError as e:
                    errors.append(e)
                    forward(signal.SIGTERM, None)
                    return

        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT)}
        heartbeat = threading.Thread(target=keep_alive, daemon=True)
        heartbeat.start()
        try:
            code = child.wait()
        finally:
            stopped.set()
            heartbeat.join()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if errors:
            raise RuntimeError(f"WEKA retention refresh failed: {errors[0]}")
        code = 128 - code if code < 0 else code
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            receipt["run_id"] = json.loads(manifest_path.read_text()).get("run", {}).get("run_id")
        if code == 0:
            artifacts = {
                str(path.relative_to(output)): _checksum(path)
                for path in output.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and "progress" not in path.relative_to(output).parts
            }
            write_atomic(complete, {"artifacts": artifacts, "run_id": receipt.get("run_id")})
            write_atomic(session, {**session_data, "completed": True})
            if (output / "progress").exists():
                shutil.rmtree(output / "progress")
        refresh_files(directory)
        write_atomic(
            receipt_path,
            {
                **receipt,
                "status": "complete"
                if code == 0
                else "interrupted"
                if code in (130, 143)
                else "failed",
                "exit_code": code,
            },
        )
        return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-uuid", required=True)
    parser.add_argument("--volume")
    parser.add_argument("--results", type=Path, default=Path("/results"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.path.ismount(args.root):
        parser.error(f"{args.root} is not a volume mount; refusing local fallback")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an evaluation command is required")
    raise SystemExit(
        run_shared(args.root, args.run_uuid, args.results, command, volume=args.volume)
    )


if __name__ == "__main__":
    main()

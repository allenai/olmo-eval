"""Own a shared WEKA run directory across Beaker replacement jobs.

The eval writes its outputs to Beaker results (``/results``), as it does without resume.
The WEKA directory holds only what a replacement job needs to continue: progress records
(``progress/``) and the run's identity (``identity/``, its manifest and upload secret).

The outer advisory lock covers the whole run, including its manifest and upload. The
child inherits its descriptor so killing only this supervisor cannot let another writer
start while the child is alive. Lock files are never unlinked while running.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import signal
import stat
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from olmo_eval.common.constants.infrastructure import (
    BEAKER_DEFAULT_RESUME_RETENTION_DAYS,
    SHARED_RUN_ENV,
    SHARED_RUN_IDENTITY_DIR,
    SHARED_RUN_IDENTITY_FILES,
    SHARED_RUN_PROGRESS_DIR,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

RETENTION_SECONDS = BEAKER_DEFAULT_RESUME_RETENTION_DAYS * 24 * 3600
REFRESH_SECONDS = 3600


class SharedRunSession(TypedDict):
    uuid: str
    command: list[str]
    progress_streams: NotRequired[list[str]]
    completed: NotRequired[bool]


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


def _files(directory: Path, *, allow_missing: bool = False) -> Iterator[Path]:
    """Enumerate regular files without hiding unreadable subtrees or following links."""

    def fail(error: OSError) -> None:
        if allow_missing and isinstance(error, FileNotFoundError):
            return
        raise error

    for root, directories, names in os.walk(directory, followlinks=False, onerror=fail):
        directories[:] = [d for d in directories if not (Path(root) / d).is_symlink()]
        for name in names:
            path = Path(root) / name
            try:
                mode = path.stat(follow_symlinks=False).st_mode
            except FileNotFoundError:
                if allow_missing:
                    continue
                raise
            if stat.S_ISREG(mode):
                yield path


def refresh_files(directory: Path) -> None:
    """Extend every recovery file's prune lifetime; traversal failures stop the run."""
    for path in _files(directory, allow_missing=True):
        with contextlib.suppress(FileNotFoundError):
            os.utime(path, follow_symlinks=False)


def _manifest_run(results: Path) -> dict[str, Any]:
    try:
        return json.loads((results / "manifest.json").read_text()).get("run") or {}
    except (OSError, ValueError):
        return {}


def _has_complete_results(results: Path, run_id: str | None) -> bool:
    """Whether ``results`` holds the final outputs of the completed run ``run_id``."""
    run = _manifest_run(results)
    return (
        run.get("run_id") == run_id
        and run.get("status") in ("complete", "partial")
        and (results / "metrics.json").is_file()
    )


def _save_identity(results: Path, saved: Path) -> None:
    for name in SHARED_RUN_IDENTITY_FILES:
        if (results / name).is_file():
            saved.mkdir(mode=0o700, exist_ok=True)
            tmp = saved / f".{name}.tmp"
            shutil.copy2(results / name, tmp)
            os.replace(tmp, saved / name)


def run_shared(
    root: Path,
    run_uuid: str,
    results: Path,
    command: Sequence[str],
    *,
    refresh_seconds: float = REFRESH_SECONDS,
    volume: str | None = None,
    retention_days: int = BEAKER_DEFAULT_RESUME_RETENTION_DAYS,
) -> int:
    """Run once or resume under exclusive ownership; return the child's exit status."""
    if str(uuid.UUID(run_uuid)) != run_uuid:
        raise ValueError("run UUID must be a canonical UUID")
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Resume root {root} is not mounted; refusing local fallback")
    if type(retention_days) is not int or retention_days < 1:
        raise ValueError("retention_days must be a positive integer")
    root = root / f"tmp-{retention_days}d"
    root.mkdir(mode=0o755, exist_ok=True)
    directory = root / run_uuid
    directory.mkdir(mode=0o700, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    session = directory / "session.json"
    progress = directory / SHARED_RUN_PROGRESS_DIR
    session_data: SharedRunSession
    with (directory / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f"Another attempt owns WEKA run {run_uuid} ({directory})") from e
        if session.exists():
            if time.time() - session.stat().st_mtime >= retention_days * 24 * 3600:
                raise RuntimeError("WEKA recovery state expired; submit a new evaluation")
            session_data = json.loads(session.read_text())
            if session_data["command"] != list(command):
                raise RuntimeError("WEKA run command changed; submit a new evaluation")
        elif any(
            (directory / name).exists()
            for name in (SHARED_RUN_PROGRESS_DIR, SHARED_RUN_IDENTITY_DIR, "complete.json")
        ):
            raise RuntimeError("WEKA recovery session is missing; submit a new evaluation")
        else:
            session_data = {"uuid": run_uuid, "command": list(command)}
            write_atomic(session, session_data)
        complete = directory / "complete.json"
        if complete.exists():
            run_id = json.loads(complete.read_text()).get("run_id")
            if _has_complete_results(results, run_id):
                # Beaker reran a job whose eval finished, and its results came along.
                print(f"WEKA run {run_uuid} already completed as run {run_id}", flush=True)
                return 0
            # This job's results lack the completed run's outputs (Beaker carried over an
            # earlier snapshot, or none). Run the eval again, as a job without resume
            # would; the saved identity keeps the run ID. Its progress is gone.
            print(
                f"WEKA run {run_uuid} completed as run {run_id}, but this job's results do "
                "not hold its outputs; running it again",
                flush=True,
            )
            session_data = {"uuid": run_uuid, "command": list(command)}
            write_atomic(session, session_data)
            complete.unlink()
        if session_data.get("completed"):
            raise RuntimeError("WEKA completion record is missing; submit a new evaluation")
        for relative in session_data.get("progress_streams", []):
            stream = directory / relative
            if (
                not stream.resolve().is_relative_to(directory.resolve())
                or not stream.is_dir()
                or not (stream / "index.json").is_file()
            ):
                raise RuntimeError(f"WEKA recovery stream is missing: {stream}")
        # This directory belongs to one submission. The outer lock proves the old
        # writer is gone, including on another node, so its PID lock can be reclaimed.
        for stale in progress.glob("*/.lock"):
            stale.unlink()
        refresh_files(directory)
        print(f"WEKA run {run_uuid}: recovery state in {directory}", flush=True)
        stopped = threading.Event()
        errors: list[OSError] = []
        # The eval writes its outputs to Beaker results, as it does without resume. Only
        # its progress records and run identity go to the volume (SHARED_RUN_ENV).
        child = subprocess.Popen(
            [*command, "-O", str(results)],
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
        # Keep the newest identity, including an upload secret created after the run
        # started (by the final upload, or when marking the run interrupted).
        _save_identity(results, directory / SHARED_RUN_IDENTITY_DIR)
        if code == 0:
            write_atomic(complete, {"run_id": _manifest_run(results).get("run_id")})
            write_atomic(session, {**json.loads(session.read_text()), "completed": True})
            # The identity stays (it is small) so a rerun of this completed job keeps
            # the run ID; retention prunes it.
            if progress.exists():
                shutil.rmtree(progress)
        refresh_files(directory)
        return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-uuid", required=True)
    parser.add_argument("--volume")
    parser.add_argument("--retention-days", type=int, default=BEAKER_DEFAULT_RESUME_RETENTION_DAYS)
    parser.add_argument("--results", type=Path, default=Path("/results"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not os.path.ismount(args.root):
        parser.error(f"{args.root} is not a volume mount; refusing local fallback")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an evaluation command is required")
    raise SystemExit(
        run_shared(
            args.root,
            args.run_uuid,
            args.results,
            command,
            volume=args.volume,
            retention_days=args.retention_days,
        )
    )


if __name__ == "__main__":
    main()

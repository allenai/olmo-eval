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
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

RETENTION_SECONDS = BEAKER_DEFAULT_RESUME_RETENTION_DAYS * 24 * 3600
REFRESH_SECONDS = 3600


class SharedRunSession(TypedDict):
    uuid: str
    command: list[str]
    output_initialized: NotRequired[bool]
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


def _checksum(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _artifacts(output: Path) -> list[Path]:
    """Return the run's result files, leaving out recovery progress."""
    return [
        path.relative_to(output)
        for path in _files(output)
        if "progress" not in path.relative_to(output).parts
    ]


def copy_results(output: Path, results: Path, artifacts: Iterable[Path]) -> None:
    """Copy result files from WEKA into Beaker ``/results`` with the usual layout.

    WEKA recovery files are pruned after the retention window, so Beaker results keep
    the durable copy of predictions, metrics, and logs, as they did before resume.
    Each file is written to a temporary name and renamed, so a periodic Beaker upload
    never sees a partial file under its final name.
    """
    for relative in artifacts:
        target = results / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.copying")
        try:
            shutil.copy2(output / relative, tmp)
            tmp.replace(target)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise


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
    receipt_path = results / "weka-resume.json"
    session = directory / "session.json"
    output = directory / "output"
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
        elif receipt_path.exists() or output.exists():
            raise RuntimeError("WEKA recovery session is missing; submit a new evaluation")
        else:
            session_data = {"uuid": run_uuid, "command": list(command), "output_initialized": False}
            write_atomic(session, session_data)
        if session_data.get("output_initialized", True) and not output.is_dir():
            raise RuntimeError("WEKA recovery output is missing; submit a new evaluation")
        output.mkdir(mode=0o700, exist_ok=True)
        if not session_data.get("output_initialized", True):
            session_data["output_initialized"] = True
            write_atomic(session, session_data)
        receipt = {
            "uuid": run_uuid,
            "output_dir": str(output),
            "status": "running",
            "volume": volume,
            "retention_days": retention_days,
        }
        complete = directory / "complete.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            for name, digest in saved["artifacts"].items():
                path = output / name
                if not path.is_file() or _checksum(path) != digest:
                    raise RuntimeError(f"Completed WEKA run is missing or changed: {path}")
            copy_results(output, results, map(Path, saved["artifacts"]))
            write_atomic(
                receipt_path,
                {**receipt, "status": "complete", "run_id": saved.get("run_id"), "exit_code": 0},
            )
            return 0
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
            artifacts = {str(path): _checksum(output / path) for path in _artifacts(output)}
            write_atomic(complete, {"artifacts": artifacts, "run_id": receipt.get("run_id")})
            write_atomic(session, {**json.loads(session.read_text()), "completed": True})
            if (output / "progress").exists():
                shutil.rmtree(output / "progress")
            copy_results(output, results, map(Path, artifacts))
        elif code not in (130, 143):
            # Keep a failed attempt's logs and partial predictions for debugging. An
            # interrupted attempt skips the copy to shut down quickly; its replacement
            # copies the final results.
            try:
                copy_results(output, results, _artifacts(output))
            except OSError as e:
                # Report the eval's own failure, not the copy's.
                print(f"Could not copy failed run's output to {results}: {e}", flush=True)
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

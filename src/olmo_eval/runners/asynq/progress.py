"""Progress records that let a restarted run skip the work it already finished.

A run started with ``--resume`` records progress under ``<output_dir>/progress/<model>/``
and continues from it when it is there. The records are one run-level stream of
tagged entries (``run.segments``) plus one ``<spec>_<hash6>.final.json``
per finalized task. For each task the stream has a header naming what the task
measures (task, model, harness, engine and code) and where it ran, then one entry per
finished instance, written by the runner's main process once the instance's outcome
is final. A resumed run keeps the finalized tasks, rebuilds the recorded instances of
the others, and queues only the rest, so its predictions and metrics are the same as
an uninterrupted run's. A run that completes deletes its own tasks' records, and they
are never uploaded. Other runs may share the output directory: their tasks' records are
kept, and a lock file in the model's directory (:class:`ProgressLock`) stops two runs from
recording into the same stream at once.

An entry keeps what scoring produced (outputs, scores, the provider's request trace
and instance metadata that scoring changed), not the prompt: the instance and request
are rebuilt by preparing the task again, and a short hash of the prepared instance
(request, sampling parameters, gold answer and metadata) checks that they are the same.
A generation's per-token logprobs are left out when nothing after scoring reads them
(see :func:`keeps_logprobs`); its output metadata keeps the summaries the predictions
writer reads (``sum_logits``, ``num_tokens``).

An instance's outcome is final, and recorded, when the uninterrupted run would keep
it too: a scored response (including one some scorer could not score, which the
metrics leave out), or a generation failure the request itself caused
(``ResultItem.request_caused``, such as a context overflow). Not recorded, so a
resumed run tries them again: every other generation failure (providers report
connection and server errors as marked outputs, which cannot be told from request
errors), a scoring failure flagged as infrastructure (the judge or sandbox was
unavailable), and anything that arrives after shutdown was requested. A task that
failed (an error or an exceeded failure budget) is not finalized, so a retry computes
it again from its recorded instances.

:class:`ProgressStore` writes the buffered entries to a new numbered file every few
minutes (or at 8 MB) with an atomic rename, so a results uploader that re-sends changed
files sends each segment once and a long run makes hundreds of files, not millions.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import pickle
import shutil
import signal
import socket
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from olmo_eval.common.constants.infrastructure import BEAKER_RESULT_DIR
from olmo_eval.common.logging import get_logger
from olmo_eval.common.scoring_failures import output_scoring_errors
from olmo_eval.common.shutdown import shutdown_requested
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.runners.common.types import TaskResult
from olmo_eval.runners.io.writers import output_path
from olmo_eval.runners.processing.utils import compute_task_hash, sanitize_spec_for_filename

if TYPE_CHECKING:
    from olmo_eval.common.types import LogProbEntry
    from olmo_eval.evals.tasks.common import Task
    from olmo_eval.harness.config import HarnessConfig
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker

logger = get_logger(__name__)

PROGRESS_DIR = "progress"
DEBUG_EXIT_AT_ENV = "OLMO_EVAL_DEBUG_EXIT_AT"
RECORD_VERSION = 6
FINAL_SUFFIX = ".final.json"
LOCK_NAME = ".lock"

# The segment stream is written when its buffered entries reach this size or age, and on
# close. Beaker uploads /results every few minutes, so writing more often buys little.
SEGMENT_MAX_BYTES = 8 * 1024 * 1024
SEGMENT_MAX_SECONDS = 120.0

# Header fields that decide what an instance's record means; a record that differs
# in one cannot be resumed. The others describe where the run executed, and only warn.
MEASUREMENT_FIELDS = (
    "version",
    "spec",
    "task_hash",
    # Before model_hash, which covers the revision too, so a changed revision is reported
    # by name instead of as an opaque hash.
    "model_revision",
    "model_hash",
    "harness_hash",
    "engine_version",
    "git_commit",
    "diff_hash",
)

# Harness settings that change how work is scheduled, not what an instance's
# record contains. The provider is covered by the model hash.
_OPERATIONAL_HARNESS_KEYS = frozenset(
    {
        "provider",
        "metrics",
        "max_concurrency",
        "scoring_concurrency",
        "scoring_process_pools",
        "sandbox_pool_instances",
        "sandbox_pool_min_instances",
        "batching",
        "scorer_startup_timeout",
        "max_hard_failure_rate",
        "required_secrets",
    }
)


class ProgressMismatchError(RuntimeError):
    """Progress recorded in the output directory cannot be used by this run."""


class ProgressLockedError(ProgressMismatchError):
    """Another run is recording progress for this model in the output directory."""


class DroppedLogprobsError(RuntimeError):
    """Code read the per-token logprobs of a restored output whose record left them out."""


class DroppedLogprobs:
    """Stands in for the per-token logprobs a record left out of a restored output.

    It has the list's length and truth value, which is what the predictions writer reads
    besides the summaries in the output's metadata. Reading an entry raises
    :class:`DroppedLogprobsError` rather than letting a metric compute from data the
    record does not hold, so a consumer :func:`keeps_logprobs` does not know about
    fails instead of changing a score.
    """

    __slots__ = ("_count",)

    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def _missing(self) -> DroppedLogprobsError:
        return DroppedLogprobsError(
            f"The {self._count} per-token logprobs of this restored output were not recorded. "
            "Mark the scorer that reads them with reads_logprobs = True, or pass --fresh."
        )

    def __iter__(self) -> Iterator[Any]:
        raise self._missing()

    def __getitem__(self, index: Any) -> Any:
        raise self._missing()

    def __repr__(self) -> str:
        return f"DroppedLogprobs({self._count})"


# ---------------------------------------------------------------------------
# Value encoding
# ---------------------------------------------------------------------------

# JSON keeps strings, numbers, lists and string-keyed dicts exactly. Other values are
# tagged so they come back as the same Python value: tuples, dicts with other keys,
# and (pickled) anything else, such as tool calls or numpy scalars.
_TUPLE = "__tuple__"
_ITEMS = "__items__"
_PICKLE = "__pickle__"
_TAGS = (_TUPLE, _ITEMS, _PICKLE)


def encode_value(value: Any) -> Any:
    """A JSON-serializable form of ``value`` that :func:`decode_value` turns back."""
    if value is None or type(value) in (bool, str, int, float):
        return value
    if type(value) is list:
        return [encode_value(item) for item in value]
    if type(value) is tuple:
        return {_TUPLE: [encode_value(item) for item in value]}
    if type(value) is dict:
        if all(type(key) is str for key in value) and not any(tag in value for tag in _TAGS):
            return {key: encode_value(item) for key, item in value.items()}
        return {_ITEMS: [[encode_value(k), encode_value(v)] for k, v in value.items()]}
    return {_PICKLE: base64.b64encode(pickle.dumps(value)).decode("ascii")}


def decode_value(value: Any) -> Any:
    """Invert :func:`encode_value`."""
    if isinstance(value, list):
        return [decode_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        if _TUPLE in value:
            return tuple(decode_value(item) for item in value[_TUPLE])
        if _ITEMS in value:
            return {decode_value(k): decode_value(v) for k, v in value[_ITEMS]}
        if _PICKLE in value:
            return pickle.loads(base64.b64decode(value[_PICKLE]))  # noqa: S301 - our own file
    return {key: decode_value(item) for key, item in value.items()}


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _read_lines(path: Path) -> Iterator[Any]:
    """Yield the entries of a JSONL file line by line.

    A last line without a newline was cut off mid-write; it is dropped and logged.
    """
    with open(path, "rb") as f:
        for number, line in enumerate(f, start=1):
            if not line.endswith(b"\n"):
                logger.warning(f"Dropping a partly written last line from {path}")
                break
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as e:
                raise ProgressMismatchError(
                    f"Progress record {path} line {number} is not valid JSON ({e}). "
                    "Pass --fresh to discard recorded progress."
                ) from e


def _dump(entry: Mapping[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"


class ProgressStore:
    """A run's progress stream: tagged entries in a directory of numbered JSONL segments.

    Each segment is written whole and renamed into place. Entries are buffered and written
    as a new segment once they reach ``SEGMENT_MAX_BYTES``, every ``SEGMENT_MAX_SECONDS``
    by a background thread, and on close.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._buffer: list[str] = []
        self._buffered_bytes = 0
        self._next_segment = 0
        self._lock = threading.Lock()
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    def read(self) -> Iterator[dict[str, Any]]:
        """Yield every complete entry in order; nothing when there is no record.

        Drops (and logs) an entry that was cut off while being written. Appends after a
        full read continue the record.
        """
        if not self.path.is_dir():
            return
        for leftover in self.path.glob("*.tmp"):
            leftover.unlink()
        segments = sorted(self.path.glob("[0-9]*.jsonl"))
        self._next_segment = int(segments[-1].stem) + 1 if segments else 0
        for segment in segments:
            yield from _read_lines(segment)

    def reset(self) -> None:
        """Discard every existing entry."""
        with self._lock:
            self._buffer, self._buffered_bytes = [], 0
            if self.path.exists():
                shutil.rmtree(self.path)
            self._next_segment = 0

    def drop(self, keys: Collection[str] = (), specs: Collection[str] = ()) -> None:
        """Remove the entries of the tasks in ``keys`` or whose header names a spec in ``specs``.

        The other tasks' entries are rewritten into one new segment, written to a temporary
        file and renamed into place after the old segments, which are then removed newest
        first. A crash leaves the old segments, or a prefix of them followed by the new
        segment; the new segment holds every entry that is kept, so either way a reader
        sees the kept tasks' records complete. The removed tasks' records may still be
        there, in the old segments, and are removed by the next drop. A reader restarts a
        task's decoding at each of its headers, so the prefix does not affect the new
        segment's entries. Nothing is left when no task's entries are.
        """
        with self._lock:
            self._buffer, self._buffered_bytes = [], 0
        segments = sorted(self.path.glob("[0-9]*.jsonl")) if self.path.is_dir() else []
        if not segments:
            return
        for leftover in self.path.glob("*.tmp"):
            leftover.unlink()
        dropped = set(keys)
        final = self.path / f"{int(segments[-1].stem) + 1:06d}.jsonl"
        tmp = final.with_suffix(".tmp")
        kept_tasks, removed_any = False, False
        with open(tmp, "w", encoding="utf-8") as f:
            for segment in segments:
                for entry in _read_lines(segment):
                    key = entry.get("t")
                    header = entry.get("header")
                    if isinstance(header, dict) and header.get("spec") in specs:
                        dropped.add(key)
                    if key in dropped:
                        removed_any = True
                        continue
                    kept_tasks = kept_tasks or key is not None
                    f.write(_dump(entry))
            f.flush()
            os.fsync(f.fileno())
        if not removed_any:
            tmp.unlink()
            return
        if kept_tasks:
            os.replace(tmp, final)
        else:
            tmp.unlink()
        for segment in reversed(segments):
            segment.unlink()
        self._next_segment = int(final.stem) + 1 if kept_tasks else 0
        if not kept_tasks:
            with contextlib.suppress(OSError):
                self.path.rmdir()

    def append(self, entry: Mapping[str, Any]) -> None:
        """Add one entry after those already read or written."""
        line = _dump(entry)
        with self._lock:
            self._buffer.append(line)
            self._buffered_bytes += len(line)
            if self._buffered_bytes >= SEGMENT_MAX_BYTES:
                self._write_segment()
            if self._thread is None:
                self._stop = threading.Event()
                self._thread = threading.Thread(
                    target=self._flush_periodically,
                    args=(self._stop,),
                    name="progress-flush",
                    daemon=True,
                )
                self._thread.start()

    def _flush_periodically(self, stop: threading.Event) -> None:
        while not stop.wait(SEGMENT_MAX_SECONDS):
            try:
                with self._lock:
                    self._write_segment()
            except OSError as e:
                logger.warning(f"Could not write progress segment in {self.path}: {e}")

    def _write_segment(self) -> None:
        """Write the buffer as the next segment. The caller holds the lock."""
        if not self._buffer:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        final = self.path / f"{self._next_segment:06d}.jsonl"
        tmp = final.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(self._buffer)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        self._next_segment += 1
        self._buffer, self._buffered_bytes = [], 0

    def close(self) -> None:
        """Write out and fsync everything appended so far."""
        with self._lock:
            stop, thread = self._stop, self._thread
            self._stop = self._thread = None
        if stop is not None and thread is not None:
            stop.set()
            thread.join()
        with self._lock:
            self._write_segment()


def _model_dir(output_dir: str | Path, model_name: str) -> Path:
    return Path(output_dir) / PROGRESS_DIR / sanitize_spec_for_filename(model_name)


def task_key(spec: str, task_hash: str) -> str:
    """The name of a task's records, beside its predictions' name."""
    return f"{sanitize_spec_for_filename(spec)}_{task_hash[-6:]}"


def final_path(output_dir: str | Path, model_name: str, spec: str, task_hash: str) -> Path:
    """Path of a task's finalized result."""
    return _model_dir(output_dir, model_name) / f"{task_key(spec, task_hash)}{FINAL_SUFFIX}"


def model_lock(output_dir: str | Path, model_name: str) -> ProgressLock:
    """The lock on one model's progress in the output directory."""
    return ProgressLock(_model_dir(output_dir, model_name))


def stream_dir(output_dir: str | Path, model_name: str) -> Path:
    """Path of the directory holding the run's progress stream."""
    return _model_dir(output_dir, model_name) / "run.segments"


def has_progress(output_dir: str | Path) -> bool:
    """Whether the output directory holds any progress record (a lock file is not one)."""
    path = Path(output_dir) / PROGRESS_DIR
    return path.is_dir() and any(p.is_file() and p.name != LOCK_NAME for p in path.rglob("*"))


def remove_empty_progress_dirs(output_dir: str | Path, model_name: str) -> None:
    """Remove the model's progress directory and ``progress/`` when nothing is left in them."""
    _remove_empty_dirs(_model_dir(output_dir, model_name), Path(output_dir) / PROGRESS_DIR)


def _remove_empty_dirs(*directories: Path) -> None:
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            break


def discard_progress(
    output_dir: str | Path,
    model_name: str,
    specs: Collection[str],
    lock: ProgressLock | None = None,
) -> None:
    """Delete the records of ``specs`` for one model, under any task hash.

    Other tasks' records and other models' progress in the output directory are kept.
    A stream that cannot be read is deleted whole, since its tasks cannot be told apart.
    The caller's ``lock`` on the model's directory is used when it holds one; otherwise
    the lock is taken for the deletion.
    """
    model_dir = _model_dir(output_dir, model_name)
    with contextlib.nullcontext() if lock is not None and lock.held else ProgressLock(model_dir):
        store = ProgressStore(stream_dir(output_dir, model_name))
        try:
            store.drop(specs=specs)
        except ProgressMismatchError as e:
            logger.warning(f"{e} Deleting the whole stream {store.path}.")
            shutil.rmtree(store.path, ignore_errors=True)
        for path in model_dir.glob(f"*{FINAL_SUFFIX}"):
            if _final_spec(path, specs) is not None:
                path.unlink()
    _remove_empty_dirs(model_dir, Path(output_dir) / PROGRESS_DIR)


def _final_spec(path: Path, specs: Collection[str] | None = None) -> str | None:
    """The spec a ``.final.json`` belongs to (in ``specs``, when given); None if not."""
    try:
        spec = json.loads(path.read_text())["header"]["spec"]
    except (OSError, ValueError, KeyError, TypeError):
        # Unreadable: go by the name, ``<spec>_<hash6>``.
        stem = path.name[: -len(FINAL_SUFFIX)]
        candidates = specs if specs is not None else [stem[:-7]]
        return next(
            (c for c in candidates if stem == f"{sanitize_spec_for_filename(c)}_{stem[-6:]}"),
            None,
        )
    return spec if specs is None or spec in specs else None


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class ProgressLock:
    """Marks a model's progress directory as in use by one recording run.

    The lock file holds the holder's host, PID, start time and Beaker job ID. A lock that
    cannot belong to a live run is taken over with a warning:

    - In a Beaker job's own result directory (``/results``), any lock is the last job's:
      the directory is copied into the replacement job and one job of a task runs at a
      time. This includes a lock left empty by a kill between its creation and its
      contents. A lock from the same Beaker job is also taken over wherever it is.
    - Elsewhere, a lock whose process is gone from this host. A lock from another host
      (or another Beaker job in a shared directory) cannot be checked, so it stops the
      run until its holder is gone and the file is deleted.

    Two runs that judge the same stale lock within milliseconds of each other can both
    proceed. This is accepted: Beaker runs one job per task and local runs are started by
    hand.
    """

    def __init__(self, model_dir: Path) -> None:
        self.path = model_dir / LOCK_NAME
        self._token: str | None = None

    def __enter__(self) -> ProgressLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    @property
    def held(self) -> bool:
        """Whether this object took the lock and has not released it."""
        return self._token is not None

    def _read(self) -> dict[str, Any] | None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def _in_job_results(self) -> bool:
        """Whether the lock's output directory is this Beaker job's own result directory."""
        if not os.environ.get("BEAKER_JOB_ID"):
            return False
        output_dir = self.path.parent.parent.parent
        try:
            return output_dir.resolve().is_relative_to(Path(BEAKER_RESULT_DIR).resolve())
        except OSError:
            return False

    def _stale(self, holder: Mapping[str, Any] | None) -> str | None:
        """Why the holder's lock can be taken over, or None when it may still be running."""
        job_id = os.environ.get("BEAKER_JOB_ID")
        if holder is not None:
            pid = holder.get("pid")
            if (
                holder.get("host") == socket.gethostname()
                and isinstance(pid, int)
                and pid != os.getpid()
                and _pid_alive(pid)
            ):
                return None
        if job_id and holder is not None and holder.get("beaker_job_id") == job_id:
            return f"it was left by an earlier attempt of this Beaker job ({job_id})"
        if self._in_job_results():
            if holder is None:
                return f"it was left unreadable by an earlier attempt, before Beaker job {job_id}"
            left_by = holder.get("beaker_job_id") or "an earlier run"
            return f"it was left by {left_by}, not this job ({job_id})"
        if holder is None:
            return None
        pid = holder.get("pid")
        if holder.get("host") == socket.gethostname() and isinstance(pid, int):
            return None if _pid_alive(pid) else f"process {pid} is no longer running"
        return None

    def _describe(self, holder: Mapping[str, Any] | None) -> str:
        if holder is None:
            return "an unreadable lock file"
        job = f", Beaker job {holder['beaker_job_id']}" if holder.get("beaker_job_id") else ""
        return (
            f"host {holder.get('host')}, pid {holder.get('pid')}, started "
            f"{holder.get('started')}{job}"
        )

    def _create(self) -> int:
        """Create the lock file exclusively, recreating a model directory removed meanwhile."""
        # Another run releasing its lock can remove the directory between the mkdir and
        # the open.
        for _ in range(5):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                return os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileNotFoundError:
                continue
        raise ProgressLockedError(
            f"{self.path.parent} kept disappearing while this run was taking the lock; "
            "another run may be removing it."
        )

    def acquire(self) -> None:
        """Take the lock, or raise :class:`ProgressLockedError` naming who holds it."""
        token = uuid.uuid4().hex
        mine = {
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "beaker_job_id": os.environ.get("BEAKER_JOB_ID") or None,
            "token": token,
        }
        for _ in range(2):
            try:
                fd = self._create()
            except FileExistsError:
                holder = self._read()
                reason = self._stale(holder)
                if reason is None:
                    raise ProgressLockedError(
                        f"Another run is recording progress here: {self.path} is held by "
                        f"{self._describe(holder)}. Two runs recording into the same output "
                        "directory and model would overwrite each other's progress. Wait for "
                        "it to finish, use another output directory (-O), or, if that run is "
                        "gone, delete the lock file."
                    ) from None
                logger.warning(f"Taking over {self.path} ({self._describe(holder)}): {reason}")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(mine, f)
                f.flush()
                os.fsync(f.fileno())
            # Two runs that both judged a lock stale can each remove the other's new one.
            # The last to write is the holder; the other finds its token gone.
            holder = self._read()
            if holder is None or holder.get("token") != token:
                raise ProgressLockedError(
                    f"Another run took {self.path} while this one was taking it "
                    f"({self._describe(holder)}). Wait for it to finish, or use another "
                    "output directory (-O)."
                )
            self._token = token
            return
        raise ProgressLockedError(
            f"Another run took {self.path} while this one was reclaiming it "
            f"({self._describe(self._read())})."
        )

    def release(self) -> None:
        """Remove the lock file if this run still holds it."""
        token, self._token = self._token, None
        if token is None:
            return
        holder = self._read()
        if holder is not None and holder.get("token") == token:
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------


def harness_hash(harness_config: HarnessConfig) -> str:
    """Hash of the harness settings that can change what an instance's outputs are."""
    config = {
        key: value
        for key, value in harness_config.to_dict().items()
        if key not in _OPERATIONAL_HARNESS_KEYS
    }
    for sandbox in config.get("sandboxes") or ():
        sandbox.pop("instances", None)
        sandbox.pop("min_instances", None)
    return compute_task_hash(config)


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


# The package whose version decides what a provider kind generates. API providers
# have none: the model behind them is named by the model hash.
_ENGINE_PACKAGES = {
    "vllm": "vllm",
    "vllm_server": "vllm",
    "sglang": "sglang",
    "hf": "transformers",
}


# Recorded in place of a version that could not be determined.
UNKNOWN_VERSION = "unknown"
# The diff_hash of a working tree with no uncommitted changes to tracked files.
CLEAN_TREE = "clean"
_SERVER_PROBE_TIMEOUT_SECONDS = 60


def _server_package_version(package: str) -> str | None:
    """Version of ``package`` in the Python the vLLM server is launched with.

    The server can run in an environment other than the runner's (``VLLM_PYTHON``), so the
    runner's own metadata does not describe it.
    """
    import subprocess

    from olmo_eval.inference.providers.vllm_server_utils import _get_vllm_python

    python = _get_vllm_python()
    # Some images install vLLM without distribution metadata, so fall back to the module.
    code = (
        "import importlib, importlib.metadata as m\n"
        f"try:\n    print(m.version({package!r}))\n"
        "except m.PackageNotFoundError:\n"
        f"    print(importlib.import_module({package!r}).__version__)"
    )
    try:
        result = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=_SERVER_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning(f"Could not run {python} to read the {package} version: {e}")
        return None
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or not lines:
        logger.warning(
            f"Could not read the {package} version from {python}: {result.stderr[-200:]}"
        )
        return None
    return lines[-1].strip()


def engine_version(harness_config: HarnessConfig) -> str | None:
    """``package==version`` of the inference engine the provider runs, if it has one.

    A version that cannot be determined is recorded as ``package==unknown`` with a warning.
    """
    kind = str(harness_config.provider.kind)
    package = _ENGINE_PACKAGES.get(kind)
    if package is None:
        return None
    probe = _server_package_version if kind == "vllm_server" else _package_version
    found = probe(package)
    if found is None:
        logger.warning(f"Could not determine the {package} version the {kind} provider runs")
        found = UNKNOWN_VERSION
    return f"{package}=={found}"


# Resolved SGLang runtime keys that decide what the engine computes. The request-length caps
# come from the node's KV pool and truncate prompts and outputs, so they are compared;
# scheduling and memory settings change only speed.
_RUNTIME_IDENTITY_KEYS = (
    "profile",
    "packages",
    "logprobs",
    "arithmetic_environment",
    "max_total_num_tokens",
    "max_req_input_len",
)
_SCHEDULING_ENGINE_OPTIONS = frozenset(
    {"max_running_requests", "max_mamba_cache_size", "mem_fraction_static", "max_total_tokens"}
)


def runtime_identity(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a resolved runtime that identifies what the engine computes.

    ``mem_fraction_static`` and ``max_total_tokens`` matter only through the effective caps
    the engine reports, so they are left out when it reported them. A runtime that did not
    report them keeps ``max_total_tokens`` (the requested pool size) in the identity.
    """
    identity = {key: runtime[key] for key in _RUNTIME_IDENTITY_KEYS if key in runtime}
    reported = "max_total_num_tokens" in runtime and "max_req_input_len" in runtime
    skipped = _SCHEDULING_ENGINE_OPTIONS - ({"max_total_tokens"} if not reported else set())
    options = runtime.get("engine_options")
    if isinstance(options, Mapping):
        identity["engine_options"] = {
            key: value for key, value in options.items() if key not in skipped
        }
    return identity


def execution_info(harness_config: HarnessConfig) -> dict[str, Any]:
    """Where and how the run executes; a resumed run may differ here."""
    from olmo_eval.upload.provenance import beaker_info

    provider = harness_config.provider
    beaker = beaker_info() or {}
    return {
        "num_instances": provider.num_instances,
        "tensor_parallel_size": (provider.kwargs or {}).get("tensor_parallel_size"),
        "max_concurrency": provider.max_concurrency or harness_config.max_concurrency,
        "gpu_count": beaker.get("gpu_count"),
        "cluster": beaker.get("cluster"),
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict) and value:
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def _differences(recorded: Any, current: Any) -> list[str]:
    """The settings that differ between two nested dicts, as ``key: old -> new``."""
    old, new = _flatten(recorded or {}), _flatten(current or {})
    return [
        f"{key}: {old.get(key, '<unset>')!r} -> {new.get(key, '<unset>')!r}"
        for key in sorted(set(old) | set(new))
        if old.get(key, "<unset>") != new.get(key, "<unset>")
    ]


def _code_state(diff_hash: Any) -> str:
    return "no uncommitted changes" if diff_hash == CLEAN_TREE else f"changes {diff_hash}"


def check_header(path: Path, recorded: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Raise ProgressMismatchError naming the first measurement field that differs.

    Execution fields that differ are logged.
    """
    if recorded.get("version") != expected.get("version"):
        raise ProgressMismatchError(
            f"Progress record {path} has record format {recorded.get('version')!r}, and this "
            f"version of olmo-eval reads format {expected.get('version')!r}. Pass --fresh to "
            "discard recorded progress, or use another output directory."
        )
    for key in MEASUREMENT_FIELDS:
        old, new = recorded.get(key), expected.get(key)
        if key in ("git_commit", "diff_hash") and (old is None or new is None):
            continue
        if key == "engine_version" and UNKNOWN_VERSION in (
            str(old).rpartition("==")[2],
            str(new).rpartition("==")[2],
        ):
            # An unknown version cannot be compared, and refusing would block every resume
            # whenever the probe fails, so it warns instead of passing as equal.
            logger.warning(
                f"Resuming {path} without a comparable engine version "
                f"(recorded {old!r}, this run {new!r}); the engine may have changed"
            )
            continue
        if old == new:
            continue
        if key == "diff_hash":
            raise ProgressMismatchError(
                f"Progress record {path} was written with different uncommitted code changes "
                f"({_code_state(old)} then, {_code_state(new)} now): the code changed since "
                "the records were written, so restored instances would keep scores from the "
                "old code. Restore the same edits to resume, pass --fresh to rerun this "
                "run's tasks, or use another output directory (-O)."
            )
        changed = (
            _differences(recorded.get("task_config"), expected.get("task_config"))
            if key == "task_hash"
            else []
        )
        more = f", and {len(changed) - 5} more" if len(changed) > 5 else ""
        detail = f" ({'; '.join(changed[:5])}{more})" if changed else ""
        raise ProgressMismatchError(
            f"Progress record {path} was written by a different run: {key} differs "
            f"(recorded {old!r}, this run {new!r}){detail}. Pass --fresh to discard "
            "recorded progress, or use another output directory."
        )
    changed = _differences(recorded.get("execution"), expected.get("execution"))
    if changed:
        logger.warning(f"Resuming {path} on a different setup: {'; '.join(changed)}")


# ---------------------------------------------------------------------------
# Entries
# ---------------------------------------------------------------------------


def _hash_default(obj: object) -> Any:
    # Dataclasses (tool schemas) hash by content. Lazy image loaders and other objects
    # hash by type, not by their repr's address.
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return f"<{type(obj).__name__}>"


def _canonical(value: Any) -> Any:
    """``value`` as JSON-ready data that hashes the same in every process."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=_hash_default))
    except TypeError:  # keys that cannot be ordered
        return repr(value)


def request_hash(item: QueueItem) -> str:
    """Short hash of a prepared instance: its request, sampling params, gold and metadata.

    Everything a scorer reads is in it, so a dataset change that touches only a gold
    answer or a metadata field reruns the instance instead of restoring a stale score.
    Callables in the metadata (lazy loaders) are skipped.
    """
    request, instance = item.request, item.instance
    content = [
        instance.metadata.get("id"),
        request.request_type,
        request.prompt,
        request.messages,
        request.continuations,
        request.continuation_prompts,
        request.system_prompt,
        request.tools,
        request.max_length,
        request.metadata,
        asdict(item.sampling_params) if item.sampling_params is not None else None,
        instance.question,
        instance.gold_answer,
        instance.choices,
        instance.should_abstain,
        instance.tools,
        instance.expected_tool_calls,
        instance.required_trajectory,
        instance.initial_state,
        instance.expected_final_state,
        {k: v for k, v in instance.metadata.items() if not callable(v)},
    ]
    blob = json.dumps(_canonical(content), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def is_final(response: Response) -> bool:
    """Whether a scored response is the outcome an uninterrupted run would keep.

    A scoring failure flagged as infrastructure (no judge verdict, no sandbox) is
    not: a resumed run scores the instance again.
    """
    if response.scoring_error is not None:
        return response.scoring_error.get("infrastructure") != "true"
    return not any(
        error.get("infrastructure") == "true"
        for output in response.outputs
        for error in output_scoring_errors(output).values()
    )


def _metadata_changes(original: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Instance metadata that scoring set or changed. Callables (lazy loaders) are skipped."""
    changes: dict[str, Any] = {}
    for key, value in current.items():
        if callable(value):
            continue
        if key in original:
            try:
                if bool(original[key] == value):
                    continue
            except Exception:  # noqa: BLE001 - values without a truth value count as changed
                pass
        changes[key] = value
    return changes


def keeps_logprobs(task: Task) -> bool:
    """Whether a task's records keep the per-token logprobs of its generations.

    A metric whose scorer reads logprobs (``Scorer.reads_logprobs``) computes from them
    after a restore, so they are kept. Log-likelihood outputs always keep theirs.
    """
    config = task.config
    metrics = [*config.metrics, config.get_primary_metric()]
    for metric in metrics:
        scorer = getattr(metric, "scorer", None)
        if scorer is None:
            continue
        scorer_type = scorer if isinstance(scorer, type) else type(scorer)
        if getattr(scorer_type, "reads_logprobs", False):
            return True
    return False


def _drops_logprobs(output: LMOutput) -> bool:
    """Whether the predictions writer can do without the output's per-token logprobs.

    It reads ``sum_logits`` and ``num_tokens`` from the metadata when both are there,
    which every provider that returns logprobs sets, and the list's length otherwise.
    """
    metadata = output.metadata or {}
    return (
        isinstance(output.logprobs, list) and "sum_logits" in metadata and "num_tokens" in metadata
    )


def _encode_output(output: LMOutput, keep_logprobs: bool) -> dict[str, Any]:
    encoded: dict[str, Any] = {"text": output.text}
    if output.logprobs is not None:
        if keep_logprobs or not _drops_logprobs(output):
            encoded["logprobs"] = encode_value(output.logprobs)
        else:
            encoded["logprob_count"] = len(output.logprobs)
    if output.extracted_answer is not None:
        encoded["answer"] = encode_value(output.extracted_answer)
    if output.metadata:
        encoded["meta"] = encode_value(output.metadata)
    if output.tool_calls is not None:
        encoded["tool_calls"] = encode_value(output.tool_calls)
    return encoded


def _decode_output(encoded: Mapping[str, Any]) -> LMOutput:
    logprobs = decode_value(encoded.get("logprobs"))
    if "logprob_count" in encoded:
        logprobs = cast("list[LogProbEntry]", DroppedLogprobs(encoded["logprob_count"]))
    return LMOutput(
        text=encoded["text"],
        logprobs=logprobs,
        extracted_answer=decode_value(encoded.get("answer")),
        metadata=decode_value(encoded.get("meta") or {}),
        tool_calls=decode_value(encoded.get("tool_calls")),
    )


def _usage(result: ResultItem) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    if result.sent_at is not None:
        usage["sent_at"] = result.sent_at
    if result.attributed_seconds is not None:
        usage["seconds"] = result.attributed_seconds
    return usage


def encode_response(
    item: QueueItem,
    response: Response,
    result: ResultItem,
    request_hash: str,
    keep_logprobs: bool,
) -> dict[str, Any]:
    """A scored instance's entry: what scoring produced, without the prompt.

    ``request_hash`` is the instance's hash from before it was scored. Without ``keep_logprobs``
    a generation's per-token logprobs are left out where the predictions writer does not
    need them; a log-likelihood request's are always kept.
    """
    from olmo_eval.common.types import RequestType

    keep = keep_logprobs or item.request.request_type == RequestType.LOGLIKELIHOOD
    entry: dict[str, Any] = {"idx": item.instance_idx, "request_hash": request_hash}
    entry["outputs"] = [_encode_output(output, keep) for output in response.outputs]
    if response.scores:
        entry["scores"] = encode_value(response.scores)
    if response.request_trace is not None:
        entry["trace"] = encode_value(response.request_trace)
    if response.scoring_error is not None:
        entry["scoring_error"] = response.scoring_error
    changes = _metadata_changes(item.instance.metadata, response.instance.metadata)
    if changes:
        entry["meta"] = encode_value(changes)
    removed = [key for key in item.instance.metadata if key not in response.instance.metadata]
    if removed:
        entry["meta_removed"] = removed
    entry.update(_usage(result))
    return entry


def encode_failure(item: QueueItem, result: ResultItem, request_hash: str) -> dict[str, Any]:
    """A generation failure's entry."""
    entry = {
        "idx": item.instance_idx,
        "request_hash": request_hash,
        "failed": result.error or "",
    }
    entry.update(_usage(result))
    return entry


def decode_response(
    entry: Mapping[str, Any],
    item: QueueItem,
    harness_config: HarnessConfig,
    default_trace: Any = None,
) -> Response:
    """Rebuild the scored Response for ``entry`` from its prepared queue item.

    An entry without a request trace has the one the last ``trace`` entry set.
    """
    from olmo_eval.common.types.trajectory import AgentTrajectory
    from olmo_eval.harness.harness import apply_harness_config

    instance = item.instance
    changes = decode_value(entry.get("meta") or {})
    removed = entry.get("meta_removed") or []
    if changes or removed:
        metadata = {k: v for k, v in instance.metadata.items() if k not in removed}
        metadata.update(changes)
        instance = replace(instance, metadata=metadata)
    outputs = [_decode_output(output) for output in entry["outputs"]]
    # As the live path does, from the first output's metadata.
    trajectory = None
    traj_dict = outputs[0].metadata.get("trajectory") if outputs else None
    if traj_dict:
        trajectory = AgentTrajectory.from_dict(traj_dict)
    return Response(
        instance=instance,
        request=apply_harness_config(harness_config, item.request),
        outputs=outputs,
        scores=decode_value(entry.get("scores") or {}),
        trajectory=trajectory,
        request_trace=decode_value(entry.get("trace", default_trace)),
        scoring_error=entry.get("scoring_error"),
    )


# ---------------------------------------------------------------------------
# Finalized tasks
# ---------------------------------------------------------------------------

_UNSTORED_RESULT_FIELDS = ("predictions", "requests")


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Run-level progress
# ---------------------------------------------------------------------------


def parse_exit_thresholds(value: str | None) -> list[float]:
    """Fractions from OLMO_EVAL_DEBUG_EXIT_AT, such as ``0.3,0.7``."""
    if not value:
        return []
    try:
        thresholds = sorted(float(part) for part in value.split(",") if part.strip())
    except ValueError as e:
        raise ValueError(f"{DEBUG_EXIT_AT_ENV}={value!r} must be comma-separated fractions") from e
    if any(not 0.0 <= t <= 1.0 for t in thresholds):
        raise ValueError(f"{DEBUG_EXIT_AT_ENV}={value!r} must be fractions between 0 and 1")
    return thresholds


@dataclass
class Restored:
    """What a resumed run takes over from the records instead of running again."""

    responses: dict[str, dict[int, Response]] = field(default_factory=dict)
    finalized: dict[str, TaskResult] = field(default_factory=dict)
    remaining: list[QueueItem] = field(default_factory=list)


@dataclass
class _TaskProgress:
    key: str
    header: dict[str, Any]
    items: dict[int, QueueItem]
    # Hashes of the prepared instances, taken before any scoring changes them.
    hashes: dict[int, str] = field(default_factory=dict)
    # The request trace entries leave out; most tasks send every request the same way.
    trace: Any = None
    # Whether entries keep their generations' per-token logprobs (keeps_logprobs).
    keep_logprobs: bool = True


@dataclass
class RunProgress:
    """The progress records of one run: restores finished work and records new work.

    ``exit_thresholds`` (from ``OLMO_EVAL_DEBUG_EXIT_AT``) make the run send itself
    SIGTERM, after writing out its records, the first time the finished fraction of
    instances crosses one that was not crossed when the run started. With the CLI's
    handler that ends the run through the same path as a preemption, with exit code
    143.
    """

    output_dir: str
    model_name: str
    model_hash: str | None
    model_revision: str | None
    harness_hash: str | None
    git_commit: str | None
    engine_version: str | None = None
    # Hash of the uncommitted changes to tracked files (CLEAN_TREE when there are none);
    # None when git cannot say, which never refuses a resume.
    diff_hash: str | None = None
    execution: dict[str, Any] = field(default_factory=dict)
    exit_thresholds: list[float] = field(default_factory=list)
    _store: ProgressStore | None = field(default=None, init=False)
    _tasks: dict[str, _TaskProgress] = field(default_factory=dict, init=False)
    # Recorded runtime identity of each task of this run that has one, by task key.
    _runtimes: dict[str, Any] = field(default_factory=dict, init=False)
    _total: int = field(default=0, init=False)
    _done: int = field(default=0, init=False)
    _armed: list[float] = field(default_factory=list, init=False)
    _final_paths: list[Path] = field(default_factory=list, init=False)
    _run_keys: set[str] = field(default_factory=set, init=False)
    _lock: ProgressLock | None = field(default=None, init=False)

    def header(self, spec: str, task_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "spec": spec,
            "task_hash": compute_task_hash(task_config),
            "model_hash": self.model_hash,
            "model_revision": self.model_revision,
            "harness_hash": self.harness_hash,
            "engine_version": self.engine_version,
            "git_commit": self.git_commit,
            "diff_hash": self.diff_hash,
            "execution": self.execution,
            "task_config": task_config,
        }

    def _load_final(self, spec: str, header: Mapping[str, Any], path: Path) -> TaskResult | None:
        """The task's finalized result, if it was finalized and its output files are on disk."""
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            logger.warning(f"Ignoring unreadable finalized result {path}")
            return None
        check_header(path, data["header"], header)
        missing = [p for p in data.get("outputs", []) if not (Path(self.output_dir) / p).is_file()]
        if missing:
            logger.info(f"{spec} was finalized, but {missing[0]} is missing; recomputing it")
            return None
        return TaskResult(**decode_value(data["result"]))

    def restore(
        self,
        trackers: Mapping[str, TaskTracker],
        items: Sequence[QueueItem],
        harness_config: HarnessConfig,
    ) -> Restored:
        """Load every task's records; return what is done and the items left to run.

        Writes a header for every task not yet finalized that has none, and folds
        restored instances' failures and token usage into the trackers.
        """
        if self._lock is None:
            self._lock = model_lock(self.output_dir, self.model_name)
        if not self._lock.held:
            self._lock.acquire()
        try:
            return self._restore(trackers, items, harness_config)
        except BaseException:
            self.close()
            raise

    def _restore(
        self,
        trackers: Mapping[str, TaskTracker],
        items: Sequence[QueueItem],
        harness_config: HarnessConfig,
    ) -> Restored:
        from olmo_eval.runners.asynq.types import ResultItem

        by_task: dict[str, dict[int, QueueItem]] = {}
        for item in items:
            by_task.setdefault(item.task_id, {})[item.instance_idx] = item

        self._store = store = ProgressStore(stream_dir(self.output_dir, self.model_name))
        restored = Restored()
        expected: dict[str, dict[str, Any]] = {}
        active: dict[str, tuple[str, _TaskProgress]] = {}
        for spec, tracker in trackers.items():
            if tracker.task is None or tracker.error:
                continue
            header = self.header(spec, tracker.task.config.to_dict())
            expected[spec] = header
            self._run_keys.add(task_key(spec, header["task_hash"]))
            path = final_path(self.output_dir, self.model_name, spec, header["task_hash"])
            self._final_paths.append(path)
            final = self._load_final(spec, header, path)
            if final is not None:
                restored.finalized[spec] = final
                continue
            task = _TaskProgress(
                key=task_key(spec, header["task_hash"]),
                header=header,
                items=by_task.get(spec, {}),
                keep_logprobs=keeps_logprobs(tracker.task),
            )
            task.hashes = {idx: request_hash(item) for idx, item in task.items.items()}
            active[task.key] = (spec, task)

        # Entries are decoded as read, so raw entries are not held; later entries for
        # an instance replace earlier ones.
        outcomes: dict[str, dict[int, tuple[Response | str, float | None, float | None]]] = {
            spec: {} for spec, _ in active.values()
        }
        has_header: set[str | None] = set()
        mismatched = 0
        foreign: set[str] = set()
        for entry in store.read():
            key = entry.get("t")
            if "runtime" in entry:
                # Only the runtime that recorded a task this run continues is compared.
                if key in active:
                    self._runtimes.setdefault(key, entry["runtime"])
                continue
            if "header" in entry:
                # Any header for a task of this run, under any hash, must agree with it.
                recorded = entry["header"]
                spec = recorded.get("spec")
                if spec in expected:
                    check_header(store.path, recorded, expected[spec])
                    if key in has_header and key in active:
                        # A crash while rewriting the stream left a prefix of the old
                        # segments before the new one, which restates the task.
                        active[key][1].trace = None
                    has_header.add(key)
                elif isinstance(spec, str):
                    foreign.add(spec)
                continue
            if key not in active:
                continue
            spec, task = active[key]
            if "default_trace" in entry:
                task.trace = entry["default_trace"]
                continue
            item = task.items.get(entry["idx"])
            if item is None or entry.get("request_hash") != task.hashes[entry["idx"]]:
                mismatched += 1
                continue
            outcome = (
                entry["failed"]
                if "failed" in entry
                else decode_response(entry, item, harness_config, task.trace)
            )
            outcomes[spec][entry["idx"]] = (outcome, entry.get("sent_at"), entry.get("seconds"))

        for key, (spec, task) in active.items():
            if key not in has_header:
                store.append({"t": key, "header": task.header})
            tracker = trackers[spec]
            responses: dict[int, Response] = {}
            for idx, (outcome, sent_at, seconds) in sorted(outcomes[spec].items()):
                item = task.items[idx]
                usage = ResultItem(
                    model_name=item.model_name,
                    task_id=spec,
                    instance_idx=idx,
                    instance=item.instance,
                    request=item.request,
                    outputs=[],
                    sent_at=sent_at,
                    attributed_seconds=seconds,
                )
                if isinstance(outcome, str):
                    tracker.add_failure(idx, outcome)
                else:
                    responses[idx] = outcome
                    usage.request, usage.outputs = outcome.request, outcome.outputs
                tracker.record_usage(usage)
            self._tasks[spec] = task
            if responses:
                restored.responses[spec] = responses
            self._done += len(outcomes[spec])

        for path in _model_dir(self.output_dir, self.model_name).glob(f"*{FINAL_SUFFIX}"):
            if path not in self._final_paths and (spec := _final_spec(path)) is not None:
                foreign.add(spec)
        if foreign:
            logger.info(f"keeping progress for {', '.join(sorted(foreign))} from another run")
        if mismatched:
            logger.warning(
                f"{mismatched} recorded instance(s) no longer match their prepared instance "
                "(the task's data or prompt changed); running them again"
            )
        self._total = sum(t.total_instances for t in trackers.values() if not t.error)
        self._done += sum(trackers[spec].total_instances for spec in restored.finalized)
        start = self._done / self._total if self._total else 1.0
        self._armed = [t for t in self.exit_thresholds if t > start]
        restored.remaining = [
            item
            for item in items
            if item.task_id not in restored.finalized
            and item.instance_idx not in restored.responses.get(item.task_id, {})
            and item.instance_idx not in trackers[item.task_id].failed_instances
        ]
        if self._done:
            logger.info(
                f"resuming: {self._done} of {self._total} instances already done across "
                f"{len(restored.responses) + len(restored.finalized)} tasks "
                f"({len(restored.finalized)} finalized)"
            )
        return restored

    def check_runtime(self, runtime: Mapping[str, Any]) -> None:
        """Compare the workers' resolved runtime identity with the one recorded, or record it.

        Each task has its own record, so a runtime recorded for another run's tasks is
        not compared. Only :func:`runtime_identity` is compared. A requeue lands on
        another node, where scheduling settings differ; the request-length caps the
        engine derived do not change without changing the outputs, so they are compared.
        """
        if not runtime or self._store is None:
            return
        current = encode_value(runtime_identity(runtime))
        for recorded in self._runtimes.values():
            if recorded != current:
                changed = "; ".join(_differences(recorded, current)[:5])
                raise ProgressMismatchError(
                    f"Progress record {self._store.path} was written with another resolved "
                    f"model runtime ({changed}). Pass --fresh to discard recorded progress."
                )
        for task in self._tasks.values():
            if task.key not in self._runtimes:
                self._runtimes[task.key] = current
                self._store.append({"t": task.key, "runtime": current})

    def _append(
        self, spec: str, idx: int, build: Callable[[QueueItem, str], dict[str, Any]]
    ) -> None:
        if shutdown_requested() or self._store is None:
            return
        task = self._tasks.get(spec)
        if task is None:
            return
        try:
            entry = build(task.items[idx], task.hashes[idx])
            if "outputs" in entry:
                trace = entry.pop("trace", None)
                if trace != task.trace:
                    self._store.append({"t": task.key, "default_trace": trace})
                    task.trace = trace
            self._store.append({"t": task.key, **entry})
        except Exception as e:  # noqa: BLE001 - losing the record must not stop the run
            logger.error(
                f"Could not record {spec}[{idx}] in {self._store.path}; recording stops for "
                f"this task, and a resumed run repeats its unrecorded instances: {e}"
            )
            del self._tasks[spec]
            return
        self._done += 1
        crossed = [t for t in self._armed if self._done >= t * self._total]
        if crossed:
            self._armed = []
            self.close()
            logger.warning(
                f"{DEBUG_EXIT_AT_ENV}: sending SIGTERM after {self._done} of {self._total} "
                f"instances (crossed {crossed[-1]:g})"
            )
            os.kill(os.getpid(), signal.SIGTERM)

    def record(self, spec: str, idx: int, response: Response, result: ResultItem) -> None:
        """Record a scored instance whose outcome is final."""
        if not is_final(response):
            return
        task = self._tasks.get(spec)
        if task is None:
            return
        self._append(
            spec,
            idx,
            lambda item, request_hash: encode_response(
                item, response, result, request_hash, task.keep_logprobs
            ),
        )

    def record_failure(self, spec: str, idx: int, result: ResultItem) -> None:
        """Record a generation failure that the request caused.

        A scaffold failure that still returns a fallback output is not a generation failure.
        It is scored as a soft failure and recorded as final through :meth:`record`, as an
        uninterrupted run keeps it.
        """
        if result.request_caused:
            self._append(
                spec, idx, lambda item, request_hash: encode_failure(item, result, request_hash)
            )

    def finalize(self, spec: str, result: TaskResult, *, wrote: Sequence[str] = ()) -> None:
        """Keep a finalized task's result, so a resumed run does not compute it again.

        ``wrote`` names the output files ("predictions", "requests") written for it. A
        task that failed (an error, or an exceeded failure budget) is not kept: a retry
        restores its instances and computes the task again.
        """
        task = self._tasks.get(spec)
        if task is None or shutdown_requested() or result.error or result.failure_budget_exceeded:
            return
        task_hash = task.header["task_hash"]
        paths = {
            kind: output_path(self.output_dir, kind, spec, self.model_name, task_hash)
            for kind in wrote
        }
        stored = {
            f.name: getattr(result, f.name)
            for f in fields(result)
            if f.name not in _UNSTORED_RESULT_FIELDS
        }
        try:
            _write_json_atomic(
                final_path(self.output_dir, self.model_name, spec, task_hash),
                {
                    "header": task.header,
                    "outputs": [str(paths[kind].relative_to(self.output_dir)) for kind in wrote],
                    "result": encode_value(stored),
                },
            )
        except Exception as e:  # noqa: BLE001 - losing the record must not stop the run
            logger.error(f"Could not record the finalized result of {spec}: {e}")

    def discard(self) -> None:
        """Delete this run's records: its tasks' entries and finalized results.

        Records of other tasks, in the model's stream or beside it, and other models'
        progress in the same output directory are left alone, and directories this
        empties are removed. A failure leaves the records as they were (see
        :meth:`ProgressStore.drop`), so a rerun resumes from them.
        """
        self._flush()
        if self._store is not None:
            try:
                self._store.drop(keys=self._run_keys)
                for path in self._final_paths:
                    path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"Could not delete the progress records in {self._store.path}: {e}")
                self.close()
                return
        self.close()
        _remove_empty_dirs(
            _model_dir(self.output_dir, self.model_name), Path(self.output_dir) / PROGRESS_DIR
        )

    def _flush(self) -> None:
        if self._store is None:
            return
        try:
            self._store.close()
        except OSError as e:
            logger.warning(f"Could not write progress record {self._store.path}: {e}")

    def close(self) -> None:
        """Write out and fsync every record, and release the model's lock."""
        self._flush()
        if self._lock is not None:
            self._lock.release()


def open_run_progress(
    output_dir: str,
    model_name: str,
    harness_config: HarnessConfig,
    lock: ProgressLock | None = None,
) -> RunProgress:
    """RunProgress for this run, with its identity and debug settings from the environment.

    ``lock`` is the caller's lock on the model's progress, if it already holds one.
    """
    from olmo_eval.common.types import compute_model_hash
    from olmo_eval.upload.provenance import git_info, uncommitted_diff_hash

    git = git_info()
    dirty = git.get("dirty")
    diff_hash = None if dirty is None else uncommitted_diff_hash() if dirty else CLEAN_TREE
    if diff_hash not in (None, CLEAN_TREE):
        logger.warning(
            f"Recording progress from a working tree with uncommitted changes ({diff_hash}); "
            "resuming this run will require the same uncommitted changes"
        )

    progress = RunProgress(
        output_dir=output_dir,
        model_name=model_name,
        model_hash=compute_model_hash(harness_config.provider.to_dict()),
        model_revision=harness_config.provider.revision,
        harness_hash=harness_hash(harness_config),
        git_commit=git.get("commit"),
        engine_version=engine_version(harness_config),
        diff_hash=diff_hash,
        execution=execution_info(harness_config),
        exit_thresholds=parse_exit_thresholds(os.environ.get(DEBUG_EXIT_AT_ENV)),
    )
    progress._lock = lock
    return progress


__all__ = [
    "DEBUG_EXIT_AT_ENV",
    "DroppedLogprobs",
    "DroppedLogprobsError",
    "ProgressLock",
    "ProgressLockedError",
    "ProgressMismatchError",
    "ProgressStore",
    "Restored",
    "RunProgress",
    "discard_progress",
    "has_progress",
    "model_lock",
    "open_run_progress",
    "remove_empty_progress_dirs",
]

"""Progress records that let a restarted run skip the work it already finished.

Each task gets a record under ``<output_dir>/progress/<model>/<spec>_<hash6>``. The
first entry is a header naming what the record measures (task, model, harness and
code) and where it ran; every later entry is one finished instance, written by the
runner's main process once the instance's outcome is final. When a task is
finalized, its result is also written to ``<spec>_<hash6>.final.json``. A run
started with ``--resume`` on that directory keeps the finalized tasks, rebuilds the
recorded instances of the others, and queues only the rest, so its predictions and
metrics are the same as an uninterrupted run's. A run that completes deletes its
records.

An entry keeps what scoring produced (outputs, scores, the provider's request trace
and instance metadata that scoring changed), not the prompt: the instance and request
are rebuilt by preparing the task again, and a short hash of the prepared request
checks that they are the same.

An instance's outcome is final, and recorded, when the uninterrupted run would keep
it too: a scored response (including one some scorer could not score, which the
metrics leave out), or a generation failure the request itself caused. Not recorded,
so a resumed run tries them again: a generation failure from the call failing
(``ResultItem.retryable``), a scoring failure flagged as infrastructure (the judge or
sandbox was unavailable), and anything that arrives after shutdown was requested.

Two storage layouts sit behind :class:`ProgressStore`, chosen with
``OLMO_EVAL_PROGRESS_LAYOUT``: ``segments`` (default) writes batches of entries to new
files with an atomic rename, so a results uploader that re-sends changed files sends
each segment once; ``append`` appends one JSON line per entry to a single file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pickle
import shutil
import signal
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.scoring_failures import output_scoring_errors
from olmo_eval.common.shutdown import shutdown_requested
from olmo_eval.common.types import LMOutput, Response
from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX, TaskResult
from olmo_eval.runners.processing.utils import compute_task_hash, sanitize_spec_for_filename

if TYPE_CHECKING:
    from olmo_eval.harness.config import HarnessConfig
    from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker

logger = get_logger(__name__)

PROGRESS_DIR = "progress"
PROGRESS_LAYOUT_ENV = "OLMO_EVAL_PROGRESS_LAYOUT"
DEBUG_EXIT_AT_ENV = "OLMO_EVAL_DEBUG_EXIT_AT"
RECORD_VERSION = 1
FINAL_SUFFIX = ".final.json"

# Append layout: seconds between fsyncs. Segments layout: a segment is written when
# its buffered entries reach this size or this age, and on close.
FSYNC_INTERVAL_SECONDS = 5.0
SEGMENT_MAX_BYTES = 8 * 1024 * 1024
SEGMENT_MAX_SECONDS = 20.0

# Header fields that decide what an instance's record means; a record that differs
# in one cannot be resumed. The others describe where the run executed, and only warn.
MEASUREMENT_FIELDS = (
    "version",
    "spec",
    "task_hash",
    "model_hash",
    "model_revision",
    "harness_hash",
    "git_commit",
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
# Storage layouts
# ---------------------------------------------------------------------------


def _read_lines(path: Path, on_end: Callable[[int, int], None] | None = None) -> Iterator[Any]:
    """Yield the entries of a JSONL file line by line.

    A last line without a newline was cut off mid-write; it is dropped and logged.
    ``on_end`` gets the byte length of the complete lines and of the file.
    """
    good = 0
    with open(path, "rb") as f:
        for number, line in enumerate(f, start=1):
            if not line.endswith(b"\n"):
                logger.warning(f"Dropping a partly written last line from {path}")
                break
            good += len(line)
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except ValueError as e:
                raise ProgressMismatchError(
                    f"Progress record {path} line {number} is not valid JSON ({e}). "
                    "Pass --fresh to discard recorded progress."
                ) from e
        size = f.seek(0, os.SEEK_END)
    if on_end is not None:
        on_end(good, size)


def _dump(entry: Mapping[str, Any]) -> str:
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"


class ProgressStore(ABC):
    """Storage for one task's progress record: a header entry, then one entry per instance."""

    suffix: str

    def __init__(self, base: Path) -> None:
        self.path = base.with_name(base.name + self.suffix)

    @abstractmethod
    def read(self) -> Iterator[dict[str, Any]]:
        """Yield every complete entry in order, header first; nothing when there is no record.

        Reads one line at a time. Drops (and logs) an entry that was cut off while
        being written. Appends after a full read continue the record.
        """

    @abstractmethod
    def reset(self, header: Mapping[str, Any]) -> None:
        """Discard any existing entries and start the record with ``header``."""

    @abstractmethod
    def append(self, entry: Mapping[str, Any]) -> None:
        """Add one entry after those already read or written."""

    @abstractmethod
    def close(self) -> None:
        """Write out and fsync everything appended so far."""


class AppendProgressStore(ProgressStore):
    """One JSONL file, flushed after every line and fsynced every few seconds."""

    suffix = ".jsonl"

    def __init__(self, base: Path) -> None:
        super().__init__(base)
        self._file: Any = None
        self._last_sync = 0.0

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_file():
            return

        def truncate_torn_tail(good: int, size: int) -> None:
            if good < size:
                with open(self.path, "r+b") as f:
                    f.truncate(good)

        yield from _read_lines(self.path, truncate_torn_tail)

    def reset(self, header: Mapping[str, Any]) -> None:
        self.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", encoding="utf-8")  # noqa: SIM115 - closed in close()
        self.append(header)
        self._sync()

    def append(self, entry: Mapping[str, Any]) -> None:
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        self._file.write(_dump(entry))
        self._file.flush()
        if time.monotonic() - self._last_sync >= FSYNC_INTERVAL_SECONDS:
            self._sync()

    def _sync(self) -> None:
        os.fsync(self._file.fileno())
        self._last_sync = time.monotonic()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._sync()
        self._file.close()
        self._file = None


class SegmentProgressStore(ProgressStore):
    """A directory of numbered JSONL segments, each written whole and renamed into place.

    Entries are buffered and written as a new segment once they reach
    ``SEGMENT_MAX_BYTES``, when ``SEGMENT_MAX_SECONDS`` have passed at the next
    append, and on close.
    """

    suffix = ".segments"

    def __init__(self, base: Path) -> None:
        super().__init__(base)
        self._buffer: list[str] = []
        self._buffered_bytes = 0
        self._next_segment = 0
        self._last_write = time.monotonic()

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.is_dir():
            return
        for leftover in self.path.glob("*.tmp"):
            leftover.unlink()
        segments = sorted(self.path.glob("[0-9]*.jsonl"))
        self._next_segment = int(segments[-1].stem) + 1 if segments else 0
        for segment in segments:
            yield from _read_lines(segment)

    def reset(self, header: Mapping[str, Any]) -> None:
        self._buffer, self._buffered_bytes = [], 0
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True)
        self._next_segment = 0
        self.append(header)
        self._write_segment()

    def append(self, entry: Mapping[str, Any]) -> None:
        line = _dump(entry)
        self._buffer.append(line)
        self._buffered_bytes += len(line)
        if (
            self._buffered_bytes >= SEGMENT_MAX_BYTES
            or time.monotonic() - self._last_write >= SEGMENT_MAX_SECONDS
        ):
            self._write_segment()

    def _write_segment(self) -> None:
        if not self._buffer:
            return
        final = self.path / f"{self._next_segment:06d}.jsonl"
        tmp = final.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(self._buffer)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        self._next_segment += 1
        self._buffer, self._buffered_bytes = [], 0
        self._last_write = time.monotonic()

    def close(self) -> None:
        self._write_segment()


_LAYOUTS: dict[str, type[ProgressStore]] = {
    "segments": SegmentProgressStore,
    "append": AppendProgressStore,
}


def store_class(layout: str | None = None) -> type[ProgressStore]:
    """The store for ``layout``, by default the one ``OLMO_EVAL_PROGRESS_LAYOUT`` names."""
    name = (layout or os.environ.get(PROGRESS_LAYOUT_ENV) or "segments").strip().lower()
    if name not in _LAYOUTS:
        raise ValueError(
            f"{PROGRESS_LAYOUT_ENV}={name!r} is not a progress layout; "
            f"use one of {', '.join(sorted(_LAYOUTS))}"
        )
    return _LAYOUTS[name]


def progress_base(output_dir: str | Path, model_name: str, spec: str, task_hash: str) -> Path:
    """Path of a task's record without the layout's suffix, beside its predictions' name."""
    model_dir = Path(output_dir) / PROGRESS_DIR / sanitize_spec_for_filename(model_name)
    return model_dir / f"{sanitize_spec_for_filename(spec)}_{task_hash[-6:]}"


def has_progress(output_dir: str | Path) -> bool:
    """Whether the output directory holds any progress record."""
    path = Path(output_dir) / PROGRESS_DIR
    return path.is_dir() and any(path.iterdir())


def discard_progress(output_dir: str | Path) -> None:
    """Delete every progress record in the output directory."""
    path = Path(output_dir) / PROGRESS_DIR
    if path.exists():
        shutil.rmtree(path)


def _output_paths(output_dir: str, model_name: str, spec: str, task_hash: str) -> dict[str, Path]:
    """The predictions and requests files the runner writes for a task."""
    name = f"{sanitize_spec_for_filename(spec)}_{task_hash[-6:]}"
    model = sanitize_spec_for_filename(model_name)
    return {
        "predictions": Path(output_dir) / "predictions" / model / f"{name}{PREDICTIONS_SUFFIX}",
        "requests": Path(output_dir) / "requests" / model / f"{name}{REQUESTS_SUFFIX}",
    }


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
        "vllm_version": _package_version("vllm"),
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


def check_header(path: Path, recorded: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Raise ProgressMismatchError naming the first measurement field that differs.

    Execution fields that differ are logged.
    """
    for key in MEASUREMENT_FIELDS:
        old, new = recorded.get(key), expected.get(key)
        if key == "git_commit" and (old is None or new is None):
            continue
        if old == new:
            continue
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


def _hash_default(obj: object) -> str:
    # Lazy image loaders and other objects hash by type, not by their repr's address.
    return f"<{type(obj).__name__}>"


def request_hash(item: QueueItem) -> str:
    """Short hash of a prepared instance's request, sampling params and id."""
    request = item.request
    content = [
        item.instance.metadata.get("id"),
        request.request_type,
        request.prompt,
        request.messages,
        request.continuations,
        request.continuation_prompts,
        request.system_prompt,
        asdict(item.sampling_params) if item.sampling_params is not None else None,
    ]
    blob = json.dumps(content, sort_keys=True, default=_hash_default)
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


def _encode_output(output: LMOutput) -> dict[str, Any]:
    encoded: dict[str, Any] = {"text": output.text}
    if output.logprobs is not None:
        encoded["logprobs"] = encode_value(output.logprobs)
    if output.extracted_answer is not None:
        encoded["answer"] = encode_value(output.extracted_answer)
    if output.metadata:
        encoded["meta"] = encode_value(output.metadata)
    if output.tool_calls is not None:
        encoded["tool_calls"] = encode_value(output.tool_calls)
    return encoded


def _decode_output(encoded: Mapping[str, Any]) -> LMOutput:
    return LMOutput(
        text=encoded["text"],
        logprobs=decode_value(encoded.get("logprobs")),
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


def encode_response(item: QueueItem, response: Response, result: ResultItem) -> dict[str, Any]:
    """A scored instance's entry: what scoring produced, without the prompt."""
    entry: dict[str, Any] = {"idx": item.instance_idx, "rh": request_hash(item)}
    entry["outputs"] = [_encode_output(output) for output in response.outputs]
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


def encode_failure(item: QueueItem, result: ResultItem) -> dict[str, Any]:
    """A generation failure's entry."""
    entry = {"idx": item.instance_idx, "rh": request_hash(item), "failed": result.error or ""}
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
    store: ProgressStore
    header: dict[str, Any]
    items: dict[int, QueueItem]
    runtime: Any = None
    # The request trace entries leave out; most tasks send every request the same way.
    trace: Any = None


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
    execution: dict[str, Any] = field(default_factory=dict)
    exit_thresholds: list[float] = field(default_factory=list)
    _tasks: dict[str, _TaskProgress] = field(default_factory=dict, init=False)
    _total: int = field(default=0, init=False)
    _done: int = field(default=0, init=False)
    _armed: list[float] = field(default_factory=list, init=False)

    def header(self, spec: str, task_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "spec": spec,
            "task_hash": compute_task_hash(task_config),
            "model_hash": self.model_hash,
            "model_revision": self.model_revision,
            "harness_hash": self.harness_hash,
            "git_commit": self.git_commit,
            "execution": self.execution,
            "task_config": task_config,
        }

    def _find_store(self, spec: str, header: Mapping[str, Any]) -> ProgressStore:
        """This task's store; fail if the directory holds the task's record under another hash."""
        cls = store_class()
        base = progress_base(self.output_dir, self.model_name, spec, header["task_hash"])
        store = cls(base)
        if store.path.exists():
            return store
        pattern = f"{sanitize_spec_for_filename(spec)}_{'?' * 6}{cls.suffix}"
        for other_path in sorted(base.parent.glob(pattern)):
            other = cls(other_path.with_name(other_path.name[: -len(cls.suffix)]))
            first = next(iter(other.read()), None)
            if first is not None and first.get("spec") == spec:
                check_header(other.path, first, header)
        return store

    def _load_final(self, spec: str, header: Mapping[str, Any], base: Path) -> TaskResult | None:
        """The task's finalized result, if it was finalized and its output files are on disk."""
        path = base.with_name(base.name + FINAL_SUFFIX)
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
        """Load every task's record; return what is done and the items left to run.

        Opens a record for every task not yet finalized, writing its header if it has
        none, and folds restored instances' failures and token usage into the trackers.
        """
        from olmo_eval.runners.asynq.types import ResultItem

        by_task: dict[str, dict[int, QueueItem]] = {}
        for item in items:
            by_task.setdefault(item.task_id, {})[item.instance_idx] = item

        restored = Restored()
        mismatched = 0
        for spec, tracker in trackers.items():
            if tracker.task is None or tracker.error:
                continue
            task_items = by_task.get(spec, {})
            header = self.header(spec, tracker.task.config.to_dict())
            base = progress_base(self.output_dir, self.model_name, spec, header["task_hash"])
            final = self._load_final(spec, header, base)
            if final is not None:
                restored.finalized[spec] = final
                continue

            store = self._find_store(spec, header)
            task = _TaskProgress(store=store, header=header, items=task_items)
            entries = store.read()
            first = next(entries, None)
            if first is None:
                store.reset(header)
            else:
                check_header(store.path, first, header)
            # Decoded as read, so raw entries are not held; later entries for an
            # instance replace earlier ones.
            outcomes: dict[int, tuple[Response | str, float | None, float | None]] = {}
            for entry in entries:
                if "runtime" in entry:
                    if task.runtime is None:
                        task.runtime = entry["runtime"]
                    continue
                if "default_trace" in entry:
                    task.trace = entry["default_trace"]
                    continue
                item = task_items.get(entry["idx"])
                if item is None or entry.get("rh") != request_hash(item):
                    mismatched += 1
                    continue
                outcome = (
                    entry["failed"]
                    if "failed" in entry
                    else decode_response(entry, item, harness_config, task.trace)
                )
                outcomes[entry["idx"]] = (outcome, entry.get("sent_at"), entry.get("seconds"))
            responses: dict[int, Response] = {}
            for idx, (outcome, sent_at, seconds) in sorted(outcomes.items()):
                item = task_items[idx]
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
            self._done += len(outcomes)

        if mismatched:
            logger.warning(
                f"{mismatched} recorded instance(s) no longer match their prepared request "
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
        """Compare the workers' resolved runtime identity with the one recorded, or record it."""
        if not runtime:
            return
        current = encode_value(dict(runtime))
        for task in self._tasks.values():
            if task.runtime is None:
                task.runtime = current
                task.store.append({"runtime": current})
            elif task.runtime != current:
                changed = "; ".join(_differences(task.runtime, current)[:5])
                raise ProgressMismatchError(
                    f"Progress record {task.store.path} was written with another resolved "
                    f"model runtime ({changed}). Pass --fresh to discard recorded progress."
                )

    def _append(self, spec: str, idx: int, build: Callable[[QueueItem], dict[str, Any]]) -> None:
        if shutdown_requested():
            return
        task = self._tasks.get(spec)
        if task is None:
            return
        try:
            entry = build(task.items[idx])
            if "outputs" in entry:
                trace = entry.pop("trace", None)
                if trace != task.trace:
                    task.store.append({"default_trace": trace})
                    task.trace = trace
            task.store.append(entry)
        except Exception as e:  # noqa: BLE001 - losing the record must not stop the run
            logger.error(
                f"Could not record {spec}[{idx}] in {task.store.path}; recording stops for "
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
        if is_final(response):
            self._append(spec, idx, lambda item: encode_response(item, response, result))

    def record_failure(self, spec: str, idx: int, result: ResultItem) -> None:
        """Record a generation failure that the request caused."""
        if not result.retryable:
            self._append(spec, idx, lambda item: encode_failure(item, result))

    def finalize(self, spec: str, result: TaskResult, *, wrote: Sequence[str] = ()) -> None:
        """Keep a finalized task's result, so a resumed run does not compute it again.

        ``wrote`` names the output files ("predictions", "requests") written for it.
        """
        task = self._tasks.get(spec)
        if task is None or shutdown_requested():
            return
        task_hash = task.header["task_hash"]
        paths = _output_paths(self.output_dir, self.model_name, spec, task_hash)
        stored = {
            f.name: getattr(result, f.name)
            for f in fields(result)
            if f.name not in _UNSTORED_RESULT_FIELDS
        }
        base = progress_base(self.output_dir, self.model_name, spec, task_hash)
        try:
            _write_json_atomic(
                base.with_name(base.name + FINAL_SUFFIX),
                {
                    "header": task.header,
                    "outputs": [str(paths[kind].relative_to(self.output_dir)) for kind in wrote],
                    "result": encode_value(stored),
                },
            )
        except Exception as e:  # noqa: BLE001 - losing the record must not stop the run
            logger.error(f"Could not record the finalized result of {spec}: {e}")

    def close(self) -> None:
        """Write out and fsync every record."""
        for task in self._tasks.values():
            try:
                task.store.close()
            except OSError as e:
                logger.warning(f"Could not write progress record {task.store.path}: {e}")


def open_run_progress(
    output_dir: str,
    model_name: str,
    harness_config: HarnessConfig,
) -> RunProgress:
    """RunProgress for this run, with its identity and debug settings from the environment."""
    from olmo_eval.common.types import compute_model_hash
    from olmo_eval.upload.provenance import git_info

    return RunProgress(
        output_dir=output_dir,
        model_name=model_name,
        model_hash=compute_model_hash(harness_config.provider.to_dict()),
        model_revision=harness_config.provider.revision,
        harness_hash=harness_hash(harness_config),
        git_commit=git_info().get("commit"),
        execution=execution_info(harness_config),
        exit_thresholds=parse_exit_thresholds(os.environ.get(DEBUG_EXIT_AT_ENV)),
    )


__all__ = [
    "DEBUG_EXIT_AT_ENV",
    "PROGRESS_LAYOUT_ENV",
    "ProgressMismatchError",
    "ProgressStore",
    "Restored",
    "RunProgress",
    "discard_progress",
    "has_progress",
    "open_run_progress",
]

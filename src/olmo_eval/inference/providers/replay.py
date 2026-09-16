"""Replay provider: serve generations that were already produced and saved to disk.

`olmo-eval run` couples generation and scoring. When the generations already exist
(they were produced by an earlier run and written to `predictions/` and `requests/`),
re-running the model would produce *different* text and score that instead. This
provider closes the gap: it loads the saved requests/predictions pair, joins them,
and hands the stored text back to the normal scoring path.

Wiring::

    -o provider.kind=python \
    -o provider.kwargs.class=olmo_eval.inference.providers.replay.StoredPredictionsProvider \
    -o provider.kwargs.results_dir=/path/to/results

Design rules, in the order they matter:

1. Never answer the wrong instance. `LMRequest` carries no instance identifier, so
   the lookup keys off request *content* (see `_request_key`). Matching is exact on a
   structurally canonicalised key -- never fuzzy, never nearest-neighbour. Any key that
   would resolve to two different stored texts is rejected at load time.
2. Never silently answer nothing. A stored prediction that *is* the empty string is a
   real result and is replayed as an empty string. A request with no stored prediction
   is a bug and raises. The two are represented differently end to end.
3. Report coverage. Every `generate` call logs matched/missing/empty counts, and a
   single missing request aborts the run instead of quietly degrading it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import LMOutput, LMRequest, SamplingParams
from olmo_eval.inference.base import InferenceProvider
from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX

logger = get_logger(__name__)

# How many offending examples to name in an error message before truncating.
_MAX_REPORTED_EXAMPLES = 5
# How much of a request key preview to keep for diagnostics.
_KEY_PREVIEW_CHARS = 200


class ReplayError(RuntimeError):
    """Base class for every failure raised by the replay provider."""


class ReplayInputError(ReplayError):
    """Saved artifacts are missing, ambiguous, or internally inconsistent."""


class ReplayCoverageError(ReplayError):
    """One or more live requests had no stored prediction to replay."""


@dataclass(frozen=True, slots=True)
class _StoredPrediction:
    """A stored prediction joined to the request that produced it."""

    doc_id: Any
    native_id: Any
    texts: tuple[str, ...]
    trajectory: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _UnusableEntry:
    """A saved request whose prediction cannot be replayed, and why.

    Kept separately from the usable index so that a lookup which lands here reports
    "the instance is known but its prediction is missing" instead of the much weaker
    "this request is unknown". Conflating those two has cost real debugging time.
    """

    doc_id: Any
    native_id: Any
    reason: str


@dataclass
class ReplayCoverage:
    """Coverage accounting for everything this provider has been asked to replay."""

    stored_requests: int = 0
    stored_predictions: int = 0
    replayable_entries: int = 0
    unusable_entries: int = 0
    requested: int = 0
    matched: int = 0
    missing: int = 0
    empty_text_replayed: int = 0
    used_keys: set[str] = field(default_factory=set)

    @property
    def unused_entries(self) -> int:
        """Stored entries that no live request ever asked for."""
        return max(self.replayable_entries - len(self.used_keys), 0)

    def as_dict(self) -> dict[str, int]:
        """Return a plain dict, for logging and for assertions in tests."""
        return {
            "stored_requests": self.stored_requests,
            "stored_predictions": self.stored_predictions,
            "replayable_entries": self.replayable_entries,
            "unusable_entries": self.unusable_entries,
            "requested": self.requested,
            "matched": self.matched,
            "missing": self.missing,
            "empty_text_replayed": self.empty_text_replayed,
            "unused_entries": self.unused_entries,
        }


def _canonical_json(value: Any) -> str:
    """Serialise `value` so that structurally identical data yields identical text.

    Sorting keys and using compact separators removes formatting differences, and
    `ensure_ascii=False` keeps both sides of the comparison in the same encoding.
    Tuples serialise as JSON arrays, which is what makes an in-memory `LMRequest`
    comparable to the same request after a round trip through JSONL.
    """
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash_key(canonical: str) -> str:
    """Hash a canonical key so the index stays small regardless of prompt length."""
    return sha256(canonical.encode("utf-8")).hexdigest()


def _key_from_context(context: Any) -> tuple[str, str]:
    """Build the lookup key for one request payload.

    `context` is either the chat message list or the plain-text prompt. The tag in
    front of the payload keeps the two namespaces apart, so a prompt string can never
    collide with a chat conversation that happens to serialise the same way.

    Returns:
        (hashed key, human-readable preview for diagnostics).
    """
    if isinstance(context, (list, tuple)):
        canonical = _canonical_json({"kind": "chat", "messages": list(context)})
    elif isinstance(context, str):
        canonical = _canonical_json({"kind": "text", "prompt": context})
    else:
        raise ReplayInputError(
            f"Cannot build a replay key from request context of type {type(context).__name__}; "
            "expected a chat message list or a prompt string."
        )
    return _hash_key(canonical), canonical[:_KEY_PREVIEW_CHARS]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, reporting the offending line number on bad JSON."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayInputError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ReplayInputError(f"{path}:{lineno} is not a JSON object")
            rows.append(row)
    return rows


def _resolve_single_file(
    root: Path,
    suffix: str,
    explicit: str | None,
    task_filter: str | None,
    label: str,
    base: Path,
) -> Path:
    """Find exactly one artifact file under `root`, or refuse to guess.

    Picking one of several candidates would silently decide which run gets scored, so
    ambiguity is an error. `task_filter` is the supported way to narrow the choice.

    Args:
        root: Directory to glob (`<results_dir>/requests` or `<results_dir>/predictions`).
        suffix: Filename suffix identifying the artifact type.
        explicit: Caller-supplied path, absolute or relative to `base`.
        task_filter: Optional substring filter over candidate filenames.
        label: Human-readable artifact name used in error messages.
        base: Directory that relative `explicit` paths resolve against.
    """
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise ReplayInputError(f"Explicit {label} file does not exist: {path}")
        return path

    if not root.is_dir():
        raise ReplayInputError(f"Expected a '{root.name}' directory at {root}, but it is missing.")

    candidates = sorted(p for p in root.rglob(f"*{suffix}") if p.is_file())
    if task_filter:
        candidates = [p for p in candidates if task_filter in p.name]

    if not candidates:
        hint = f" matching task_filter={task_filter!r}" if task_filter else ""
        raise ReplayInputError(f"No *{suffix} file found under {root}{hint}.")
    if len(candidates) > 1:
        listed = "\n  ".join(str(p) for p in candidates)
        raise ReplayInputError(
            f"Ambiguous {label} input: {len(candidates)} files match *{suffix} under {root}. "
            "Refusing to pick one. Narrow it with provider.kwargs.task_filter=<substring> or "
            f"name the file with provider.kwargs.{label}_file=<path>. Candidates:\n  {listed}"
        )
    return candidates[0]


class StoredPredictionsProvider(InferenceProvider):
    """Replay generations from a saved `predictions/` + `requests/` pair.

    The provider is generation-only. `logprobs` raises rather than returning anything,
    because a replayed run has no token-level scores and a plausible-looking stand-in
    would corrupt any metric derived from them.

    Args:
        model_name: Model identifier. Only used for labelling; nothing is loaded.
        results_dir: Directory holding `predictions/` and `requests/` subdirectories.
        task_filter: Optional substring used to disambiguate when the results directory
            holds artifacts for more than one task or model.
        predictions_file: Optional explicit path to the predictions JSONL, absolute or
            relative to `results_dir`. Bypasses globbing.
        requests_file: Optional explicit path to the requests JSONL, same rules.
        preserve_trajectory: When True (default), a `trajectory` recorded on the stored
            prediction is re-attached under `LMOutput.metadata["trajectory"]`, which is
            where the runner looks when rebuilding `Response.trajectory`.
    """

    def __init__(
        self,
        model_name: str,
        results_dir: str | None = None,
        task_filter: str | None = None,
        predictions_file: str | None = None,
        requests_file: str | None = None,
        preserve_trajectory: bool = True,
    ) -> None:
        super().__init__(model_name)

        if results_dir is None and (predictions_file is None or requests_file is None):
            raise ReplayInputError(
                "StoredPredictionsProvider needs provider.kwargs.results_dir (a directory "
                "containing predictions/ and requests/), or both predictions_file and "
                "requests_file."
            )

        self.results_dir = Path(results_dir).expanduser() if results_dir else Path()
        self.preserve_trajectory = preserve_trajectory

        self.requests_path = _resolve_single_file(
            self.results_dir / "requests",
            REQUESTS_SUFFIX,
            requests_file,
            task_filter,
            "requests",
            self.results_dir,
        )
        self.predictions_path = _resolve_single_file(
            self.results_dir / "predictions",
            PREDICTIONS_SUFFIX,
            predictions_file,
            task_filter,
            "predictions",
            self.results_dir,
        )

        self.coverage = ReplayCoverage()
        self._entries: dict[str, _StoredPrediction] = {}
        self._unusable: dict[str, _UnusableEntry] = {}
        self._load()

    # ─────────────────────────────────────────────────────────
    # Loading and joining
    # ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Join the saved requests to the saved predictions and index them by key."""
        request_rows = _read_jsonl(self.requests_path)
        prediction_rows = _read_jsonl(self.predictions_path)
        self.coverage.stored_requests = len(request_rows)
        self.coverage.stored_predictions = len(prediction_rows)

        predictions_by_doc_id = self._index_predictions(prediction_rows)

        for lineno, row in enumerate(request_rows, start=1):
            if "doc_id" not in row:
                raise ReplayInputError(f"{self.requests_path}:{lineno} has no 'doc_id' field.")
            doc_id = row["doc_id"]
            native_id = row.get("native_id")

            request_payload = row.get("request")
            if not isinstance(request_payload, dict) or "context" not in request_payload:
                raise ReplayInputError(
                    f"{self.requests_path}:{lineno} (doc_id={doc_id!r}) has no request.context; "
                    "this file cannot be used for replay."
                )
            key, _ = _key_from_context(request_payload["context"])

            prediction = predictions_by_doc_id.get(doc_id)
            if prediction is None:
                self._record_unusable(
                    key, doc_id, native_id, f"no prediction row with doc_id={doc_id!r}"
                )
                continue

            self._check_native_id(doc_id, native_id, prediction.get("native_id"))

            texts, reason = _extract_texts(prediction)
            if texts is None:
                self._record_unusable(key, doc_id, native_id, reason)
                continue

            entry = _StoredPrediction(
                doc_id=doc_id,
                native_id=native_id,
                texts=texts,
                trajectory=_trajectory_of(prediction) if self.preserve_trajectory else None,
            )
            self._register(key, entry)

        self.coverage.replayable_entries = len(self._entries)
        self.coverage.unusable_entries = len(self._unusable)
        self._log_load_summary()

    def _index_predictions(self, rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
        """Index prediction rows by doc_id, refusing duplicates."""
        by_doc_id: dict[Any, dict[str, Any]] = {}
        for lineno, row in enumerate(rows, start=1):
            if "doc_id" not in row:
                raise ReplayInputError(f"{self.predictions_path}:{lineno} has no 'doc_id' field.")
            doc_id = row["doc_id"]
            if doc_id in by_doc_id:
                raise ReplayInputError(
                    f"{self.predictions_path}:{lineno} repeats doc_id={doc_id!r}. Refusing to "
                    "choose between duplicate predictions for the same instance."
                )
            by_doc_id[doc_id] = row
        return by_doc_id

    def _check_native_id(self, doc_id: Any, request_native: Any, prediction_native: Any) -> None:
        """Fail if the two files disagree about which instance a doc_id refers to.

        The join runs on doc_id; native_id is the independent witness that the two
        files describe the same run. If they disagree, the files are misaligned and
        every replayed answer would be attached to the wrong instance.
        """
        if request_native is None or prediction_native is None:
            return
        if request_native != prediction_native:
            raise ReplayInputError(
                f"doc_id={doc_id!r} has native_id={request_native!r} in {self.requests_path} but "
                f"native_id={prediction_native!r} in {self.predictions_path}. The two files do "
                "not describe the same run; replaying them would answer the wrong instances."
            )

    def _register(self, key: str, entry: _StoredPrediction) -> None:
        """Add an entry to the index, rejecting conflicting duplicates."""
        existing = self._entries.get(key)
        if existing is not None and existing.texts != entry.texts:
            raise ReplayInputError(
                f"Two stored instances share an identical request but have different stored "
                f"text (doc_id={existing.doc_id!r} and doc_id={entry.doc_id!r}). The request "
                "content cannot identify which one to replay; refusing to guess."
            )
        self._entries[key] = entry
        # A usable entry wins over any earlier unusable record for the same key.
        self._unusable.pop(key, None)

    def _record_unusable(self, key: str, doc_id: Any, native_id: Any, reason: str) -> None:
        """Remember why a saved request has nothing replayable behind it."""
        if key in self._entries:
            return
        self._unusable[key] = _UnusableEntry(doc_id=doc_id, native_id=native_id, reason=reason)

    def _log_load_summary(self) -> None:
        logger.info(
            "Replay index built from %s and %s: %d saved requests, %d saved predictions, "
            "%d replayable, %d unusable.",
            self.requests_path,
            self.predictions_path,
            self.coverage.stored_requests,
            self.coverage.stored_predictions,
            self.coverage.replayable_entries,
            self.coverage.unusable_entries,
        )
        if self._unusable:
            examples = list(self._unusable.values())[:_MAX_REPORTED_EXAMPLES]
            detail = "; ".join(
                f"doc_id={e.doc_id!r} native_id={e.native_id!r}: {e.reason}" for e in examples
            )
            logger.warning(
                "%d saved request(s) have no replayable prediction. Asking for any of them "
                "will fail rather than return an empty answer. Examples: %s",
                len(self._unusable),
                detail,
            )

    # ─────────────────────────────────────────────────────────
    # Lookup
    # ─────────────────────────────────────────────────────────

    def _request_key(self, request: LMRequest) -> tuple[str, str]:
        """Derive the lookup key for a live request.

        Chat messages take precedence over `prompt` because a chat request stores its
        payload in `messages`, and that is exactly what the saved `request.context`
        holds for chat requests. Tools, sampling parameters and stop sequences are
        deliberately excluded: they describe *how* to generate, not *which instance*
        this is, and including them would make replay fail whenever the scoring run
        configures generation slightly differently.
        """
        if request.messages:
            return _key_from_context([dict(message) for message in request.messages])
        return _key_from_context(request.prompt)

    def has_stored_prediction(self, request: LMRequest) -> bool:
        """Whether `request` can be replayed. Useful for a dry run before scoring."""
        key, _ = self._request_key(request)
        return key in self._entries

    # ─────────────────────────────────────────────────────────
    # InferenceProvider interface
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Replay the stored generations for `requests`.

        Raises:
            ReplayCoverageError: If any request has no stored prediction. Partial runs
                are never returned, because a missing instance that quietly scores as
                an empty answer is indistinguishable from a genuinely empty answer.
        """
        params = self._default_sampling_params(sampling_params)
        num_samples = max(int(params.num_samples or 1), 1)

        results: list[list[LMOutput]] = []
        missing: list[str] = []

        for index, request in enumerate(requests):
            self.coverage.requested += 1
            key, preview = self._request_key(request)
            entry = self._entries.get(key)

            if entry is None:
                self.coverage.missing += 1
                missing.append(self._describe_miss(index, key, preview))
                results.append([])
                continue

            if num_samples > len(entry.texts):
                raise ReplayCoverageError(
                    f"Request {index} (doc_id={entry.doc_id!r}, native_id={entry.native_id!r}) "
                    f"asks for {num_samples} sample(s) but only {len(entry.texts)} stored "
                    "generation(s) exist. Replay never fabricates additional samples."
                )
            if num_samples < len(entry.texts):
                logger.warning(
                    "doc_id=%r has %d stored generations but only %d sample(s) were requested; "
                    "replaying the first %d.",
                    entry.doc_id,
                    len(entry.texts),
                    num_samples,
                    num_samples,
                )

            self.coverage.matched += 1
            self.coverage.used_keys.add(key)
            outputs = [self._build_output(entry, sample) for sample in range(num_samples)]
            # An empty stored generation is a real result, not a miss. Count it so the
            # run log shows how many empties were replayed on purpose.
            if any(output.text == "" for output in outputs):
                self.coverage.empty_text_replayed += 1
            results.append(outputs)

        self._log_batch_coverage(len(requests), len(missing))

        if missing:
            shown = missing[:_MAX_REPORTED_EXAMPLES]
            suffix = (
                f"\n  ... and {len(missing) - len(shown)} more" if len(missing) > len(shown) else ""
            )
            raise ReplayCoverageError(
                f"{len(missing)} of {len(requests)} request(s) have no stored prediction to "
                f"replay from {self.predictions_path}. Refusing to return a partial batch, "
                "because an unanswered instance must not look like an empty answer.\n  "
                + "\n  ".join(shown)
                + suffix
            )

        return results

    async def agenerate(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Async generate. Replay is a dictionary lookup, so it just calls generate."""
        return self.generate(requests, sampling_params)

    def logprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Always raises: saved predictions carry no token-level scores.

        Returning zeros or recomputed values here would silently produce a wrong
        likelihood metric, which is worse than not running at all.
        """
        raise NotImplementedError(
            "StoredPredictionsProvider replays saved generations and has no logprobs. "
            "Saved predictions do not carry token-level scores, so any value returned here "
            "would be fabricated. Score loglikelihood tasks with a real provider."
        )

    async def alogprobs(
        self,
        requests: list[LMRequest],
        sampling_params: SamplingParams | None = None,
    ) -> list[list[LMOutput]]:
        """Always raises, for the same reason as `logprobs`."""
        return self.logprobs(requests, sampling_params)

    # ─────────────────────────────────────────────────────────
    # Reporting helpers
    # ─────────────────────────────────────────────────────────

    def _build_output(self, entry: _StoredPrediction, sample_index: int) -> LMOutput:
        """Build the LMOutput handed back to the scoring path.

        `extracted_answer` is deliberately left unset: the task recomputes it from
        `text` during scoring, so replaying a stored value would be dead weight at
        best and stale at worst. Logprob-derived metadata (`sum_logits`, `num_tokens`,
        ...) is deliberately absent -- downstream code treats its absence as
        "unavailable" and its presence as a real measurement.
        """
        metadata: dict[str, Any] = {
            "replay_source": {
                "predictions_file": str(self.predictions_path),
                "requests_file": str(self.requests_path),
                "doc_id": entry.doc_id,
                "native_id": entry.native_id,
                "sample_index": sample_index,
            }
        }
        # The runner rebuilds Response.trajectory from metadata["trajectory"], and
        # tasks such as litsearch/expertqa/sage score against that trajectory. Keeping
        # it means a replayed run reproduces those tasks too, and keeps the trajectory
        # column in the re-written predictions file. DeepResearch Bench itself scores
        # only the text, so this is preservation rather than a requirement for DRB.
        if sample_index == 0 and entry.trajectory is not None:
            metadata["trajectory"] = entry.trajectory
        return LMOutput(text=entry.texts[sample_index], metadata=metadata)

    def _describe_miss(self, index: int, key: str, preview: str) -> str:
        """Explain one lookup failure as precisely as the saved data allows."""
        known = self._unusable.get(key)
        if known is not None:
            return (
                f"request {index}: matches saved doc_id={known.doc_id!r} "
                f"native_id={known.native_id!r} but {known.reason}"
            )
        return f"request {index}: no saved request matches this content. key={key[:12]} {preview!r}"

    def _log_batch_coverage(self, batch_size: int, missing: int) -> None:
        logger.info(
            "Replay coverage: %d/%d matched in this batch (%d missing); cumulative "
            "%d requested, %d matched, %d missing, %d empty-but-present, %d stored entries "
            "never requested.",
            batch_size - missing,
            batch_size,
            missing,
            self.coverage.requested,
            self.coverage.matched,
            self.coverage.missing,
            self.coverage.empty_text_replayed,
            self.coverage.unused_entries,
        )

    def coverage_report(self) -> dict[str, int]:
        """Return the cumulative coverage counters."""
        return self.coverage.as_dict()


def _extract_texts(prediction: dict[str, Any]) -> tuple[tuple[str, ...] | None, str]:
    """Pull the generated text(s) out of a stored prediction row.

    Returns:
        (texts, reason). `texts` is None when nothing replayable is present, and
        `reason` then explains what was wrong. An empty string inside `texts` is a
        real, replayable result and never becomes None.
    """
    model_output = prediction.get("model_output")
    if not isinstance(model_output, list) or not model_output:
        return None, "prediction row has an empty or missing 'model_output' list"

    texts: list[str] = []
    for position, raw_output in enumerate(model_output):
        if not isinstance(raw_output, dict):
            return None, f"model_output[{position}] is not an object"
        output = cast("dict[str, Any]", raw_output)
        if "text" not in output:
            return None, f"model_output[{position}] has no 'text' field"
        text = output["text"]
        if text is None:
            return None, f"model_output[{position}]['text'] is null (no generation recorded)"
        if not isinstance(text, str):
            return None, f"model_output[{position}]['text'] is {type(text).__name__}, not a string"
        texts.append(text)
    return tuple(texts), ""


def _trajectory_of(prediction: dict[str, Any]) -> dict[str, Any] | None:
    """Return the stored trajectory dict, if the prediction row carries one."""
    trajectory = prediction.get("trajectory")
    return trajectory if isinstance(trajectory, dict) else None

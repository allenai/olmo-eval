"""Resuming a run from the progress records an earlier attempt left in its output directory.

The end-to-end tests run the async runner on real task specs, with each task's
dataset replaced by a few local documents and the mock provider replaced by
``resume_fakes.HashProvider``, whose answers differ by request. A resumed run must
write the same predictions, requests and metrics as an uninterrupted one.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import resume_fakes

import olmo_eval.evals.tasks  # noqa: F401  (registers the tasks)
from olmo_eval.cli.run import _sigterm_raises
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import Judge
from olmo_eval.common.shutdown import (
    Terminated,
    clear_shutdown,
    request_shutdown,
    shutdown_requested,
)
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    ProviderKind,
    RequestType,
    Response,
)
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq import progress as progress_module
from olmo_eval.runners.asynq import runner as runner_module
from olmo_eval.runners.asynq.progress import (
    DEBUG_EXIT_AT_ENV,
    FINAL_SUFFIX,
    PROGRESS_DIR,
    RECORD_VERSION,
    DroppedLogprobsError,
    ProgressLockedError,
    ProgressMismatchError,
    ProgressStore,
    Restored,
    RunProgress,
    decode_value,
    discard_progress,
    encode_value,
    has_progress,
    open_run_progress,
    parse_exit_thresholds,
    stream_dir,
)
from olmo_eval.runners.asynq.results import process_results
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker
from olmo_eval.runners.asynq.workers import inference_worker as real_inference_worker
from olmo_eval.upload.manifest import RUN_SECRET_NAME, read_manifest, run_secret

# One spec per kind of task: log-likelihood multiple choice, generation with exact
# match, instruction following, a scorer that writes instance metadata (bbq), many
# samples per instance, and a model judge.
TASK_LIMITS = {
    "arc_challenge:mc": 8,
    "gsm8k": 8,
    "ifeval": 8,
    "bbq:mcq": 8,
    "aime_2024:pass_at_32": 4,
    "frontierscience_olympiad": 8,
}
TOTAL = sum(TASK_LIMITS.values())

# Each prompt carries the right answer as <<say:...>>, which HashProvider repeats for
# about half of its samples.
_FAKE_DOCS: dict[str, list[dict[str, Any]]] = {
    "allenai/ai2_arc": [
        {
            "id": f"arc-{i}",
            "question": f"Which choice is number {i}?",
            "choices": {
                "text": [f"first {i}", f"second {i}", f"third {i}", f"fourth {i}"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "ABCD"[i % 4],
        }
        for i in range(16)
    ],
    "gsm8k": [
        {
            "question": f"What is {i} plus {i}? <<say:The answer is {2 * i}.>>",
            "answer": f"{i} + {i} = {2 * i}\n#### {2 * i}",
        }
        for i in range(16)
    ],
    "google/IFEval": [
        {
            "key": 1000 + i,
            "prompt": f"Write about topic {i} without commas. <<say:Topic {i} is fine.>>",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        }
        for i in range(16)
    ],
    "allenai/olmo-eval-bbq": [
        {
            "id": i,
            "question": f"Person A and person B met. Who did thing {i}? "
            f"<<say:The answer is ({'ABC'[i % 3]}).>>",
            "choices": ["Person A", "Person B", "Not enough info"],
            "gold_label": "ABC"[i % 3],
            "bias_label": "A",
            "unknown_label": "C",
            "question_polarity": "neg" if i % 2 else "nonneg",
            "context_condition": "ambig" if i % 2 else "disambig",
            "category": "Age",
            "subcategory": "[]",
            "subsample": True,
        }
        for i in range(16)
    ],
    "allenai/aime-2022-2025": [
        {
            "id": i,
            "problem": f"Find x for problem {i}. <<say:So x is \\boxed{{{i * 7}}}.>>",
            "answer": str(i * 7),
            "year": 2024,
            "problem_number": i,
        }
        for i in range(8)
    ],
    "openai/frontierscience": [
        {
            "problem": f"Problem {i}. Answer after FINAL ANSWER. <<say:FINAL ANSWER: {i}>>",
            "answer": str(i),
            "subject": "physics",
            "task_group_id": f"group-{i}",
        }
        for i in range(16)
    ],
}


@pytest.fixture(scope="module", autouse=True)
def offline_tasks() -> Iterator[None]:
    """Serve each task's data from _FAKE_DOCS, answer with HashProvider, judge locally.

    Data loading and scoring run in the main process; workers run HashProvider.
    """
    from olmo_eval.data import DataLoader
    from olmo_eval.evals.tasks.common.judge import JudgedTask

    def load(self: Any, source: Any, streaming: bool = False) -> Iterator[dict[str, Any]]:
        yield from (dict(doc) for doc in _FAKE_DOCS[source.path])

    fake_judge = Judge(call=resume_fakes.judge, attempts=1, max_retries=0, retry_delay=0.0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DataLoader, "load", load)
        mp.setattr(JudgedTask, "_get_judge", lambda self: fake_judge)
        mp.setattr(
            "olmo_eval.runners.asynq.workers.inference_worker", resume_fakes.inference_worker
        )
        yield


@pytest.fixture
def runner_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """``caplog`` that also sees the runner's own logger, which does not propagate."""
    runner_module.runner_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        runner_module.runner_logger.removeHandler(caplog.handler)


def _harness(system_prompt: str | None = None, model: str = "mock", **kwargs: Any) -> HarnessConfig:
    # HashProvider fails some requests on purpose; keep every task's metrics.
    return HarnessConfig(
        name="default",
        provider=ProviderConfig(kind="mock", model=model, **kwargs),
        max_hard_failure_rate=1.0,
        system_prompt=system_prompt,
    )


def _runner(
    out: Path,
    limits: dict[str, int] = TASK_LIMITS,
    system_prompt: str | None = None,
    model: str = "mock",
    provider: dict[str, Any] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    **kwargs: Any,
) -> AsyncEvalRunner:
    """A runner for ``limits``; ``provider`` and ``overrides`` add provider and task settings."""
    overrides = overrides or {}
    return AsyncEvalRunner(
        harness_config=_harness(system_prompt, model, **(provider or {})),
        task_specs=list(limits),
        task_overrides={
            spec: {"limit": limit, **overrides.get(spec, {})} for spec, limit in limits.items()
        },
        output_dir=str(out),
        **kwargs,
    )


@dataclass
class RunStats:
    queued: int | None
    restored: int
    finalized: set[str]
    judge_calls: int


def run_eval(
    out: Path,
    limits: dict[str, int] = TASK_LIMITS,
    env: dict[str, str] | None = None,
    resume: bool = True,
    **kwargs: Any,
) -> RunStats:
    """Run the evaluation into ``out`` under the CLI's SIGTERM handling.

    Progress is only recorded with ``resume``, which is on unless a test turns it off.
    ``OLMO_EVAL_DEBUG_EXIT_AT`` ends the run through the CLI's SIGTERM path, without
    sending a signal to the test process: a real one would end a test session that runs
    tests on threads, where no handler is installed.
    """
    seen: dict[str, Any] = {"queued": None, "restored": 0, "finalized": set()}

    async def spy(**call: Any) -> Any:
        seen["queued"] = call["total_instances"]
        seen["restored"] = sum(len(r) for r in call["restored"].values()) + sum(
            len(t.failed_instances) for t in call["trackers"].values()
        )
        seen["finalized"] = set(call["finalized"])
        return await process_results(**call)

    real_kill = os.kill

    def sigterm(pid: int, sig: int) -> None:
        if pid != os.getpid() or sig == 0:
            real_kill(pid, sig)  # the runner stopping its workers, or a liveness check
            return
        assert sig == signal.SIGTERM
        # What the CLI's handler does when the signal arrives.
        request_shutdown()
        raise Terminated()

    judge_calls = len(resume_fakes.JUDGE_CALLS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(progress_module.os, "kill", sigterm)
        mp.delenv(DEBUG_EXIT_AT_ENV, raising=False)
        for key, value in (env or {}).items():
            mp.setenv(key, value)
        mp.setattr(runner_module, "process_results", spy)
        try:
            with _sigterm_raises():
                _runner(out, limits, resume=resume, **kwargs).run()
        finally:
            seen["judge_calls"] = len(resume_fakes.JUDGE_CALLS) - judge_calls
    return RunStats(seen["queued"], seen["restored"], seen["finalized"], seen["judge_calls"])


_TIMING_KEYS = {
    "timestamp",
    "duration_seconds",
    "experiment_duration_seconds",
    "startup_seconds",
    "processing_started_at",
    "processing_seconds",
    "first_request_at",
    "last_completed_at",
    "attributed_inference_seconds",
    "scoring_judge_seconds",
    "scoring_sandbox_seconds",
    "scoring_compute_seconds",
    "provider_init_seconds",
    "experiment_id",
}


def _without_timing(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _without_arrival_order(k, _without_timing(v))
            for k, v in value.items()
            if k not in _TIMING_KEYS
        }
    if isinstance(value, list):
        return [_without_timing(v) for v in value]
    return value


def _without_arrival_order(key: str, value: Any) -> Any:
    # "(first: ...)" names whichever failure arrived first, and a restored failure
    # always arrives before a rerun one; no run fixes that order.
    if key == "error_summary" and isinstance(value, str):
        return re.sub(r"\(first: .*\)", "(first: <any>)", value)
    return value


def _metrics(out: Path) -> dict[str, Any]:
    metrics = json.loads((out / "metrics.json").read_text())
    # Tasks are listed in the order they finished, which no run fixes.
    metrics["tasks"] = sorted(metrics["tasks"], key=lambda task: task["task"])
    return _without_timing(metrics)


def _artifacts(out: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(out)): path.read_bytes()
        for sub in ("predictions", "requests")
        for path in sorted((out / sub).rglob("*.jsonl"))
    }


def assert_same_results(resumed: Path, reference: Path) -> None:
    resumed_files, reference_files = _artifacts(resumed), _artifacts(reference)
    assert sorted(resumed_files) == sorted(reference_files)
    assert reference_files and len(reference_files) % 2 == 0
    for name, content in reference_files.items():
        assert resumed_files[name] == content, f"{name} differs"
    assert _metrics(resumed) == _metrics(reference)


def _stream(out: Path) -> ProgressStore:
    return ProgressStore(stream_dir(out, "mock"))


def _entries(out: Path) -> list[dict[str, Any]]:
    return list(_stream(out).read())


def _instance_entries(out: Path) -> list[dict[str, Any]]:
    return [entry for entry in _entries(out) if "idx" in entry]


def _recorded(out: Path) -> int:
    return len(_instance_entries(out))


def keep_first_entries(out: Path, keep: int) -> int:
    """Cut the records back to ``keep`` instance entries in all, as an interrupted run leaves.

    Finalized results go too. Returns the number of entries kept.
    """
    for final in (out / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"):
        final.unlink()
    store = _stream(out)
    entries = list(store.read())
    store.reset()
    kept_total = 0
    for entry in entries:
        if "idx" in entry:
            if kept_total >= keep:
                continue
            kept_total += 1
        store.append(entry)
    store.close()
    return kept_total


def _clear_outputs(out: Path) -> None:
    for sub in ("predictions", "requests"):
        shutil.rmtree(out / sub, ignore_errors=True)
    (out / "metrics.json").unlink()


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory):
    """An uninterrupted run with its records kept."""
    out = tmp_path_factory.mktemp("full")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RunProgress, "discard", lambda self: None)
        stats = run_eval(out)
    assert stats.queued == TOTAL and stats.restored == 0
    return out


INTERRUPTED_LIMITS = {"gsm8k": 8}


@pytest.fixture(scope="module")
def interrupted_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run of one task that SIGTERM ended part way, as Beaker job ``job-1``."""
    out = tmp_path_factory.mktemp("interrupted")
    with pytest.raises(Terminated):
        run_eval(out, INTERRUPTED_LIMITS, env={DEBUG_EXIT_AT_ENV: "0.3", "BEAKER_JOB_ID": "job-1"})
    assert has_progress(out) and _recorded(out) >= 2
    return out


def _copy(full: Path, tmp_path: Path, *, keep_outputs: bool = False) -> Path:
    out = tmp_path / "run"
    shutil.copytree(full, out)
    if not keep_outputs:
        _clear_outputs(out)
    return out


class TestRoundTrip:
    def test_records_hold_outcomes_not_prompts(self, full_run) -> None:
        full = full_run
        entries = _entries(full)
        headers = [entry for entry in entries if "header" in entry]
        assert len(headers) == len(TASK_LIMITS)
        assert len(list((full / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))) == len(TASK_LIMITS)
        for entry in headers:
            header = entry["header"]
            assert header["task_hash"] and header["model_hash"] and header["harness_hash"]
            assert "engine_version" in header and "execution" in header
        instances = [entry for entry in entries if "idx" in entry]
        # Only failures the request caused are recorded; the fake's connection errors
        # are retried, so they never appear.
        failed = [e for e in instances if "failed" in e]
        assert failed and all(e["failed"] == resume_fakes.FAILED_REQUEST for e in failed)
        assert resume_fakes.TRANSIENT_REQUEST not in json.dumps(entries)
        assert len({(e["t"], e["idx"]) for e in instances}) == len(instances)
        assert "<<say:" not in json.dumps(instances)
        # Instances whose judge was unavailable are not final, so they are not recorded.
        assert len(instances) < TOTAL

    def test_one_stream_holds_every_task(self, full_run) -> None:
        full = full_run
        stream = _stream(full)
        assert stream.path.exists()
        assert len(list((full / PROGRESS_DIR).glob("*/*.segments"))) == 1
        # A run is written in a few segments however many tasks it has: the flush at
        # close, plus none by size or age within these seconds.
        assert len(list(stream.path.glob("*.jsonl"))) == 1

    def test_finalized_tasks_are_not_run_again(self, full_run, tmp_path) -> None:
        full = full_run
        out = _copy(full, tmp_path, keep_outputs=True)
        (out / "metrics.json").unlink()

        stats = run_eval(out, resume=True)

        assert stats.queued == 0
        assert stats.judge_calls == 0
        assert stats.finalized == set(TASK_LIMITS)
        assert_same_results(out, full)
        assert not (out / PROGRESS_DIR).exists()

    def test_rerun_from_records_generates_only_unrecorded(self, full_run, tmp_path) -> None:
        full = full_run
        out = _copy(full, tmp_path)

        stats = run_eval(out, resume=True)

        # Predictions are gone, so no finalized result can be used; the records can.
        assert stats.finalized == set()
        assert stats.restored == _recorded(full)
        assert stats.queued == TOTAL - stats.restored
        assert_same_results(out, full)

    @pytest.mark.parametrize("cut", [0, 1, 0.37, 0.99])
    def test_resume_after_a_cut_matches_an_uninterrupted_run(self, full_run, tmp_path, cut) -> None:
        full = full_run
        out = _copy(full, tmp_path)
        kept = keep_first_entries(out, cut if isinstance(cut, int) else int(cut * TOTAL))

        stats = run_eval(out, resume=True)

        # Every instance not taken from the records is generated by this attempt.
        assert stats.restored == kept
        assert stats.queued == TOTAL - kept
        assert_same_results(out, full)

    def test_sigterm_then_resume(self, full_run, tmp_path, runner_log) -> None:
        full = full_run
        out = tmp_path / "run"

        with pytest.raises(Terminated):
            run_eval(out, env={DEBUG_EXIT_AT_ENV: "0.37"})
        recorded = _recorded(out)
        assert recorded >= 0.37 * TOTAL
        specs = {e["t"]: e["header"]["spec"] for e in _entries(out) if "header" in e}
        recorded_by_spec = Counter(specs[e["t"]] for e in _instance_entries(out))
        first = read_manifest(out)
        assert first is not None and first["run"]["status"] == "running"
        secret = run_secret(out)

        # Without --resume the progress is ignored, with a warning, and not touched.
        elsewhere = tmp_path / "ordinary"
        shutil.copytree(out, elsewhere)
        before = sorted(p.name for p in (elsewhere / PROGRESS_DIR).rglob("*"))
        with runner_log.at_level(logging.WARNING):
            ordinary = run_eval(elsewhere, resume=False)
        assert ordinary.queued == TOTAL and ordinary.restored == 0
        assert "holds progress from an earlier run, which this run ignores" in runner_log.text
        assert sorted(p.name for p in (elsewhere / PROGRESS_DIR).rglob("*")) == before
        assert_same_results(elsewhere, full)

        # The threshold was crossed before this attempt started, so it finishes.
        stats = run_eval(out, env={DEBUG_EXIT_AT_ENV: "0.37"}, resume=True)

        # Finalized tasks are taken whole; the other tasks restore what was recorded.
        unfinalized = sum(n for spec, n in recorded_by_spec.items() if spec not in stats.finalized)
        finalized = sum(TASK_LIMITS[spec] for spec in stats.finalized)
        assert stats.finalized and stats.restored == unfinalized
        assert stats.queued == TOTAL - finalized - stats.restored
        assert_same_results(out, full)
        manifest = read_manifest(out)
        assert manifest is not None
        assert manifest["run"]["run_id"] == first["run"]["run_id"]
        assert manifest["run"]["status"] == "complete"
        assert manifest["resume"]["resumed_count"] == 1
        assert (out / RUN_SECRET_NAME).read_text().strip() == secret
        assert not (out / PROGRESS_DIR).exists()

    def test_an_unreadable_finalized_result_is_ignored(self, full_run, tmp_path, caplog) -> None:
        full = full_run
        out = _copy(full, tmp_path, keep_outputs=True)
        (out / "metrics.json").unlink()
        broken = sorted((out / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))[0]
        spec = json.loads(broken.read_text())["header"]["spec"]
        broken.write_text('{"header": ')

        with caplog.at_level(logging.WARNING):
            stats = run_eval(out)

        assert f"Ignoring unreadable finalized result {broken}" in caplog.text
        assert stats.finalized == set(TASK_LIMITS) - {spec}
        assert_same_results(out, full)

    def test_a_finalized_task_whose_predictions_are_missing_is_recomputed(
        self, full_run, tmp_path, caplog
    ) -> None:
        full = full_run
        out = _copy(full, tmp_path, keep_outputs=True)
        (out / "metrics.json").unlink()
        final = sorted((out / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))[0]
        data = json.loads(final.read_text())
        (out / data["outputs"][0]).unlink()

        with caplog.at_level(logging.INFO):
            stats = run_eval(out)

        assert "is missing; recomputing it" in caplog.text
        assert stats.finalized == set(TASK_LIMITS) - {data["header"]["spec"]}
        assert_same_results(out, full)

    def test_a_failing_record_stops_recording_that_task_and_the_run_completes(
        self, full_run, tmp_path, monkeypatch, caplog
    ) -> None:
        full = full_run
        out = tmp_path / "run"
        attempts: list[str] = []
        real_append = ProgressStore.append

        def append(self: ProgressStore, entry: Any) -> None:
            if "header" in entry or "runtime" in entry:
                real_append(self, entry)
                return
            attempts.append(entry["t"])
            raise OSError("disk full")

        monkeypatch.setattr(ProgressStore, "append", append)

        with caplog.at_level(logging.WARNING):
            stats = run_eval(out)

        # Each task tried once; after the failure it records nothing more.
        assert attempts and len(attempts) == len(set(attempts))
        assert caplog.text.count("recording stops for this task") == len(attempts)
        assert "disk full" in caplog.text
        assert stats.queued == TOTAL
        assert_same_results(out, full)

    def test_fresh_discards_recorded_progress(self, full_run, tmp_path) -> None:
        full = full_run
        out = _copy(full, tmp_path)

        stats = run_eval(out, fresh=True, resume=False)

        assert stats.queued == TOTAL and stats.restored == 0
        assert_same_results(out, full)
        assert not has_progress(out)

    def test_a_run_without_resume_writes_no_progress(self, full_run, tmp_path) -> None:
        full = full_run
        out = tmp_path / "ordinary"

        stats = run_eval(out, resume=False)

        assert stats.queued == TOTAL
        assert not (out / PROGRESS_DIR).exists()
        assert_same_results(out, full)

    def test_resume_without_progress_matches_an_ordinary_run(self, full_run, tmp_path) -> None:
        full = full_run
        out = tmp_path / "first"

        run_eval(out, resume=True)

        assert_same_results(out, full)
        assert not (out / PROGRESS_DIR).exists()

    def test_a_system_prompt_reaches_restored_requests(self, tmp_path) -> None:
        # Restored responses carry the request the harness sent, so the saved requests
        # must hold the system prompt exactly as an uninterrupted run saves them.
        limits = {"gsm8k": 6, "arc_challenge:mc": 6}
        reference = tmp_path / "reference"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(RunProgress, "discard", lambda self: None)
            run_eval(reference, limits, system_prompt="Be brief.")
        assert "Be brief." in b"".join(_artifacts(reference).values()).decode()
        out = tmp_path / "cut"
        shutil.copytree(reference, out)
        _clear_outputs(out)
        kept = keep_first_entries(out, 8)

        stats = run_eval(out, limits, system_prompt="Be brief.")

        assert stats.restored == kept and kept > 0
        assert_same_results(out, reference)


class TestStores:
    def test_a_torn_last_line_is_dropped(self, tmp_path) -> None:
        store = ProgressStore(tmp_path / "run")
        store.reset()
        store.append({"header": True})
        store.append({"idx": 0})
        store.close()
        torn = b'{"idx": 1, "outputs": [{"te'
        (store.path / "000099.jsonl").write_bytes(b'{"idx": 1}\n' + torn)

        reopened = ProgressStore(tmp_path / "run")
        expected = [{"header": True}, {"idx": 0}, {"idx": 1}]
        assert list(reopened.read()) == expected
        reopened.append({"idx": 2})
        reopened.close()
        assert list(ProgressStore(tmp_path / "run").read()) == [*expected, {"idx": 2}]

    def test_segments_are_flushed_by_size(self, tmp_path, monkeypatch) -> None:
        # Every line is 50 bytes, so a 100-byte limit writes one segment per two entries.
        def entry(i: int) -> dict[str, Any]:
            return {"idx": f"{i:02d}", "pad": ""}

        line = len(json.dumps(entry(0), separators=(",", ":"))) + 1
        pad = "x" * (50 - line)
        monkeypatch.setattr(progress_module, "SEGMENT_MAX_BYTES", 100)
        store = ProgressStore(tmp_path / "run")
        for i in range(31):
            store.append({**entry(i), "pad": pad})

        assert len(list(store.path.glob("*.jsonl"))) == 15  # the 31st entry is buffered
        store.close()
        assert len(list(store.path.glob("*.jsonl"))) == 16
        assert [e["idx"] for e in ProgressStore(tmp_path / "run").read()] == [
            f"{i:02d}" for i in range(31)
        ]

    def test_buffered_entries_are_flushed_when_they_age(self, tmp_path, monkeypatch) -> None:
        """The flush thread writes the buffer each time its wait times out."""
        flushed = threading.Event()

        class AgingEvent:
            """The first wait times out at once, as if the age limit had passed."""

            def __init__(self) -> None:
                self._set = threading.Event()
                self._waits = 0

            def wait(self, timeout: float) -> bool:
                self._waits += 1
                if self._waits == 1:
                    return False
                flushed.set()  # the flush after the first timeout is done
                return self._set.wait(30)

            def set(self) -> None:
                self._set.set()

        monkeypatch.setattr(progress_module, "SEGMENT_MAX_BYTES", 10**9)
        monkeypatch.setattr(
            progress_module,
            "threading",
            SimpleNamespace(Lock=threading.Lock, Thread=threading.Thread, Event=AgingEvent),
        )
        store = ProgressStore(tmp_path / "aged")
        store.append({"idx": 0})
        assert flushed.wait(30)

        assert [e["idx"] for e in ProgressStore(tmp_path / "aged").read()] == [0]
        store.close()
        assert len(list(store.path.glob("*.jsonl"))) == 1  # close had nothing left to write

    def test_values_round_trip_through_json(self) -> None:
        value = {
            "tuple": (1, "a", (2.5, None)),
            "int_keys": {1: "x", (2, 3): [1.5]},
            "looks_tagged": {"__tuple__": [1]},
            "object": Instance(question="q", choices=("a", "b")),
            "nan": float("nan"),
        }
        decoded = decode_value(json.loads(json.dumps(encode_value(value))))
        assert math.isnan(decoded.pop("nan"))
        value.pop("nan")
        assert decoded == value
        assert isinstance(decoded["tuple"][2], tuple)

    def test_exit_thresholds_are_parsed(self) -> None:
        assert parse_exit_thresholds("0.7, 0.3") == [0.3, 0.7]
        assert parse_exit_thresholds("") == []
        with pytest.raises(ValueError, match="between 0 and 1"):
            parse_exit_thresholds("30")


def _vllm() -> HarnessConfig:
    return HarnessConfig(name="default", provider=ProviderConfig(kind="vllm", model="m"))


def _resume_copy(interrupted_run: Path, tmp_path: Path) -> Path:
    out = tmp_path / "run"
    shutil.copytree(interrupted_run, out)
    return out


class TestHeaders:
    """What a run must match to continue the records of another."""

    def _restore(self, out: Path, harness: HarnessConfig | None = None) -> Any:
        # Below run_eval: a vllm provider cannot run offline, and the git commit and
        # execution fields are set where the progress is opened.
        progress, _, restored = _open(out, harness)
        progress.close()
        return restored

    def test_a_different_max_tokens_names_the_setting(self, interrupted_run, tmp_path) -> None:
        out = _resume_copy(interrupted_run, tmp_path)

        with pytest.raises(
            ProgressMismatchError,
            match=r"task_hash differs.*sampling_params.max_tokens: 512 -> 128",
        ):
            run_eval(out, INTERRUPTED_LIMITS, overrides={"gsm8k": {"max_tokens": 128}})

    def test_a_different_model_revision_names_the_revision(self, interrupted_run, tmp_path) -> None:
        # The model hash covers the revision too, but only the revision is readable.
        out = _resume_copy(interrupted_run, tmp_path)

        with pytest.raises(
            ProgressMismatchError, match=r"model_revision differs \(recorded None, this run 'v2'"
        ):
            run_eval(out, INTERRUPTED_LIMITS, provider={"revision": "v2"})

    def test_a_different_model_is_refused(self, tmp_path) -> None:
        self._restore(tmp_path)

        with pytest.raises(ProgressMismatchError, match="model_hash differs"):
            self._restore(tmp_path, _harness(model="another-model"))

    def test_a_different_system_prompt_is_refused(self, interrupted_run, tmp_path) -> None:
        out = _resume_copy(interrupted_run, tmp_path)

        with pytest.raises(ProgressMismatchError, match="harness_hash differs"):
            run_eval(out, INTERRUPTED_LIMITS, system_prompt="Be brief.")

    def test_a_second_conflicting_header_is_refused(self, tmp_path) -> None:
        """Every header of a task is checked, not only the first."""
        self._restore(tmp_path)
        store = _stream(tmp_path)
        header = next(e for e in store.read() if "header" in e)
        store.append({"t": "local_other", "header": {**header["header"], "task_hash": "f" * 64}})
        store.close()

        with pytest.raises(ProgressMismatchError, match="task_hash differs"):
            self._restore(tmp_path)

    def test_a_different_engine_version_is_refused(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(progress_module, "_package_version", lambda name: "0.10.0")
        self._restore(tmp_path, _vllm())
        monkeypatch.setattr(progress_module, "_package_version", lambda name: "0.11.0")

        with pytest.raises(
            ProgressMismatchError, match=r"engine_version differs.*vllm==0.10.0.*vllm==0.11.0"
        ):
            self._restore(tmp_path, _vllm())

    def test_an_unknown_engine_version_warns_instead_of_matching(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(progress_module, "_package_version", lambda name: "0.10.0")
        self._restore(tmp_path, _vllm())
        monkeypatch.setattr(progress_module, "_package_version", lambda name: None)

        with caplog.at_level("WARNING"):
            self._restore(tmp_path, _vllm())

        assert "without a comparable engine version" in caplog.text
        assert "vllm==unknown" in caplog.text

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("vllm", "vllm==1.2.3"),
            ("sglang", "sglang==1.2.3"),
            ("hf", "transformers==1.2.3"),
            ("litellm", None),
            ("mock", None),
        ],
    )
    def test_the_engine_version_is_that_of_the_engine_package(
        self, monkeypatch, kind, expected
    ) -> None:
        monkeypatch.setattr(progress_module, "_package_version", lambda name: "1.2.3")
        config = HarnessConfig(name="default", provider=ProviderConfig(kind=kind, model="m"))

        assert progress_module.engine_version(config) == expected

    def test_vllm_server_version_comes_from_the_server_python(self, monkeypatch) -> None:
        import subprocess

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="warn\n0.11.2\n", stderr="")

        monkeypatch.setenv("VLLM_PYTHON", "/venvs/vllm/bin/python")
        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr(progress_module, "_package_version", lambda name: None)
        config = HarnessConfig(
            name="default", provider=ProviderConfig(kind="vllm_server", model="m")
        )

        assert progress_module.engine_version(config) == "vllm==0.11.2"
        assert calls[0][0] == "/venvs/vllm/bin/python"

    def test_vllm_server_version_failure_is_recorded_as_unknown(self, monkeypatch, caplog) -> None:
        import subprocess

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No module named vllm")

        monkeypatch.setattr(subprocess, "run", fake_run)
        config = HarnessConfig(
            name="default", provider=ProviderConfig(kind="vllm_server", model="m")
        )

        with caplog.at_level("WARNING"):
            assert progress_module.engine_version(config) == "vllm==unknown"
        assert "Could not determine the vllm version" in caplog.text

    @pytest.mark.parametrize(
        ("recorded", "current", "refused"),
        [
            ("a" * 40, "b" * 40, True),
            ("a" * 40, "a" * 40, False),
            (None, "a" * 40, False),
            ("a" * 40, None, False),
        ],
    )
    def test_a_different_commit_is_refused_unless_one_is_unknown(
        self, tmp_path, monkeypatch, recorded, current, refused
    ) -> None:
        def at(commit: str | None) -> None:
            monkeypatch.setattr("olmo_eval.upload.provenance.git_info", lambda: {"commit": commit})

        at(recorded)
        self._restore(tmp_path)
        stored = next(e for e in _entries(tmp_path) if "header" in e)
        assert stored["header"]["git_commit"] == recorded
        at(current)

        if refused:
            with pytest.raises(ProgressMismatchError, match="git_commit differs"):
                self._restore(tmp_path)
        else:
            self._restore(tmp_path)

    def test_a_different_setup_only_warns(self, tmp_path, caplog) -> None:
        self._restore(tmp_path)

        with caplog.at_level(logging.WARNING):
            self._restore(tmp_path, _harness(num_instances=4))
        assert "num_instances: 1 -> 4" in caplog.text

    def test_a_changed_request_is_run_again(self, tmp_path, caplog) -> None:
        _first_attempt(tmp_path)

        with caplog.at_level(logging.WARNING):
            progress, _, restored = _open(tmp_path, prompt="changed q{i}")
        progress.close()
        assert restored.responses == {}
        assert len(restored.remaining) == 4
        assert "4 recorded instance(s) no longer match" in caplog.text

    @pytest.mark.parametrize("changed", ["gold_answer", "metadata"])
    def test_a_changed_gold_or_metadata_is_run_again(self, tmp_path, changed) -> None:
        _first_attempt(tmp_path)

        def edit(items: list[QueueItem]) -> None:
            edited = items[2]
            if changed == "gold_answer":
                edited.instance = replace(edited.instance, gold_answer="another answer")
            else:
                edited.instance.metadata["source"] = "a corrected document"

        progress, _, restored = _open(tmp_path, edit=edit)
        progress.close()

        assert sorted(restored.responses["local"]) == [0, 1, 3]
        assert [item.instance_idx for item in restored.remaining] == [2]

    def test_the_batching_strategy_is_not_part_of_what_must_match(self, tmp_path) -> None:
        from olmo_eval.runners.asynq.batching import BatchConfig

        _first_attempt(tmp_path)

        streaming = replace(_harness(), batching=BatchConfig.streaming())
        progress, _, restored = _open(tmp_path, streaming)
        progress.close()

        assert sorted(restored.responses["local"]) == [0, 1, 2, 3]

    def test_the_harness_hash_ignores_how_work_is_scheduled(self) -> None:
        from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode

        def config(sandbox: dict[str, Any] | None = None, **kwargs: Any) -> HarnessConfig:
            return HarnessConfig(
                name="default",
                provider=ProviderConfig(kind="mock", model="m"),
                sandboxes=(SandboxConfig(image="img", mode=SandboxMode.LOCAL, **(sandbox or {})),),
                **kwargs,
            )

        base = progress_module.harness_hash(config())
        scheduling = config(
            max_concurrency=7,
            scoring_concurrency=3,
            sandbox_pool_instances=4,
            sandbox_pool_min_instances=2,
            sandbox={"instances": 6, "min_instances": 3},
        )
        assert progress_module.harness_hash(scheduling) == base
        assert progress_module.harness_hash(config(system_prompt="Be brief.")) != base
        assert progress_module.harness_hash(config(tools=("browse_webpage",))) != base
        assert progress_module.harness_hash(config(sandbox={"image_pull": "always"})) != base


class TestRuntimeIdentity:
    """The workers' resolved SGLang runtime must be the same when a run continues."""

    runtime = {
        "profile": "qwen",
        "packages": {"sglang": "0.5"},
        "logprobs": "original",
        "arithmetic_environment": {"A": "1"},
        "engine_options": {"attention_backend": "fa3", "max_running_requests": 64},
        "effective_max_running_requests": 64,
        "max_total_num_tokens": 1000,
        "max_req_input_len": 999,
    }

    def _recorded(self, out: Path) -> RunProgress:
        """A run that recorded ``runtime``, restarted on the same directory."""
        progress, _, _ = _open(out)
        progress.check_runtime(self.runtime)
        progress.close()
        again, _, _ = _open(out)
        return again

    def test_what_depends_on_the_node_is_ignored(self, tmp_path) -> None:
        again = self._recorded(tmp_path)
        other_node = {
            **self.runtime,
            "engine_options": {"attention_backend": "fa3", "max_running_requests": 48},
            "effective_max_running_requests": 48,
        }
        again.check_runtime(other_node)

        new_engine = {**self.runtime, "packages": {"sglang": "0.6"}}
        with pytest.raises(ProgressMismatchError, match=r"packages.sglang: '0.5' -> '0.6'"):
            again.check_runtime(new_engine)
        backend = {**self.runtime, "engine_options": {"attention_backend": "triton"}}
        with pytest.raises(ProgressMismatchError, match="attention_backend"):
            again.check_runtime(backend)
        again.close()

    @pytest.mark.parametrize("cap", ["max_total_num_tokens", "max_req_input_len"])
    def test_each_request_length_cap_is_part_of_the_identity(self, tmp_path, cap) -> None:
        again = self._recorded(tmp_path)
        # The requested pool and memory fraction differ, but the caps they led to match.
        options = {
            **self.runtime["engine_options"],
            "max_total_tokens": 2000,
            "mem_fraction_static": 0.8,
        }
        again.check_runtime({**self.runtime, "engine_options": options})

        with pytest.raises(ProgressMismatchError, match=rf"{cap}: \d+ -> \d+"):
            again.check_runtime({**self.runtime, cap: self.runtime[cap] - 500})
        again.close()

    def test_unreported_caps_fall_back_to_the_requested_pool(self) -> None:
        runtime = {"engine_options": {"max_total_tokens": 1000, "max_running_requests": 8}}
        assert progress_module.runtime_identity(runtime) == {
            "engine_options": {"max_total_tokens": 1000}
        }

    def test_the_caps_reach_the_resolved_runtime(self) -> None:
        from olmo_eval.runners.asynq.batching import BatchConfig
        from olmo_eval.runners.asynq.runtime import resolved_worker_config

        config = HarnessConfig(name="default", provider=ProviderConfig(kind="sglang", model="m"))
        profile = {"selected": "generic", "effective_options": {}}

        capped = resolved_worker_config(
            config,
            BatchConfig(),
            profile={**profile, "max_total_num_tokens": 1000, "max_req_input_len": 999},
        )
        uncapped = resolved_worker_config(config, BatchConfig(), profile=profile)

        assert capped.provider.resolved_runtime["max_total_num_tokens"] == 1000
        assert capped.provider.resolved_runtime["max_req_input_len"] == 999
        assert "max_total_num_tokens" not in uncapped.provider.resolved_runtime
        assert "max_req_input_len" not in uncapped.provider.resolved_runtime


# --- Accounting, driven through process_results without worker processes ---


@dataclass(frozen=True)
class _PickyScorer(Scorer):
    """Scores 1.0, except for instances marked undecidable or unreachable."""

    name: str = "picky"

    def score(self, instance: Instance, output: LMOutput) -> float:
        from olmo_eval.common.scorers.execution import SandboxInfrastructureError

        if instance.metadata.get("undecidable"):
            raise RuntimeError("cannot decide")
        if instance.metadata.get("unreachable"):
            raise SandboxInfrastructureError("grader unreachable")
        return 1.0


class _LocalTask(Task):
    @property
    def instances(self) -> Iterator[Instance]:
        yield from ()

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question)

    def extract_answer(self, output: LMOutput) -> str | None:
        return output.text


def _local_setup(
    undecidable: set[int] | frozenset[int] = frozenset(),
    unreachable: frozenset[int] = frozenset(),
    prompt: str = "q{i}",
    request_type: RequestType = RequestType.COMPLETION,
) -> tuple[dict[str, TaskTracker], list[QueueItem]]:
    task = _LocalTask(
        TaskConfig(
            name="local",
            data_source="test/local",
            metrics=(AccuracyMetric(scorer=_PickyScorer),),
        )
    )
    items = [
        QueueItem(
            model_name="mock",
            task_id="local",
            instance_idx=i,
            instance=Instance(
                question=prompt.format(i=i),
                gold_answer="a",
                metadata={
                    "id": i,
                    "undecidable": i in undecidable,
                    "unreachable": i in unreachable,
                },
            ),
            request=LMRequest(request_type=request_type, prompt=prompt.format(i=i)),
        )
        for i in range(4)
    ]
    tracker = TaskTracker(model_name="mock", spec="local", task=task, total_instances=len(items))
    return {"local": tracker}, items


def _process(
    trackers: dict[str, TaskTracker],
    items: list[QueueItem],
    progress: RunProgress,
    restored: dict[str, Any],
    failures: dict[int, bool] | None = None,
) -> dict[str, Any]:
    """Feed one result per item to process_results.

    ``failures`` maps the idx of an item whose call failed to whether the request
    caused the failure.
    """
    failures = failures or {}
    results = []
    for item in items:
        failed = item.instance_idx in failures
        results.append(
            ResultItem(
                model_name="mock",
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=item.request,
                outputs=[] if failed else [LMOutput(text="a", metadata={"prompt_tokens": 3})],
                error="provider failed" if failed else None,
                request_caused=failures.get(item.instance_idx, False),
            )
        )
    return _process_results(trackers, results, progress, restored)


def _process_results(
    trackers: dict[str, TaskTracker],
    result_items: list[ResultItem],
    progress: RunProgress,
    restored: dict[str, Any],
) -> dict[str, Any]:
    results: queue.Queue[ResultItem] = queue.Queue()
    for result in result_items:
        results.put(result)
    return asyncio.run(
        process_results(
            trackers=trackers,
            result_queue=results,  # type: ignore[arg-type]
            workers=[],
            scoring_context=None,  # type: ignore[arg-type]
            scoring_concurrency=2,
            total_tasks=len(trackers),
            total_instances=len(result_items),
            model_name="mock",
            save_predictions=False,
            write_predictions_fn=None,
            save_requests=False,
            write_requests_fn=None,
            max_hard_failure_rate=1.0,
            restored=restored,
            progress=progress,
        )
    )


def _progress(out: Path) -> RunProgress:
    return open_run_progress(str(out), "mock", _harness())


def _drop_final(out: Path) -> None:
    for final in (out / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"):
        final.unlink()


def _open(
    out: Path,
    harness: HarnessConfig | None = None,
    edit: Any = None,
    **setup: Any,
) -> tuple[RunProgress, dict[str, TaskTracker], Restored]:
    """Open the run's progress and restore it for the local task (see ``_local_setup``).

    ``edit`` may change the prepared items first, as a dataset change would.
    """
    harness = harness or _harness()
    trackers, items = _local_setup(**setup)
    if edit is not None:
        edit(items)
    progress = open_run_progress(str(out), "mock", harness)
    return progress, trackers, progress.restore(trackers, items, harness)


def _first_attempt(
    out: Path, failures: dict[int, bool] | None = None, **setup: Any
) -> dict[str, Any]:
    """Run the local task once, recording progress, then leave only the records as a kill would.

    Returns the task results of this attempt.
    """
    progress, trackers, restored = _open(out, **setup)
    results = _process(trackers, restored.remaining, progress, restored.responses, failures)
    progress.close()
    _drop_final(out)
    return results


class TestAccounting:
    def test_only_outcomes_an_uninterrupted_run_keeps_are_recorded(self, tmp_path) -> None:
        # 0: a request-caused failure (final); 1: a failed call (retried); 2: undecidable
        # (final, unscored); 3: grader unreachable (retried).
        marked = {"undecidable": {2}, "unreachable": frozenset({3})}
        first = _first_attempt(tmp_path, {0: True, 1: False}, **marked)

        progress, trackers, restored = _open(tmp_path, **marked)

        assert sorted(restored.responses["local"]) == [2]
        assert trackers["local"].failed_instances == {0: "provider failed"}
        assert [item.instance_idx for item in restored.remaining] == [1, 3]
        second = _process(trackers, restored.remaining, progress, restored.responses, {1: False})
        progress.close()
        assert second["local"].metrics == first["local"].metrics
        assert second["local"].instances_failed == first["local"].instances_failed == 2
        assert second["local"].scoring_failures == first["local"].scoring_failures

    def test_provider_errors_reach_the_records_as_the_runner_sees_them(self, tmp_path) -> None:
        """Failures take the path they take at run time, through process_batch.

        API providers turn every exception into a marked output, so a connection
        error and a context overflow both arrive as outputs. Only the overflow is the
        request's fault; the connection error must run again.
        """
        from olmo_eval.inference.errors import CONTEXT_OVERFLOW_KEY, REQUEST_ERROR_KEY
        from olmo_eval.inference.utils import dispatch_requests
        from olmo_eval.runners.asynq.processing import process_batch

        class Provider:
            def describe_request(self, request: LMRequest, params: Any) -> None:
                return None

            async def agenerate(self, requests: list[LMRequest], params: Any) -> Any:
                async def call(request: LMRequest) -> list[LMOutput]:
                    if request.prompt == "overflow-text":
                        raise ValueError("maximum context length is 8 tokens, got 90")
                    if request.prompt == "reset":
                        raise ConnectionError("connection reset by peer")
                    if request.prompt == "overflow-flag":
                        return [
                            LMOutput(
                                text="",
                                metadata={
                                    REQUEST_ERROR_KEY: "too long",
                                    CONTEXT_OVERFLOW_KEY: True,
                                },
                            )
                        ]
                    return [LMOutput(text="a", metadata={"prompt_tokens": 3})]

                return await dispatch_requests(requests, call)

        class Harness:
            provider = Provider()

            def _apply_config(self, request: LMRequest) -> LMRequest:
                return request

            def flush_metrics(self, batch_hash: str) -> None:
                return None

        prompts = ["overflow-text", "reset", "overflow-flag", "fine"]

        def setup() -> tuple[dict[str, TaskTracker], list[QueueItem]]:
            trackers, items = _local_setup(undecidable=set())
            for item, prompt in zip(items, prompts, strict=True):
                item.request = replace(item.request, prompt=prompt)
            return trackers, items

        trackers, items = setup()
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        sent: queue.Queue[ResultItem] = queue.Queue()
        asyncio.run(process_batch(restored.remaining, Harness(), sent))  # type: ignore[arg-type]
        result_items = [sent.get_nowait() for _ in range(sent.qsize())]
        by_idx = {result.instance_idx: result for result in result_items}
        assert [by_idx[i].request_caused for i in range(4)] == [True, False, True, False]
        assert by_idx[1].error is not None and "ConnectionError" in by_idx[1].error
        _process_results(trackers, result_items, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)

        trackers, items = setup()
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        progress.close()

        assert sorted(trackers["local"].failed_instances) == [0, 2]
        assert sorted(restored.responses["local"]) == [3]
        assert [item.instance_idx for item in restored.remaining] == [1]

    def test_a_scorers_metadata_writes_survive_a_restore(self, tmp_path) -> None:
        """Instance metadata a scorer sets is what the task's metric reads."""
        from olmo_eval.common.metrics.base import Metric

        @dataclass(frozen=True)
        class Stamping(Scorer):
            name: str = "stamping"

            def score(self, instance: Instance, output: LMOutput) -> float:
                instance.metadata["stamp"] = len(output.text) + int(instance.metadata["id"])
                instance.metadata.pop("draft", None)
                return 1.0

        @dataclass(frozen=True, slots=True)
        class StampSum(Metric):
            name: str = "stamp_sum"
            scorer: type[Scorer] | Scorer = Stamping

            def compute(self, responses: Any) -> float:
                return float(
                    sum(r.instance.metadata["stamp"] for r in responses)
                    + sum("draft" in r.instance.metadata for r in responses) * 1000
                )

        def setup() -> tuple[dict[str, TaskTracker], list[QueueItem]]:
            trackers, items = _local_setup(undecidable=set())
            trackers["local"].task = _LocalTask(
                TaskConfig(name="local", data_source="test/local", metrics=(StampSum(),))
            )
            for item in items:
                item.instance.metadata["draft"] = True
            return trackers, items

        def feed(trackers: Any, items: Any, progress: Any, restored: Any) -> dict[str, Any]:
            # As over the queue, a result carries its own copy of the instance.
            results = [
                ResultItem(
                    model_name="mock",
                    task_id="local",
                    instance_idx=item.instance_idx,
                    instance=replace(item.instance, metadata=dict(item.instance.metadata)),
                    request=item.request,
                    outputs=[LMOutput(text="abc", metadata={"prompt_tokens": 3})],
                )
                for item in items
            ]
            return _process_results(trackers, results, progress, restored)

        trackers, items = setup()
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        first = feed(trackers, restored.remaining, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)
        expected = first["local"].metrics
        assert expected["stamp_sum"][next(iter(expected["stamp_sum"]))] == 18.0

        trackers, items = setup()
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        assert restored.remaining == []
        second = feed(trackers, [], progress, restored.responses)
        assert second["local"].metrics == expected

    def test_a_task_recorded_in_full_completes_with_nothing_queued(self, tmp_path) -> None:
        first = _first_attempt(tmp_path)

        progress, trackers, restored = _open(tmp_path)
        assert restored.remaining == []
        second = _process(trackers, [], progress, restored.responses)
        progress.close()

        assert second["local"].metrics == first["local"].metrics
        assert second["local"].predictions == first["local"].predictions
        assert trackers["local"].prompt_tokens_total == 12

    def test_a_restored_instance_that_arrives_again_is_rejected(self, tmp_path) -> None:
        _first_attempt(tmp_path)

        progress, trackers, restored = _open(tmp_path)
        again = _local_setup()[1][:1]
        with pytest.raises(RuntimeError, match="accounting mismatch"):
            _process(trackers, again, progress, restored.responses)
        progress.close()


def test_request_times_and_worker_seconds_are_restored(tmp_path) -> None:
    progress, trackers, restored = _open(tmp_path)
    results = [
        ResultItem(
            model_name="mock",
            task_id="local",
            instance_idx=item.instance_idx,
            instance=item.instance,
            request=item.request,
            outputs=[LMOutput(text="a", metadata={"prompt_tokens": 3})],
            sent_at=1000.0 + 10 * (3 - item.instance_idx),
            attributed_seconds=2.0,
        )
        for item in restored.remaining
    ]
    _process_results(trackers, results, progress, restored.responses)
    progress.close()
    _drop_final(tmp_path)
    assert trackers["local"].first_request_at == 1000.0

    progress, trackers, restored = _open(tmp_path)
    progress.close()

    assert restored.remaining == []
    # The earliest request of the first attempt, as if it had not been interrupted.
    assert trackers["local"].first_request_at == 1000.0
    assert trackers["local"].attributed_inference_seconds == 8.0


def _scored(item: QueueItem, failed: bool = False) -> tuple[Response, ResultItem]:
    """The scored response of one local-task item, and the result it came from."""
    result = ResultItem(
        model_name="mock",
        task_id=item.task_id,
        instance_idx=item.instance_idx,
        instance=item.instance,
        request=item.request,
        outputs=[] if failed else [LMOutput(text="a")],
        error="too long" if failed else None,
        request_caused=failed,
    )
    response = Response(
        instance=item.instance,
        request=item.request,
        outputs=result.outputs,
        scores={"picky": 1.0},
    )
    return response, result


class TestShutdown:
    @pytest.fixture(autouse=True)
    def _clear_shutdown(self) -> Iterator[None]:
        clear_shutdown()
        yield
        clear_shutdown()

    def test_nothing_is_recorded_after_shutdown_was_requested(self, tmp_path) -> None:
        progress, trackers, restored = _open(tmp_path)
        request_shutdown()
        try:
            with pytest.raises(Terminated):
                _process(trackers, restored.remaining, progress, restored.responses, {0: False})
        finally:
            clear_shutdown()
            progress.close()

        progress, trackers, restored = _open(tmp_path)
        progress.close()
        assert restored.responses == {} and trackers["local"].failed_instances == {}
        assert not list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))

    def test_a_result_that_arrives_after_shutdown_is_not_recorded(self, tmp_path) -> None:
        progress, _, restored = _open(tmp_path)
        before, scored_after, failed_after = restored.remaining[:3]

        progress.record("local", before.instance_idx, *_scored(before))
        request_shutdown()
        progress.record("local", scored_after.instance_idx, *_scored(scored_after))
        progress.record_failure("local", failed_after.instance_idx, _scored(failed_after, True)[1])
        progress.close()

        assert [e["idx"] for e in _instance_entries(tmp_path)] == [before.instance_idx]

    def test_a_task_finished_after_shutdown_is_not_finalized(self, tmp_path) -> None:
        from olmo_eval.runners.common.types import TaskResult

        progress, trackers, _ = _open(tmp_path)
        config = trackers["local"].task.config.to_dict()  # type: ignore[union-attr]
        request_shutdown()
        progress.finalize("local", TaskResult("local", config, 4, {}))
        progress.close()

        assert not list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))

    def test_the_sigterm_handler_requests_shutdown_and_raises(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            pytest.skip("signal handlers are only installed from the main thread")
        previous = signal.getsignal(signal.SIGTERM)

        with _sigterm_raises():
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler) and handler is not previous
            with pytest.raises(Terminated):
                handler(signal.SIGTERM, None)
            assert shutdown_requested()

        # A later run in this process starts with no request pending.
        assert not shutdown_requested()
        assert signal.getsignal(signal.SIGTERM) is previous

    def test_leaving_the_sigterm_context_clears_a_request_made_inside_it(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            pytest.skip("signal handlers are only installed from the main thread")

        with _sigterm_raises():
            request_shutdown()
        assert not shutdown_requested()

        with pytest.raises(RuntimeError, match="boom"), _sigterm_raises():
            request_shutdown()
            raise RuntimeError("boom")
        assert not shutdown_requested()


class TestFailedTasks:
    def test_a_failed_task_is_recomputed_not_finalized(self, tmp_path) -> None:
        # Every instance fails from its request, so nothing is scored and the task fails.
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        failures = {i: True for i in range(4)}
        first = _process(trackers, restored.remaining, progress, restored.responses, failures)
        progress.close()
        assert first["local"].error
        assert not list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))

        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        # The failures the request caused are restored; the task is not.
        assert restored.finalized == {}
        assert sorted(trackers["local"].failed_instances) == [0, 1, 2, 3]

    def test_a_task_with_an_error_is_not_finalized(self, tmp_path) -> None:
        from olmo_eval.runners.common.types import TaskResult

        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        progress.restore(trackers, items, _harness())
        config = trackers["local"].task.config.to_dict()  # type: ignore[union-attr]
        progress.finalize("local", TaskResult("local", config, 0, {}, error="boom"))
        progress.finalize(
            "local",
            TaskResult("local", config, 0, {}, failure_budget_exceeded=True),
        )
        assert not list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))
        progress.finalize("local", TaskResult("local", config, 0, {}))
        assert len(list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))) == 1
        progress.close()


def _run_id(out: Path) -> str:
    manifest = read_manifest(out)
    assert manifest is not None
    return manifest["run"]["run_id"]


class TestRunIdentity:
    def test_a_resumed_run_keeps_its_id_and_counts_its_restarts(
        self, interrupted_run, tmp_path
    ) -> None:
        out = _resume_copy(interrupted_run, tmp_path)

        run_eval(out, INTERRUPTED_LIMITS, env={"BEAKER_JOB_ID": "job-2"})

        manifest = read_manifest(out)
        assert manifest is not None
        assert manifest["run"]["run_id"] == _run_id(interrupted_run)
        assert manifest["run"]["status"] == "complete"
        assert manifest["resume"] == {"resumed_count": 1, "beaker_job_ids": ["job-1", "job-2"]}

    @pytest.mark.parametrize(
        "case", ["another model", "other task specs", "without resume", "fresh", "discarded"]
    )
    def test_a_run_that_cannot_continue_the_records_gets_a_new_id(
        self, interrupted_run, tmp_path, case
    ) -> None:
        out = _resume_copy(interrupted_run, tmp_path)
        limits = {"gsm8k": 2}
        kwargs: dict[str, Any] = {}
        if case == "another model":
            kwargs["model"] = "another-model"
        elif case == "other task specs":
            limits = {"ifeval": 4}
        elif case == "without resume":
            kwargs["resume"] = False
        elif case == "fresh":
            kwargs.update(fresh=True, resume=False)
        else:
            discard_progress(out, "mock", ["gsm8k"])

        run_eval(out, limits, **kwargs)

        manifest = read_manifest(out)
        assert manifest is not None
        assert manifest["run"]["run_id"] != _run_id(interrupted_run)
        assert manifest["resume"]["resumed_count"] == 0

    def test_a_completed_run_is_not_continued(self, interrupted_run, tmp_path) -> None:
        # Below run_eval: a completed run deletes its progress, so no run can reach a
        # completed manifest that still has records beside it.
        out = _resume_copy(interrupted_run, tmp_path)
        path = out / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["run"]["status"] = "complete"
        path.write_text(json.dumps(manifest))

        runner = _runner(out, INTERRUPTED_LIMITS, resume=True)

        assert runner._previous_run()[0] is None


class TestProgressIsNotUploaded:
    def test_the_uploader_skips_the_runners_progress_directory(self) -> None:
        from olmo_eval.upload.uploader import _UNUPLOADED_DIRS

        assert PROGRESS_DIR in _UNUPLOADED_DIRS

    def test_collect_artifacts_skips_progress(self, tmp_path) -> None:
        from olmo_eval.upload.uploader import collect_artifacts

        (tmp_path / "metrics.json").write_text("{}")
        (tmp_path / "predictions" / "m").mkdir(parents=True)
        (tmp_path / "predictions" / "m" / "t.jsonl").write_text("{}\n")
        for name in ("run.segments/000000.jsonl", "m/t_abcdef.final.json"):
            path = tmp_path / PROGRESS_DIR / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")

        artifacts, files = collect_artifacts(tmp_path, {}, [])

        assert sorted(files) == ["metrics.json", "predictions/m/t.jsonl"]
        assert not any(a["path"].startswith(PROGRESS_DIR) for a in artifacts)


class TestProgressLifetime:
    """When a run deletes the records it wrote."""

    def _upload_run(
        self, out: Path, upload_error: Exception | None = None, **kwargs: Any
    ) -> list[bool]:
        """Run with uploads on; whether the records were still there when the upload ran."""
        from unittest.mock import patch

        from olmo_eval.upload import UploadConfig

        seen: list[bool] = []

        def upload(output_dir: str, config: Any) -> None:
            assert (Path(output_dir) / "metrics.json").exists()
            seen.append(has_progress(output_dir))
            if upload_error is not None:
                raise upload_error

        with (
            patch("olmo_eval.upload.upload_results_dir", side_effect=upload),
            patch("olmo_eval.upload.register_run_start"),
        ):
            run_eval(
                out,
                INTERRUPTED_LIMITS,
                upload_config=UploadConfig(enabled=True, api_url="https://ingest.test"),
                **kwargs,
            )
        return seen

    def test_progress_is_deleted_after_a_completed_run_uploads(self, tmp_path) -> None:
        # The records outlive the upload, so a preemption during it still resumes.
        assert self._upload_run(tmp_path) == [True]
        assert not has_progress(tmp_path)

    def test_a_failed_upload_keeps_progress_for_the_requeue(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="preempted"):
            self._upload_run(tmp_path, RuntimeError("preempted"))

        assert has_progress(tmp_path)

    def test_a_completed_run_deletes_only_its_own_records(self, interrupted_run, tmp_path) -> None:
        out = _resume_copy(interrupted_run, tmp_path)
        other = out / PROGRESS_DIR / "other-model"
        other.mkdir(parents=True)
        (other / "run.segments").write_text("{}\n")
        (other / f"gsm8k_abcdef{FINAL_SUFFIX}").write_text("{}")

        run_eval(out, INTERRUPTED_LIMITS)

        assert not stream_dir(out, "mock").parent.exists()
        assert sorted(p.name for p in other.iterdir()) == [
            f"gsm8k_abcdef{FINAL_SUFFIX}",
            "run.segments",
        ]

    def test_a_run_without_resume_leaves_stale_progress_and_warns(
        self, interrupted_run, tmp_path, runner_log
    ) -> None:
        out = _resume_copy(interrupted_run, tmp_path)
        before = sorted(str(p.relative_to(out)) for p in (out / PROGRESS_DIR).rglob("*"))

        with runner_log.at_level(logging.WARNING):
            run_eval(out, INTERRUPTED_LIMITS, resume=False)

        assert "It is stale: this run did not use it" in runner_log.text
        assert sorted(str(p.relative_to(out)) for p in (out / PROGRESS_DIR).rglob("*")) == before

    def test_a_run_that_trips_the_gate_keeps_progress_for_its_retry(self, tmp_path) -> None:
        # Below run_eval: no fake provider failure rate trips the budget at a chosen task.
        from olmo_eval.runners.asynq.results import FailureBudgetExceeded

        progress, trackers, restored = _open(tmp_path)
        results = _process(trackers, restored.remaining, progress, restored.responses)
        results["local"].failure_budget_exceeded = True
        runner = _runner(tmp_path, {"gsm8k": 3}, resume=True)
        runner._start_run_record = lambda experiment_id: None  # type: ignore[method-assign]
        results_dict = {"model": "mock", "model_config": {}, "tasks": {}, "errors": []}

        with pytest.raises(FailureBudgetExceeded):
            runner._finalize_and_gate(
                results, results_dict, experiment_id="abc123def456", progress=progress
            )
        progress.close()

        assert has_progress(tmp_path)

    def test_discard_removes_emptied_directories(self, tmp_path) -> None:
        progress, trackers, restored = _open(tmp_path)
        _process(trackers, restored.remaining, progress, restored.responses)
        assert has_progress(tmp_path)

        progress.discard()

        assert not (tmp_path / PROGRESS_DIR).exists()


def test_the_resume_flags_reach_the_runner(tmp_path) -> None:
    from click.testing import CliRunner

    from olmo_eval.cli import main
    from olmo_eval.cli.run.config import RunConfigBuilder
    from olmo_eval.cli.run.factory import RunnerFactory
    from olmo_eval.upload import UploadConfig

    help_text = CliRunner().invoke(main, ["run", "--help"]).output
    assert "--resume" in help_text and "--fresh" in help_text
    both = CliRunner().invoke(
        main, ["run", "-m", "mock", "-t", "gsm8k", "--resume", "--fresh", "--no-upload"]
    )
    assert both.exit_code != 0 and "cannot be used together" in both.output
    for flag in ("resume", "fresh"):
        config = RunConfigBuilder(
            model="mock", task=("gsm8k",), output_dir=str(tmp_path), **{flag: True}
        )
        runner = RunnerFactory(config.build(), UploadConfig(enabled=False)).create()
        assert isinstance(runner, AsyncEvalRunner) and getattr(runner, flag)


# --- Compact records ---


def _long_generation(item: QueueItem, samples: int, tokens: int) -> ResultItem:
    """A result with ``samples`` generations of ``tokens`` tokens, each with its logprob.

    Outputs carry per-token logprobs and their summaries, as vllm_server returns them
    when the generation kwargs ask for logprobs.
    """
    outputs = []
    for sample in range(samples):
        logprobs = [
            {"token": f" w{(sample + i) % 997}", "logprob": -((sample * 31 + i) % 4000) / 1000.0}
            for i in range(tokens)
        ]
        outputs.append(
            LMOutput(
                text="".join(entry["token"] for entry in logprobs),
                logprobs=logprobs,  # type: ignore[arg-type]
                metadata={
                    "finish_reason": "length",
                    "prompt_tokens": 3,
                    "completion_tokens": tokens,
                    "sum_logits": sum(entry["logprob"] for entry in logprobs),
                    "num_tokens": tokens,
                    "num_tokens_all": tokens,
                },
            )
        )
    return ResultItem(
        model_name="mock",
        task_id=item.task_id,
        instance_idx=item.instance_idx,
        instance=item.instance,
        request=item.request,
        outputs=outputs,
    )


def _record_and_restore(
    tmp_path: Path,
    results: Any,
    *metrics: Any,
    request_type: RequestType = RequestType.COMPLETION,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Record a run of the local task, then restore it in full and finish it again.

    ``results`` maps each queue item to its result. Returns both runs' task results
    and the recorded instance entries.
    """

    def setup() -> tuple[dict[str, TaskTracker], list[QueueItem]]:
        trackers, items = _local_setup(request_type=request_type)
        if metrics:
            trackers["local"].task = _LocalTask(
                TaskConfig(name="local", data_source="test/local", metrics=metrics)
            )
        return trackers, items

    trackers, items = setup()
    progress = _progress(tmp_path)
    restored = progress.restore(trackers, items, _harness())
    first = _process_results(
        trackers, [results(item) for item in restored.remaining], progress, restored.responses
    )
    progress.close()
    _drop_final(tmp_path)
    recorded = _instance_entries(tmp_path)

    trackers, items = setup()
    progress = _progress(tmp_path)
    restored = progress.restore(trackers, items, _harness())
    assert restored.remaining == []
    second = _process_results(trackers, [], progress, restored.responses)
    progress.close()
    return first, second, recorded


class TestCompactRecords:
    def test_generations_are_recorded_without_per_token_logprobs(self, full_run) -> None:
        full = full_run
        entries = _entries(full)
        spec_of = {e["t"]: e["header"]["spec"] for e in entries if "header" in e}
        outputs: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            for output in entry.get("outputs", ()):
                outputs.setdefault(spec_of[entry["t"]], []).append(output)

        # Log-likelihood outputs keep theirs: the multiple-choice metrics read them.
        assert all("logprobs" in output for output in outputs.pop("arc_challenge:mc"))
        # The fake returns per-token logprobs with every generation, as vllm_server does.
        generations = [output for task in outputs.values() for output in task]
        assert generations
        assert all("logprobs" not in o and o["logprob_count"] > 0 for o in generations)
        # The summaries the predictions report come from the metadata the records keep.
        predictions = b"".join(
            content for name, content in _artifacts(full).items() if "/gsm8k_" in name
        )
        assert b'"sum_logits"' in predictions and b'"bits_per_byte"' in predictions

    def test_a_long_generation_is_recorded_without_its_token_logprobs(self, tmp_path) -> None:
        samples, tokens = 32, 2000
        first, second, recorded = _record_and_restore(
            tmp_path, lambda item: _long_generation(item, samples, tokens)
        )

        assert len(recorded) == 4
        for entry in recorded:
            assert [output["logprob_count"] for output in entry["outputs"]] == [tokens] * samples
            assert not any("logprobs" in output for output in entry["outputs"])
        # The same entry with every token's logprob, as records held it before.
        with_tokens = {
            **recorded[0],
            "outputs": [
                {**output, "logprobs": [{"token": " w1", "logprob": -1.234}] * tokens}
                for output in recorded[0]["outputs"]
            ],
        }
        assert len(json.dumps(recorded[0])) * 4 < len(json.dumps(with_tokens))
        # A restore writes byte-identical predictions and the same metrics.
        assert second["local"].metrics == first["local"].metrics
        assert json.dumps(second["local"].predictions) == json.dumps(first["local"].predictions)
        assert '"sum_logits"' in json.dumps(first["local"].predictions)

    def test_a_metric_that_reads_logprobs_keeps_them(self, tmp_path) -> None:
        from olmo_eval.common.metrics import BPBMetricInstanceAvg

        first, second, recorded = _record_and_restore(
            tmp_path, lambda item: _long_generation(item, 2, 5), BPBMetricInstanceAvg()
        )

        assert all("logprobs" in output for e in recorded for output in e["outputs"])
        assert first["local"].metrics["bits_per_byte"]
        assert second["local"].metrics == first["local"].metrics
        assert json.dumps(second["local"].predictions) == json.dumps(first["local"].predictions)

    def test_loglikelihood_outputs_keep_their_logprobs_for_any_metric(self, tmp_path) -> None:
        """No scorer is marked as reading logprobs, so only the request type keeps them."""
        from olmo_eval.common.metrics.base import Metric

        @dataclass(frozen=True, slots=True)
        class FirstChoiceAccuracy(Metric):
            """Multiple-choice accuracy from the logprobs of each choice, gold index 0."""

            name: str = "first_choice_accuracy"
            scorer: type[Scorer] | Scorer = _PickyScorer

            def compute(self, responses: Any) -> float:
                correct = 0
                for response in responses:
                    totals = [sum(t["logprob"] for t in o.logprobs) for o in response.outputs]
                    correct += totals.index(max(totals)) == 0
                return correct / len(responses)

        def choices(item: QueueItem) -> ResultItem:
            outputs = []
            for choice, logprob in enumerate((-0.5, -2.0 - item.instance_idx, -3.0)):
                outputs.append(
                    LMOutput(
                        text=f"choice {choice}",
                        logprobs=[{"token": f"c{choice}", "logprob": logprob}],
                        # As every provider that returns logprobs sets them, so the
                        # records could do without the logprobs if they were generated.
                        metadata={"sum_logits": logprob, "num_tokens": 1, "num_tokens_all": 1},
                    )
                )
            return ResultItem(
                model_name="mock",
                task_id="local",
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=item.request,
                outputs=outputs,
            )

        first, second, recorded = _record_and_restore(
            tmp_path, choices, FirstChoiceAccuracy(), request_type=RequestType.LOGLIKELIHOOD
        )

        assert len(recorded) == 4
        for entry in recorded:
            assert len(entry["outputs"]) == 3
            assert all(
                output["logprobs"] and "logprob_count" not in output for output in entry["outputs"]
            )
        assert first["local"].metrics["first_choice_accuracy"]
        assert second["local"].metrics == first["local"].metrics

    def test_reading_left_out_logprobs_fails_instead_of_scoring(self, tmp_path) -> None:
        """A consumer the records do not know about cannot compute from missing tokens."""
        _record_and_restore(tmp_path, lambda item: _long_generation(item, 1, 7))
        _drop_final(tmp_path)
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        progress.close()

        output = restored.responses["local"][0].outputs[0]
        assert output.logprobs is not None and len(output.logprobs) == 7 and output.logprobs
        with pytest.raises(DroppedLogprobsError, match="7 per-token logprobs"):
            sum(entry["logprob"] for entry in output.logprobs)

    def test_a_record_in_an_older_format_is_refused(self, tmp_path) -> None:
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        progress.restore(trackers, items, _harness())
        progress.close()
        store = _stream(tmp_path)
        entries = list(store.read())
        store.reset()
        for entry in entries:
            if "header" in entry:
                entry["header"]["version"] = RECORD_VERSION - 1
            store.append(entry)
        store.close()

        trackers, items = _local_setup(undecidable=set())
        with pytest.raises(
            ProgressMismatchError, match=rf"record format {RECORD_VERSION - 1}.*--fresh"
        ):
            _progress(tmp_path).restore(trackers, items, _harness())


# --- Results return per request while progress is recorded ---


class _SlowProvider:
    """Answers each request with its prompt; the request "slow" waits to be released.

    ``fast_answered`` is set once every request other than "slow" and "dead" has been
    answered; ``in_flight`` and ``peak`` count the calls running at once.
    """

    max_concurrency: int | None = 8

    def __init__(self, fast: int = 0) -> None:
        self.release = asyncio.Event()
        self.fast_answered = asyncio.Event()
        self.calls: list[list[str]] = []
        self._fast_left = fast
        self.in_flight = 0
        self.peak = 0

    def describe_request(self, request: LMRequest, params: Any) -> None:
        return None

    async def agenerate(self, requests: list[LMRequest], params: Any) -> list[list[LMOutput]]:
        from olmo_eval.inference.errors import TerminalProviderError

        self.calls.append([str(request.prompt) for request in requests])

        async def one(request: LMRequest) -> list[LMOutput]:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                if request.prompt == "slow":
                    await self.release.wait()
                if request.prompt == "dead":
                    raise TerminalProviderError("engine died")
                await asyncio.sleep(0)  # let other requests start
                if request.prompt != "slow":
                    self._fast_left -= 1
                    if self._fast_left == 0:
                        self.fast_answered.set()
                return [LMOutput(text=str(request.prompt))]
            finally:
                self.in_flight -= 1

        return list(await asyncio.gather(*(one(request) for request in requests)))


class _FakeHarness:
    def __init__(self, provider: _SlowProvider) -> None:
        self.provider = provider
        self.config = HarnessConfig(name="default", provider=ProviderConfig(kind="mock"))
        self.flushed: list[str] = []

    def _apply_config(self, request: LMRequest) -> LMRequest:
        return request

    def flush_metrics(self, batch_hash: str) -> None:
        self.flushed.append(batch_hash)


def _queued(prompts: list[str]) -> queue.Queue[QueueItem | None]:
    items: queue.Queue[QueueItem | None] = queue.Queue()
    for idx, prompt in enumerate(prompts):
        items.put(
            QueueItem(
                model_name="mock",
                task_id="local",
                instance_idx=idx,
                instance=Instance(question=prompt),
                request=LMRequest(request_type=RequestType.COMPLETION, prompt=prompt),
            )
        )
    items.put(None)
    return items


_WAIT_SECONDS = 60  # far longer than any run takes; it only bounds a hang


async def _run_batched(
    prompts: list[str], per_request: bool
) -> tuple[list[str], _SlowProvider, list[ResultItem]]:
    """Run one batch through the batched strategy; return what arrived before "slow" ended.

    Waits for the provider to answer every other request, not for a time to pass.
    """
    from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy

    fast = sum(prompt not in ("slow", "dead") for prompt in prompts)
    provider = _SlowProvider(fast)
    harness = _FakeHarness(provider)
    provider.harness = harness  # type: ignore[attr-defined]
    results: queue.Queue[ResultItem] = queue.Queue()
    strategy = get_strategy(
        BatchConfig(chunk_size=len(prompts), chunk_timeout=0.05), per_request=per_request
    )
    run = asyncio.create_task(
        strategy.run(
            _queued(prompts),  # type: ignore[arg-type]
            harness,  # type: ignore[arg-type]
            results,  # type: ignore[arg-type]
            None,
            logging.getLogger(__name__),
            len(prompts),
        )
    )
    answered = asyncio.create_task(provider.fast_answered.wait())
    try:
        await asyncio.wait(
            {run, answered}, timeout=_WAIT_SECONDS, return_when=asyncio.FIRST_COMPLETED
        )
        if not run.done():
            # Per-request results are queued as each call returns; a whole batch's, never
            # before the batch ends, so give those every chance to appear.
            for _ in range(200 if per_request else 50):
                if per_request and results.qsize() >= fast:
                    break
                await asyncio.sleep(0.001)
        early = [str(results.get_nowait().request.prompt) for _ in range(results.qsize())]
        provider.release.set()
        await run
    finally:
        answered.cancel()
        rest = [results.get_nowait() for _ in range(results.qsize())]
    return early, provider, rest


class TestPerRequestResults:
    def test_a_slow_request_does_not_hold_back_its_batch(self) -> None:
        early, provider, rest = asyncio.run(_run_batched(["slow", "a", "b", "c"], True))

        # Each finished request reached the result queue while "slow" was still running.
        assert sorted(early) == ["a", "b", "c"]
        assert [result.request.prompt for result in rest] == ["slow"]
        assert sorted(provider.calls) == [["a"], ["b"], ["c"], ["slow"]]
        assert all(result.error is None for result in rest)

    def test_without_progress_a_batch_returns_whole(self) -> None:
        early, provider, rest = asyncio.run(_run_batched(["slow", "a", "b", "c"], False))

        assert early == []
        assert provider.calls == [["slow", "a", "b", "c"]]
        assert sorted(str(result.request.prompt) for result in rest) == ["a", "b", "c", "slow"]

    @pytest.mark.parametrize("per_request", [True, False])
    def test_a_batch_flushes_one_metrics_record(self, per_request) -> None:
        _, provider, _ = asyncio.run(_run_batched(["slow", "a", "b", "c"], per_request))

        assert len(provider.harness.flushed) == 1  # type: ignore[attr-defined]

    def test_per_request_and_batched_flush_the_same_batch_identity(self) -> None:
        flushed = []
        for per_request in (True, False):
            _, provider, _ = asyncio.run(_run_batched(["slow", "a", "b", "c"], per_request))
            flushed.append(provider.harness.flushed)  # type: ignore[attr-defined]

        assert flushed[0] == flushed[1]

    @pytest.mark.parametrize("per_request", [True, False])
    def test_both_modes_credit_the_same_worker_time(self, per_request) -> None:
        """A batch's results share the wall time the batch had a request in flight."""
        from olmo_eval.runners.asynq.processing import process_batch, process_batch_per_request
        from olmo_eval.runners.asynq.share_clock import ShareClock

        now = [0.0]
        steps = {"a": 1.0, "b": 2.0, "c": 4.0}

        class Provider(_SlowProvider):
            async def agenerate(self, requests, params):
                await asyncio.sleep(0)
                outputs = []
                for request in requests:
                    now[0] += steps[str(request.prompt)]
                    outputs.append([LMOutput(text=str(request.prompt))])
                return outputs

        results: queue.Queue[ResultItem] = queue.Queue()
        items = list(_queued(["a", "b", "c"]).queue)[:-1]
        run = process_batch_per_request if per_request else process_batch
        asyncio.run(
            run(items, _FakeHarness(Provider()), results, clock=ShareClock(lambda: now[0]))  # type: ignore[arg-type]
        )

        shares = [results.get_nowait().attributed_seconds for _ in range(results.qsize())]
        assert len(shares) == 3
        assert sum(shares) == pytest.approx(7.0)

    @pytest.mark.parametrize(
        ("provider_limit", "argument_limit", "expected_peak"),
        [(2, None, 2), (2, 5, 2), (None, 3, 3), (None, None, 6)],
    )
    def test_requests_in_flight_stay_within_the_providers_concurrency(
        self, provider_limit, argument_limit, expected_peak
    ) -> None:
        """The provider's own limit applies, as it would to one batched call."""
        from olmo_eval.runners.asynq.processing import process_batch_per_request

        class Provider(_SlowProvider):
            max_concurrency = provider_limit

            async def agenerate(self, requests, params):
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                for _ in range(5):  # stay in flight while the others get their turn
                    await asyncio.sleep(0)
                self.in_flight -= 1
                return [[LMOutput(text=str(request.prompt))] for request in requests]

        provider = Provider()
        results: queue.Queue[ResultItem] = queue.Queue()
        items = list(_queued([f"r{i}" for i in range(6)]).queue)[:-1]

        asyncio.run(
            process_batch_per_request(
                items,
                _FakeHarness(provider),  # type: ignore[arg-type]
                results,  # type: ignore[arg-type]
                max_concurrency=argument_limit,
            )
        )

        assert provider.peak == expected_peak
        assert results.qsize() == 6

    def test_a_terminal_error_still_stops_the_worker(self) -> None:
        from olmo_eval.inference.errors import TerminalProviderError

        with pytest.raises(TerminalProviderError, match="engine died"):
            asyncio.run(_run_batched(["slow", "dead", "a"], True))

    def test_the_engines_are_the_providers_that_keep_whole_batches(self) -> None:
        from olmo_eval.common.types import ProviderKind
        from olmo_eval.runners.asynq.batching import BatchConfig

        kinds = {str(kind) for kind in ProviderKind}
        # Spelled out: in-process engines batch inside the engine, so splitting their
        # batches per request would change what they compute.
        engines = {"vllm", "sglang", "hf", "olmo_core", "olmo_core_vlm"}
        assert engines == BatchConfig._SEQUENTIAL_ONLY
        for kind in engines:
            assert not BatchConfig.dispatches_per_request(kind)
        for kind in sorted(kinds):
            assert BatchConfig.dispatches_per_request(kind) is (
                kind not in BatchConfig._SEQUENTIAL_ONLY
            )
        assert {"vllm_server", "litellm", "mock"} <= {
            kind for kind in kinds if BatchConfig.dispatches_per_request(kind)
        }

    @pytest.mark.parametrize("per_request", [True, False])
    @pytest.mark.parametrize("kind", sorted(str(kind) for kind in ProviderKind))
    def test_the_worker_splits_batches_only_for_providers_that_dispatch_one_by_one(
        self, monkeypatch, kind, per_request
    ) -> None:
        from olmo_eval.runners.asynq import batching
        from olmo_eval.runners.asynq.batching import BatchConfig

        chosen: list[bool] = []

        def get_strategy(*args: Any, **kwargs: Any) -> Any:
            chosen.append(kwargs["per_request"])
            return SimpleNamespace(run=AsyncMock())

        class Provider:
            # What the worker reads from an SGLang provider.
            profile_report = {"selected": "generic", "effective_options": {}}
            recommended_batch_sequences = None

        def make_harness(config: HarnessConfig) -> Any:
            return SimpleNamespace(
                provider=Provider(), initialize_reporters=Mock(), cleanup=AsyncMock()
            )

        monkeypatch.setattr(batching, "get_strategy", get_strategy)
        monkeypatch.setattr("olmo_eval.harness.Harness", make_harness)
        config = HarnessConfig(name="default", provider=ProviderConfig(kind=kind, model="m"))

        real_inference_worker(
            "worker-0",
            [],
            Mock(),
            Mock(),
            config.to_dict(),
            0,
            per_request=per_request,
        )

        assert chosen == [per_request and kind not in BatchConfig._SEQUENTIAL_ONLY]

    @pytest.mark.parametrize("resume", [True, False])
    def test_the_runner_tells_workers_whether_it_records_progress(
        self, tmp_path, monkeypatch, resume
    ) -> None:
        from olmo_eval.inference.provider_manager import ProviderManager

        told: list[bool] = []
        sent: list[dict[str, Any]] = []

        class Context:
            def Process(self, target: Any, args: tuple[Any, ...]) -> Any:
                from inspect import signature

                parameters = signature(real_inference_worker).parameters
                sent.append(dict(zip(parameters, args, strict=False)))
                raise RuntimeError("stop before starting workers")

        real_init = ProviderManager.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            told.append(kwargs.get("per_request", False))
            real_init(self, *args, **kwargs)

        def start(self: Any, ctx: Any, *args: Any) -> Any:
            return real_start(self, Context(), *args)

        real_start = ProviderManager.start
        monkeypatch.setattr(ProviderManager, "__init__", init)
        monkeypatch.setattr(ProviderManager, "start", start)

        with pytest.raises(RuntimeError, match="stop before starting workers"):
            run_eval(tmp_path, {"gsm8k": 2}, resume=resume)

        # The runner gives the manager the flag, and the manager passes it to each worker.
        assert told == [resume]
        assert sent[0]["per_request"] is resume


# ---------------------------------------------------------------------------
# Uncommitted code changes
# ---------------------------------------------------------------------------

_COMMIT = "a" * 40


def _code_at(monkeypatch, dirty: bool | None, diff: str | None = None) -> None:
    """Make the git helpers report a working tree: ``dirty`` None means no git."""
    info: dict[str, Any] = {"commit": None if dirty is None else _COMMIT, "dirty": dirty}
    monkeypatch.setattr("olmo_eval.upload.provenance.git_info", lambda: info)
    monkeypatch.setattr("olmo_eval.upload.provenance.uncommitted_diff_hash", lambda: diff)


class TestCodeChanges:
    def _restore(self, out: Path) -> None:
        progress, _, _ = _open(out)
        progress.close()

    @pytest.mark.parametrize(
        ("recorded", "current", "refused"),
        [
            ((True, "d1"), (False, None), True),
            ((False, None), (True, "d1"), True),
            ((True, "d1"), (True, "d2"), True),
            ((True, "d1"), (True, "d1"), False),
            ((False, None), (False, None), False),
            ((None, None), (True, "d1"), False),
            ((True, "d1"), (None, None), False),
        ],
    )
    def test_a_different_uncommitted_diff_is_refused_unless_git_is_unknown(
        self, tmp_path, monkeypatch, recorded, current, refused
    ) -> None:
        _code_at(monkeypatch, *recorded)
        self._restore(tmp_path)
        stored = next(e for e in _entries(tmp_path) if "header" in e)
        assert (
            stored["header"]["diff_hash"]
            == {
                (True, "d1"): "d1",
                (False, None): "clean",
                (None, None): None,
            }[recorded]
        )
        _code_at(monkeypatch, *current)

        if refused:
            with pytest.raises(ProgressMismatchError, match="code changed since the records"):
                self._restore(tmp_path)
            assert not (stream_dir(tmp_path, "mock").parent / ".lock").exists()
        else:
            self._restore(tmp_path)

    def test_the_refusal_says_what_to_do(self, tmp_path, monkeypatch) -> None:
        _code_at(monkeypatch, True, "d1")
        self._restore(tmp_path)
        _code_at(monkeypatch, False)

        with pytest.raises(ProgressMismatchError) as error:
            self._restore(tmp_path)

        message = str(error.value)
        assert "changes d1 then, no uncommitted changes now" in message
        assert "--fresh" in message and "another output directory (-O)" in message

    def test_starting_from_a_dirty_tree_warns_once(self, tmp_path, monkeypatch, caplog) -> None:
        _code_at(monkeypatch, True, "d1")
        with caplog.at_level(logging.WARNING):
            open_run_progress(str(tmp_path), "mock", _harness())
        warnings = [r for r in caplog.records if "uncommitted changes" in r.getMessage()]
        assert len(warnings) == 1
        assert "same uncommitted changes" in warnings[0].getMessage()

    @pytest.mark.parametrize("dirty", [False, None])
    def test_a_clean_or_unknown_tree_does_not_warn(
        self, tmp_path, monkeypatch, caplog, dirty
    ) -> None:
        _code_at(monkeypatch, dirty)
        with caplog.at_level(logging.WARNING):
            open_run_progress(str(tmp_path), "mock", _harness())
        assert "uncommitted" not in caplog.text

    def test_a_beaker_checkout_is_clean_without_running_git(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_REF", _COMMIT)
        monkeypatch.setattr(
            "olmo_eval.upload.provenance.uncommitted_diff_hash",
            lambda: pytest.fail("git was run"),
        )
        progress = open_run_progress(str(tmp_path), "mock", _harness())
        assert progress.diff_hash == "clean" and progress.git_commit == _COMMIT


# ---------------------------------------------------------------------------
# Runs that share an output directory
# ---------------------------------------------------------------------------

A_LIMITS = {"gsm8k": 8}
B_LIMITS = {"ifeval": 4}


def _lock_path(out: Path) -> Path:
    return stream_dir(out, "mock").parent / ".lock"


def _header_specs(out: Path) -> set[str]:
    return {e["header"]["spec"] for e in _entries(out) if "header" in e}


def _write_lock(out: Path, **holder: Any) -> None:
    import socket

    lock = _lock_path(out)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(
        json.dumps({"host": socket.gethostname(), "pid": 1, "started": "then", **holder})
    )


def _dead_pid() -> int:
    import subprocess
    import sys

    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def _same_artifacts(out: Path, reference: Path) -> None:
    """``out`` holds the reference run's predictions and requests, whatever else it holds."""
    ours, theirs = _artifacts(out), _artifacts(reference)
    assert theirs
    assert {name: ours.get(name) for name in theirs} == theirs
    assert _metrics(out) == _metrics(reference)


@pytest.fixture(scope="module")
def a_reference(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("a_reference")
    run_eval(out, A_LIMITS)
    return out


def _interrupt_a(out: Path) -> int:
    with pytest.raises(Terminated):
        run_eval(out, A_LIMITS, env={DEBUG_EXIT_AT_ENV: "0.5"})
    recorded = _recorded(out)
    assert recorded >= 2
    return recorded


class TestSharedOutputDirectory:
    def test_another_run_completing_keeps_this_runs_records(self, tmp_path, a_reference) -> None:
        recorded = _interrupt_a(tmp_path)

        stats_b = run_eval(tmp_path, B_LIMITS)

        assert stats_b.restored == 0
        assert _header_specs(tmp_path) == {"gsm8k"}
        assert _recorded(tmp_path) == recorded
        stats_a = run_eval(tmp_path, A_LIMITS)
        assert stats_a.restored == recorded
        _same_artifacts(tmp_path, a_reference)
        assert not (tmp_path / PROGRESS_DIR).exists()

    @pytest.mark.parametrize("failure", ["rename", "first removal", "second removal"])
    def test_a_crash_during_discard_leaves_the_other_runs_records_restorable(
        self, tmp_path, a_reference, monkeypatch, failure, caplog
    ) -> None:
        recorded = _interrupt_a(tmp_path)
        real_discard, real_unlink = RunProgress.discard, Path.unlink

        def discard(self: RunProgress) -> None:
            self._flush()  # so that the failure is in rewriting, not in this run's last segment
            removals = 0

            def unlink(path: Path, missing_ok: bool = False) -> None:
                nonlocal removals
                if path.suffix == ".jsonl":
                    removals += 1
                    if removals == (1 if failure == "first removal" else 2):
                        raise OSError("disk went away")
                real_unlink(path, missing_ok=missing_ok)

            def replace(src: Any, dst: Any) -> None:
                raise OSError("disk went away")

            with pytest.MonkeyPatch.context() as inner:
                if failure == "rename":
                    inner.setattr(progress_module.os, "replace", replace)
                else:
                    inner.setattr(Path, "unlink", unlink)
                real_discard(self)

        monkeypatch.setattr(RunProgress, "discard", discard)
        with caplog.at_level(logging.WARNING):
            run_eval(tmp_path, B_LIMITS)
        monkeypatch.undo()
        assert "Could not delete the progress records" in caplog.text

        assert not _lock_path(tmp_path).exists()
        assert _recorded(tmp_path) >= recorded
        stats = run_eval(tmp_path, A_LIMITS)
        assert stats.restored == recorded
        _same_artifacts(tmp_path, a_reference)

    def test_discard_keeps_other_tasks_final_results(self, tmp_path) -> None:
        run_eval(tmp_path, {"gsm8k": 4}, resume=True)  # completes: nothing is left
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(RunProgress, "discard", lambda self: None)
            run_eval(tmp_path, B_LIMITS)
        finals = sorted((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))
        assert len(finals) == 1

        run_eval(tmp_path, {"gsm8k": 4})

        assert sorted((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}")) == finals
        assert _header_specs(tmp_path) == {"ifeval"}

    def test_restoring_names_the_tasks_of_other_runs(self, interrupted_run, tmp_path, caplog):
        out = _resume_copy(interrupted_run, tmp_path)

        with caplog.at_level(logging.INFO):
            progress, _, _ = _open(out)
        progress.close()

        assert "keeping progress for gsm8k from another run" in caplog.text

    def test_a_run_of_its_own_tasks_does_not_warn(self, interrupted_run, tmp_path, caplog):
        out = _resume_copy(interrupted_run, tmp_path)

        with caplog.at_level(logging.INFO):
            run_eval(out, INTERRUPTED_LIMITS)

        assert "from another run" not in caplog.text


class TestFresh:
    def test_fresh_discards_only_this_runs_tasks_for_its_model(self, tmp_path) -> None:
        _interrupt_a(tmp_path)
        with pytest.raises(Terminated):
            run_eval(tmp_path, {"ifeval": 8}, env={DEBUG_EXIT_AT_ENV: "0.5"})
        other = tmp_path / PROGRESS_DIR / "other-model"
        other.mkdir()
        (other / "run.segments").write_text("{}\n")
        assert _header_specs(tmp_path) == {"gsm8k", "ifeval"}

        run_eval(tmp_path, A_LIMITS, fresh=True, resume=False)

        assert _header_specs(tmp_path) == {"ifeval"}
        assert (other / "run.segments").is_file()

    def test_fresh_removes_the_tasks_final_results_and_empty_directories(self, tmp_path) -> None:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(RunProgress, "discard", lambda self: None)
            run_eval(tmp_path, A_LIMITS)
        assert list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))

        run_eval(tmp_path, A_LIMITS, fresh=True, resume=False)

        assert not (tmp_path / PROGRESS_DIR).exists()

    def test_fresh_deletes_a_stream_it_cannot_read(self, tmp_path, caplog) -> None:
        _interrupt_a(tmp_path)
        segment = next(stream_dir(tmp_path, "mock").glob("*.jsonl"))
        segment.write_text("not json\n")

        with caplog.at_level(logging.WARNING):
            run_eval(tmp_path, A_LIMITS, fresh=True, resume=False)

        assert "Deleting the whole stream" in caplog.text
        assert not has_progress(tmp_path)

    def test_fresh_is_refused_while_another_run_records(self, tmp_path) -> None:
        progress, _, _ = _open(tmp_path)
        try:
            with pytest.raises(ProgressLockedError):
                run_eval(tmp_path, A_LIMITS, fresh=True, resume=False)
        finally:
            progress.close()


class TestProgressLock:
    def test_a_second_recording_run_is_refused_and_names_the_holder(self, tmp_path) -> None:
        import socket

        first, _, _ = _open(tmp_path)
        try:
            with pytest.raises(ProgressLockedError) as error:
                _open(tmp_path)
        finally:
            first.close()

        message = str(error.value)
        assert socket.gethostname() in message and f"pid {os.getpid()}" in message
        assert "another output directory (-O)" in message
        _open(tmp_path)[0].close()  # free again once the first run is done

    def test_models_lock_independently(self, tmp_path) -> None:
        first, _, _ = _open(tmp_path)
        try:
            other = open_run_progress(str(tmp_path), "other-model", _harness(model="other-model"))
            trackers, items = _local_setup()
            other.restore(trackers, items, _harness(model="other-model"))
            other.close()
        finally:
            first.close()

    def test_a_lock_of_a_dead_process_is_taken_over(self, tmp_path, caplog) -> None:
        _write_lock(tmp_path, pid=_dead_pid())

        with caplog.at_level(logging.WARNING):
            progress, _, _ = _open(tmp_path)

        assert "Taking over" in caplog.text and "no longer running" in caplog.text
        assert json.loads(_lock_path(tmp_path).read_text())["pid"] == os.getpid()
        progress.close()
        assert not _lock_path(tmp_path).exists()

    @pytest.mark.parametrize("holder_job", ["job-1", "job-2", None])
    def test_a_beaker_job_takes_over_the_lock_it_inherited(
        self, tmp_path, monkeypatch, caplog, holder_job
    ) -> None:
        # /results of the preempted job (or an earlier attempt of this one) was copied here,
        # from another host whose process cannot be checked.
        _write_lock(tmp_path, host="another-node", beaker_job_id=holder_job)
        monkeypatch.setenv("BEAKER_JOB_ID", "job-2")

        with caplog.at_level(logging.WARNING):
            progress, _, _ = _open(tmp_path)

        assert "Taking over" in caplog.text
        assert json.loads(_lock_path(tmp_path).read_text())["beaker_job_id"] == "job-2"
        progress.close()

    def test_a_lock_from_another_host_is_not_guessed_at_off_beaker(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("BEAKER_JOB_ID", raising=False)
        _write_lock(tmp_path, host="another-node", pid=_dead_pid())

        with pytest.raises(ProgressLockedError, match="another-node.*delete the lock file"):
            _open(tmp_path)

    def test_an_unreadable_lock_is_refused(self, tmp_path) -> None:
        _lock_path(tmp_path).parent.mkdir(parents=True)
        _lock_path(tmp_path).write_text("")

        with pytest.raises(ProgressLockedError, match="unreadable"):
            _open(tmp_path)

    def test_a_lock_is_not_progress(self, tmp_path) -> None:
        progress, _, _ = _open(tmp_path)
        try:
            assert not has_progress(tmp_path)
        finally:
            progress.close()

    def test_release_leaves_a_lock_another_run_took_over(self, tmp_path) -> None:
        progress, _, _ = _open(tmp_path)
        _write_lock(tmp_path, pid=os.getpid() + 1, token="someone-else")

        progress.close()

        assert json.loads(_lock_path(tmp_path).read_text())["token"] == "someone-else"

    def test_the_lock_is_released_after_a_completed_run(self, tmp_path) -> None:
        run_eval(tmp_path, A_LIMITS)

        assert not (tmp_path / PROGRESS_DIR).exists()

    def test_the_lock_is_released_after_sigterm(self, tmp_path) -> None:
        _interrupt_a(tmp_path)

        assert has_progress(tmp_path) and not _lock_path(tmp_path).exists()

    def test_the_lock_is_released_when_the_records_are_refused(self, interrupted_run, tmp_path):
        out = _resume_copy(interrupted_run, tmp_path)

        with pytest.raises(ProgressMismatchError):
            run_eval(out, INTERRUPTED_LIMITS, overrides={"gsm8k": {"max_tokens": 128}})

        assert not _lock_path(out).exists()

    def test_a_run_without_resume_takes_no_lock(self, tmp_path) -> None:
        progress, _, _ = _open(tmp_path)
        try:
            run_eval(tmp_path, A_LIMITS, resume=False)
        finally:
            progress.close()

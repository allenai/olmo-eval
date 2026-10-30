"""Resuming a run from the progress records an earlier attempt left in its output directory.

The end-to-end tests run the async runner on real task specs, with each task's
dataset replaced by a few local documents and the mock provider replaced by
``resume_fakes.HashProvider``, whose answers differ by request. A resumed run must
write the same predictions, requests and metrics as an uninterrupted one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import queue
import re
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import resume_fakes

import olmo_eval.evals.tasks  # noqa: F401  (registers the tasks)
from olmo_eval.cli.run import _sigterm_raises
from olmo_eval.common.metrics import AccuracyMetric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.llm_judge import Judge
from olmo_eval.common.shutdown import Terminated, clear_shutdown, request_shutdown
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType
from olmo_eval.evals.tasks.common import Task, TaskConfig
from olmo_eval.harness.config import HarnessConfig, ProviderConfig
from olmo_eval.runners.asynq import runner as runner_module
from olmo_eval.runners.asynq.progress import (
    DEBUG_EXIT_AT_ENV,
    FINAL_SUFFIX,
    PROGRESS_DIR,
    PROGRESS_LAYOUT_ENV,
    RECORD_VERSION,
    DroppedLogprobsError,
    ProgressMismatchError,
    RunProgress,
    decode_value,
    discard_progress,
    encode_value,
    has_progress,
    open_run_progress,
    parse_exit_thresholds,
    store_class,
    stream_base,
)
from olmo_eval.runners.asynq.results import process_results
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker
from olmo_eval.runners.asynq.workers import inference_worker as real_inference_worker
from olmo_eval.upload.manifest import RUN_SECRET_NAME, read_manifest, run_secret

LAYOUTS = ["segments", "append"]

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


def _harness(system_prompt: str | None = None, **kwargs: Any) -> HarnessConfig:
    # HashProvider fails some requests on purpose; keep every task's metrics.
    return HarnessConfig(
        name="default",
        provider=ProviderConfig(kind="mock", model="mock", **kwargs),
        max_hard_failure_rate=1.0,
        system_prompt=system_prompt,
    )


def _runner(
    out: Path,
    limits: dict[str, int] = TASK_LIMITS,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> AsyncEvalRunner:
    return AsyncEvalRunner(
        harness_config=_harness(system_prompt),
        task_specs=list(limits),
        task_overrides={spec: {"limit": limit} for spec, limit in limits.items()},
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
    layout: str,
    limits: dict[str, int] = TASK_LIMITS,
    env: dict[str, str] | None = None,
    resume: bool = True,
    **kwargs: Any,
) -> RunStats:
    """Run the evaluation into ``out`` under the CLI's SIGTERM handling.

    Progress is only recorded with ``resume``, which is on unless a test turns it off.
    """
    seen: dict[str, Any] = {"queued": None, "restored": 0, "finalized": set()}

    async def spy(**call: Any) -> Any:
        seen["queued"] = call["total_instances"]
        seen["restored"] = sum(len(r) for r in call["restored"].values()) + sum(
            len(t.failed_instances) for t in call["trackers"].values()
        )
        seen["finalized"] = set(call["finalized"])
        return await process_results(**call)

    judge_calls = len(resume_fakes.JUDGE_CALLS)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(PROGRESS_LAYOUT_ENV, layout)
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


def _stream(out: Path, layout: str) -> Any:
    return store_class(layout)(stream_base(out, "mock"))


def _entries(out: Path, layout: str) -> list[dict[str, Any]]:
    return list(_stream(out, layout).read())


def _instance_entries(out: Path, layout: str) -> list[dict[str, Any]]:
    return [entry for entry in _entries(out, layout) if "idx" in entry]


def _recorded(out: Path, layout: str) -> int:
    return len(_instance_entries(out, layout))


def keep_first_entries(out: Path, layout: str, keep: int) -> int:
    """Cut the records back to ``keep`` instance entries in all, as an interrupted run leaves.

    Finalized results go too. Returns the number of entries kept.
    """
    for final in (out / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"):
        final.unlink()
    store = _stream(out, layout)
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


@pytest.fixture(scope="module", params=LAYOUTS)
def full_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory):
    """An uninterrupted run in each layout, with its records kept, and its layout."""
    out = tmp_path_factory.mktemp(f"full-{request.param}")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(RunProgress, "discard", lambda self: None)
        stats = run_eval(out, request.param)
    assert stats.queued == TOTAL and stats.restored == 0
    return out, request.param


def _copy(full: Path, tmp_path: Path, *, keep_outputs: bool = False) -> Path:
    out = tmp_path / "run"
    shutil.copytree(full, out)
    if not keep_outputs:
        _clear_outputs(out)
    return out


class TestRoundTrip:
    def test_records_hold_outcomes_not_prompts(self, full_run) -> None:
        full, layout = full_run
        entries = _entries(full, layout)
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
        full, layout = full_run
        stream = _stream(full, layout)
        assert stream.path.exists()
        assert len(list((full / PROGRESS_DIR).glob(f"*/*{stream.suffix}"))) == 1
        if layout == "segments":
            # A run is written in a few segments however many tasks it has.
            assert len(list(stream.path.glob("*.jsonl"))) <= 2

    def test_finalized_tasks_are_not_run_again(self, full_run, tmp_path) -> None:
        full, layout = full_run
        out = _copy(full, tmp_path, keep_outputs=True)
        (out / "metrics.json").unlink()

        stats = run_eval(out, layout, resume=True)

        assert stats.queued == 0
        assert stats.judge_calls == 0
        assert stats.finalized == set(TASK_LIMITS)
        assert_same_results(out, full)
        assert not (out / PROGRESS_DIR).exists()

    def test_rerun_from_records_generates_only_unrecorded(self, full_run, tmp_path) -> None:
        full, layout = full_run
        out = _copy(full, tmp_path)

        stats = run_eval(out, layout, resume=True)

        # Predictions are gone, so no finalized result can be used; the records can.
        assert stats.finalized == set()
        assert stats.restored == _recorded(full, layout)
        assert stats.queued == TOTAL - stats.restored
        assert_same_results(out, full)

    @pytest.mark.parametrize("cut", [0, 1, 0.37, 0.99])
    def test_resume_after_a_cut_matches_an_uninterrupted_run(self, full_run, tmp_path, cut) -> None:
        full, layout = full_run
        out = _copy(full, tmp_path)
        kept = keep_first_entries(out, layout, cut if isinstance(cut, int) else int(cut * TOTAL))

        stats = run_eval(out, layout, resume=True)

        # Every instance not taken from the records is generated by this attempt.
        assert stats.restored == kept
        assert stats.queued == TOTAL - kept
        assert_same_results(out, full)

    def test_sigterm_then_resume(self, full_run, tmp_path, monkeypatch) -> None:
        full, layout = full_run
        out = tmp_path / "run"

        with pytest.raises(Terminated):
            run_eval(out, layout, env={DEBUG_EXIT_AT_ENV: "0.37"})
        recorded = _recorded(out, layout)
        assert recorded >= 0.37 * TOTAL
        first = read_manifest(out)
        assert first is not None and first["run"]["status"] == "running"
        secret = run_secret(out)

        # Without --resume the progress is ignored, with a warning, and not touched.
        elsewhere = tmp_path / "ordinary"
        shutil.copytree(out, elsewhere)
        before = sorted(p.name for p in (elsewhere / PROGRESS_DIR).rglob("*"))
        warnings: list[str] = []
        monkeypatch.setattr(runner_module.runner_logger, "warning", warnings.append)
        ordinary = run_eval(elsewhere, layout, resume=False)
        assert ordinary.queued == TOTAL and ordinary.restored == 0
        assert any("ignores" in message for message in warnings)
        assert sorted(p.name for p in (elsewhere / PROGRESS_DIR).rglob("*")) == before
        assert_same_results(elsewhere, full)

        # The threshold was crossed before this attempt started, so it finishes.
        stats = run_eval(out, layout, env={DEBUG_EXIT_AT_ENV: "0.37"}, resume=True)

        assert stats.queued is not None and stats.queued <= TOTAL - recorded
        assert_same_results(out, full)
        manifest = read_manifest(out)
        assert manifest is not None
        assert manifest["run"]["run_id"] == first["run"]["run_id"]
        assert manifest["run"]["status"] == "complete"
        assert manifest["resume"]["resumed_count"] == 1
        assert (out / RUN_SECRET_NAME).read_text().strip() == secret
        assert not (out / PROGRESS_DIR).exists()

    def test_fresh_discards_recorded_progress(self, full_run, tmp_path) -> None:
        full, layout = full_run
        out = _copy(full, tmp_path)

        stats = run_eval(out, layout, fresh=True, resume=False)

        assert stats.queued == TOTAL and stats.restored == 0
        assert_same_results(out, full)
        assert not has_progress(out)

    def test_a_run_without_resume_writes_no_progress(self, full_run, tmp_path) -> None:
        full, layout = full_run
        out = tmp_path / "ordinary"

        stats = run_eval(out, layout, resume=False)

        assert stats.queued == TOTAL
        assert not (out / PROGRESS_DIR).exists()
        assert_same_results(out, full)

    def test_resume_without_progress_matches_an_ordinary_run(self, full_run, tmp_path) -> None:
        full, layout = full_run
        out = tmp_path / "first"

        run_eval(out, layout, resume=True)

        assert_same_results(out, full)
        assert not (out / PROGRESS_DIR).exists()

    def test_a_system_prompt_reaches_restored_requests(self, tmp_path) -> None:
        # Restored responses carry the request the harness sent, so the saved requests
        # must hold the system prompt exactly as an uninterrupted run saves them.
        limits = {"gsm8k": 6, "arc_challenge:mc": 6}
        reference = tmp_path / "reference"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(RunProgress, "discard", lambda self: None)
            run_eval(reference, "segments", limits, system_prompt="Be brief.")
        assert "Be brief." in b"".join(_artifacts(reference).values()).decode()
        out = tmp_path / "cut"
        shutil.copytree(reference, out)
        _clear_outputs(out)
        kept = keep_first_entries(out, "segments", 8)

        stats = run_eval(out, "segments", limits, system_prompt="Be brief.")

        assert stats.restored == kept and kept > 0
        assert_same_results(out, reference)


class TestStores:
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_a_torn_last_line_is_dropped(self, tmp_path, layout) -> None:
        store = store_class(layout)(tmp_path / "run")
        store.reset()
        store.append({"header": True})
        store.append({"idx": 0})
        store.close()
        torn = b'{"idx": 1, "outputs": [{"te'
        if layout == "append":
            with open(store.path, "ab") as f:
                f.write(torn)
        else:
            (store.path / "000099.jsonl").write_bytes(b'{"idx": 1}\n' + torn)

        reopened = store_class(layout)(tmp_path / "run")
        expected = [{"header": True}, {"idx": 0}] + ([] if layout == "append" else [{"idx": 1}])
        assert list(reopened.read()) == expected
        reopened.append({"idx": 2})
        reopened.close()
        assert list(store_class(layout)(tmp_path / "run").read()) == [
            *expected,
            {"idx": 2},
        ]

    def test_segments_are_flushed_by_size_and_age(self, tmp_path, monkeypatch) -> None:
        from olmo_eval.runners.asynq import progress as progress_module

        monkeypatch.setattr(progress_module, "SEGMENT_MAX_BYTES", 100)
        store = store_class("segments")(tmp_path / "run")
        for i in range(30):
            store.append({"idx": i, "pad": "x" * 20})
        # Size flushes write a few segments; the buffered rest waits for close.
        written = len(list(store.path.glob("*.jsonl")))
        assert 1 <= written < 30
        store.close()
        assert [e["idx"] for e in store_class("segments")(tmp_path / "run").read()] == list(
            range(30)
        )

        monkeypatch.setattr(progress_module, "SEGMENT_MAX_BYTES", 10**9)
        monkeypatch.setattr(progress_module, "SEGMENT_MAX_SECONDS", 0.05)
        aged = store_class("segments")(tmp_path / "aged")
        aged.append({"idx": 0})
        deadline = time.monotonic() + 5
        while not list(aged.path.glob("*.jsonl")) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert list(aged.path.glob("*.jsonl"))
        aged.close()

    def test_segments_are_the_default_layout(self, monkeypatch) -> None:
        monkeypatch.delenv(PROGRESS_LAYOUT_ENV, raising=False)
        assert store_class().suffix == ".segments"
        monkeypatch.setenv(PROGRESS_LAYOUT_ENV, "sqlite")
        with pytest.raises(ValueError, match="not a progress layout"):
            store_class()

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


class TestHeaders:
    def _restore(self, out: Path, runner: AsyncEvalRunner) -> Any:
        _, trackers, items = runner._prepare_tasks()
        progress = open_run_progress(str(out), runner.model_name, runner.harness_config)
        try:
            return progress.restore(trackers, items, runner.harness_config)
        finally:
            progress.close()

    def test_a_different_max_tokens_names_the_setting(self, tmp_path) -> None:
        self._restore(tmp_path, _runner(tmp_path, {"gsm8k": 3}))
        changed = _runner(tmp_path, {"gsm8k": 3})
        changed.task_overrides["gsm8k"]["max_tokens"] = 128

        with pytest.raises(
            ProgressMismatchError,
            match=r"task_hash differs.*sampling_params.max_tokens: 512 -> 128",
        ):
            self._restore(tmp_path, changed)

    def test_a_different_model_revision_is_refused(self, tmp_path) -> None:
        self._restore(tmp_path, _runner(tmp_path, {"gsm8k": 3}))
        changed = _runner(tmp_path, {"gsm8k": 3})
        changed.harness_config = _harness(revision="v2")

        with pytest.raises(ProgressMismatchError, match="model_hash differs"):
            self._restore(tmp_path, changed)

    def test_a_different_system_prompt_is_refused(self, tmp_path) -> None:
        self._restore(tmp_path, _runner(tmp_path, {"gsm8k": 3}))
        changed = _runner(tmp_path, {"gsm8k": 3})
        changed.harness_config = HarnessConfig(
            name="default",
            provider=ProviderConfig(kind="mock", model="mock"),
            system_prompt="Be brief.",
        )

        with pytest.raises(ProgressMismatchError, match="harness_hash differs"):
            self._restore(tmp_path, changed)

    def test_a_different_engine_version_is_refused(self, tmp_path, monkeypatch) -> None:
        from olmo_eval.runners.asynq import progress as progress_module

        monkeypatch.setattr(progress_module, "_package_version", lambda name: "0.10.0")
        runner = _runner(tmp_path, {"gsm8k": 3})
        runner.harness_config = HarnessConfig(
            name="default", provider=ProviderConfig(kind="vllm", model="m")
        )
        self._restore(tmp_path, runner)
        monkeypatch.setattr(progress_module, "_package_version", lambda name: "0.11.0")

        with pytest.raises(
            ProgressMismatchError, match=r"engine_version differs.*vllm==0.10.0.*vllm==0.11.0"
        ):
            self._restore(tmp_path, runner)

    def test_an_unknown_engine_version_warns_instead_of_matching(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        from olmo_eval.runners.asynq import progress as progress_module

        monkeypatch.setattr(progress_module, "_package_version", lambda name: "0.10.0")
        runner = _runner(tmp_path, {"gsm8k": 3})
        runner.harness_config = HarnessConfig(
            name="default", provider=ProviderConfig(kind="vllm", model="m")
        )
        self._restore(tmp_path, runner)
        monkeypatch.setattr(progress_module, "_package_version", lambda name: None)

        with caplog.at_level("WARNING"):
            self._restore(tmp_path, runner)

        assert "without a comparable engine version" in caplog.text
        assert "vllm==unknown" in caplog.text

    def test_vllm_server_version_comes_from_the_server_python(self, monkeypatch) -> None:
        import subprocess

        from olmo_eval.runners.asynq import progress as progress_module

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

        from olmo_eval.runners.asynq import progress as progress_module

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="No module named vllm")

        monkeypatch.setattr(subprocess, "run", fake_run)
        config = HarnessConfig(
            name="default", provider=ProviderConfig(kind="vllm_server", model="m")
        )

        with caplog.at_level("WARNING"):
            assert progress_module.engine_version(config) == "vllm==unknown"
        assert "Could not determine the vllm version" in caplog.text

    def test_a_different_commit_is_refused(self, tmp_path, monkeypatch) -> None:
        runner = _runner(tmp_path, {"gsm8k": 3})
        monkeypatch.setattr("olmo_eval.upload.provenance.git_info", lambda: {"commit": "a" * 40})
        self._restore(tmp_path, runner)
        stored = next(e for e in _entries(tmp_path, "segments") if "header" in e)
        assert stored["header"]["git_commit"] == "a" * 40
        monkeypatch.setattr("olmo_eval.upload.provenance.git_info", lambda: {"commit": "b" * 40})

        with pytest.raises(ProgressMismatchError, match="git_commit differs"):
            self._restore(tmp_path, runner)

    def test_runtime_identity_ignores_what_depends_on_the_node(self, tmp_path) -> None:
        runtime = {
            "profile": "qwen",
            "packages": {"sglang": "0.5"},
            "logprobs": "original",
            "arithmetic_environment": {"A": "1"},
            "engine_options": {"attention_backend": "fa3", "max_running_requests": 64},
            "effective_max_running_requests": 64,
        }
        progress = _progress(tmp_path)
        trackers, items = _local_setup(undecidable=set())
        progress.restore(trackers, items, _harness())
        progress.check_runtime(runtime)
        progress.close()

        other_node = {
            **runtime,
            "engine_options": {"attention_backend": "fa3", "max_running_requests": 48},
            "effective_max_running_requests": 48,
        }
        again = _progress(tmp_path)
        trackers, items = _local_setup(undecidable=set())
        again.restore(trackers, items, _harness())
        again.check_runtime(other_node)

        new_engine = {**runtime, "packages": {"sglang": "0.6"}}
        with pytest.raises(ProgressMismatchError, match=r"packages.sglang: '0.5' -> '0.6'"):
            again.check_runtime(new_engine)
        backend = {**runtime, "engine_options": {"attention_backend": "triton"}}
        with pytest.raises(ProgressMismatchError, match="attention_backend"):
            again.check_runtime(backend)
        again.close()

    def test_request_length_caps_are_part_of_the_runtime_identity(self, tmp_path) -> None:
        runtime = {
            "profile": "qwen",
            "engine_options": {"max_total_tokens": 1000, "mem_fraction_static": 0.9},
            "max_total_num_tokens": 1000,
            "max_req_input_len": 999,
        }
        progress = _progress(tmp_path)
        trackers, items = _local_setup(undecidable=set())
        progress.restore(trackers, items, _harness())
        progress.check_runtime(runtime)
        progress.close()

        again = _progress(tmp_path)
        trackers, items = _local_setup(undecidable=set())
        again.restore(trackers, items, _harness())
        # The requested pool and memory fraction differ, but the caps they led to match.
        again.check_runtime(
            {**runtime, "engine_options": {"max_total_tokens": 2000, "mem_fraction_static": 0.8}}
        )
        smaller_pool = {**runtime, "max_total_num_tokens": 500, "max_req_input_len": 499}
        with pytest.raises(ProgressMismatchError, match="max_total_num_tokens: 1000 -> 500"):
            again.check_runtime(smaller_pool)
        again.close()

    def test_unreported_caps_fall_back_to_the_requested_pool(self) -> None:
        from olmo_eval.runners.asynq.progress import runtime_identity

        runtime = {"engine_options": {"max_total_tokens": 1000, "max_running_requests": 8}}
        assert runtime_identity(runtime) == {"engine_options": {"max_total_tokens": 1000}}

    def test_a_different_setup_only_warns(self, tmp_path, caplog) -> None:
        self._restore(tmp_path, _runner(tmp_path, {"gsm8k": 3}))
        changed = _runner(tmp_path, {"gsm8k": 3})
        changed.harness_config = _harness(num_instances=4)

        with caplog.at_level(logging.WARNING):
            self._restore(tmp_path, changed)
        assert "num_instances: 1 -> 4" in caplog.text

    def test_a_changed_request_is_run_again(self, tmp_path, caplog) -> None:
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        _process(trackers, restored.remaining, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)

        trackers, items = _local_setup(undecidable=set(), prompt="changed q{i}")
        progress = _progress(tmp_path)
        with caplog.at_level(logging.WARNING):
            restored = progress.restore(trackers, items, _harness())
        progress.close()
        assert restored.responses == {}
        assert len(restored.remaining) == len(items)
        assert "4 recorded instance(s) no longer match" in caplog.text

    @pytest.mark.parametrize("changed", ["gold_answer", "metadata"])
    def test_a_changed_gold_or_metadata_is_run_again(self, tmp_path, changed) -> None:
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        _process(trackers, restored.remaining, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)

        trackers, items = _local_setup(undecidable=set())
        edited = items[2]
        if changed == "gold_answer":
            edited.instance = replace(edited.instance, gold_answer="another answer")
        else:
            edited.instance.metadata["source"] = "a corrected document"
        restored = _progress(tmp_path).restore(trackers, items, _harness())

        assert sorted(restored.responses["local"]) == [0, 1, 3]
        assert [item.instance_idx for item in restored.remaining] == [2]


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
    undecidable: set[int], unreachable: frozenset[int] = frozenset(), prompt: str = "q{i}"
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
            request=LMRequest(request_type=RequestType.COMPLETION, prompt=prompt.format(i=i)),
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


class TestAccounting:
    def test_only_outcomes_an_uninterrupted_run_keeps_are_recorded(self, tmp_path) -> None:
        # 0: a request-caused failure (final); 1: a failed call (retried); 2: undecidable
        # (final, unscored); 3: grader unreachable (retried).
        trackers, items = _local_setup(undecidable={2}, unreachable=frozenset({3}))
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        first = _process(
            trackers, restored.remaining, progress, restored.responses, {0: True, 1: False}
        )
        progress.close()
        _drop_final(tmp_path)

        trackers, items = _local_setup(undecidable={2}, unreachable=frozenset({3}))
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())

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
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        first = _process(trackers, restored.remaining, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)

        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        assert restored.remaining == []
        second = _process(trackers, [], progress, restored.responses)
        progress.close()

        assert second["local"].metrics == first["local"].metrics
        assert second["local"].predictions == first["local"].predictions
        assert trackers["local"].prompt_tokens_total == 12

    def test_a_restored_instance_that_arrives_again_is_rejected(self, tmp_path) -> None:
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        _process(trackers, restored.remaining, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)

        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        with pytest.raises(RuntimeError, match="accounting mismatch"):
            _process(trackers, items[:1], progress, restored.responses)
        progress.close()

    def test_nothing_is_recorded_after_shutdown_was_requested(self, tmp_path) -> None:
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        request_shutdown()
        try:
            with pytest.raises(Terminated):
                _process(trackers, restored.remaining, progress, restored.responses, {0: False})
        finally:
            clear_shutdown()
            progress.close()

        trackers, items = _local_setup(undecidable=set())
        restored = _progress(tmp_path).restore(trackers, items, _harness())
        assert restored.responses == {} and trackers["local"].failed_instances == {}
        assert not list((tmp_path / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))


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


class TestRunIdentity:
    def _write_manifest(self, out: Path, status: str, specs: list[str]) -> str:
        runner = _runner(out, {spec: 3 for spec in specs})
        runner._previous_run_id()
        runner._start_run_record("abc123def456")
        path = out / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["run"]["status"] = status
        path.write_text(json.dumps(manifest))
        return "abc123def456"

    @staticmethod
    def _leave_progress(out: Path) -> None:
        record = stream_base(out, "mock").with_suffix(".jsonl")
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text("{}\n")

    def test_a_resumed_run_keeps_its_id(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("BEAKER_JOB_ID", "job-1")
        run_id = self._write_manifest(tmp_path, "failed", ["gsm8k"])
        self._leave_progress(tmp_path)
        monkeypatch.setenv("BEAKER_JOB_ID", "job-2")
        runner = _runner(tmp_path, {"gsm8k": 3}, resume=True)

        assert runner._previous_run_id() == run_id
        runner._start_run_record(run_id)

        manifest = read_manifest(tmp_path)
        assert manifest is not None
        assert manifest["resume"] == {"resumed_count": 1, "beaker_job_ids": ["job-1", "job-2"]}
        assert manifest["run"]["status"] == "running"

    @pytest.mark.parametrize(
        ("status", "specs", "resume"),
        [
            ("complete", ["gsm8k"], True),
            ("running", ["ifeval"], True),
            ("running", ["gsm8k"], False),
        ],
    )
    def test_a_finished_other_or_unresumed_run_gets_a_new_id(
        self, tmp_path, status, specs, resume
    ) -> None:
        self._write_manifest(tmp_path, status, specs)
        assert _runner(tmp_path, {"gsm8k": 3}, resume=resume)._previous_run_id() is None

    def test_discarded_progress_gives_a_new_id(self, tmp_path) -> None:
        self._write_manifest(tmp_path, "failed", ["gsm8k"])
        runner = _runner(tmp_path, {"gsm8k": 3}, resume=True)
        assert runner._previous_run_id() is None
        self._leave_progress(tmp_path)
        assert runner._previous_run_id() == "abc123def456"
        discard_progress(tmp_path)
        assert _runner(tmp_path, {"gsm8k": 3}, resume=True)._previous_run_id() is None


class TestProgressIsNotUploaded:
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

    def _finalize(self, tmp_path, results, progress, resume=True) -> bool:
        """Run the finalize step; whether the records were still there when the upload ran."""
        from unittest.mock import patch

        from olmo_eval.runners.asynq.results import FailureBudgetExceeded
        from olmo_eval.upload import UploadConfig

        runner = _runner(tmp_path, {"gsm8k": 3}, resume=resume)
        runner.upload_config = UploadConfig(enabled=True, api_url="https://ingest.test")
        runner._start_run_record = lambda experiment_id: None  # type: ignore[method-assign]
        seen: list[bool] = []

        def upload(output_dir, config) -> None:
            assert (Path(output_dir) / "metrics.json").exists()
            seen.append(has_progress(output_dir))

        results_dict = {"model": "mock", "model_config": {}, "tasks": {}, "errors": []}
        with (
            patch("olmo_eval.upload.upload_results_dir", side_effect=upload),
            contextlib.suppress(FailureBudgetExceeded),
        ):
            runner._finalize_and_gate(
                results, results_dict, experiment_id="abc123def456", progress=progress
            )
        assert len(seen) == 1
        return seen[0]

    def _finished(self, tmp_path):
        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        results = _process(trackers, restored.remaining, progress, restored.responses)
        assert has_progress(tmp_path)
        return progress, results

    def test_progress_is_deleted_after_a_completed_run_uploads(self, tmp_path) -> None:
        progress, results = self._finished(tmp_path)

        assert self._finalize(tmp_path, results, progress) is True
        assert not has_progress(tmp_path)

    def test_a_failed_upload_keeps_progress_for_the_requeue(self, tmp_path) -> None:
        from unittest.mock import patch

        from olmo_eval.upload import UploadConfig

        progress, results = self._finished(tmp_path)
        runner = _runner(tmp_path, {"gsm8k": 3}, resume=True)
        runner.upload_config = UploadConfig(enabled=True, api_url="https://ingest.test")
        runner._start_run_record = lambda experiment_id: None  # type: ignore[method-assign]
        results_dict = {"model": "mock", "model_config": {}, "tasks": {}, "errors": []}

        with (
            patch("olmo_eval.upload.upload_results_dir", side_effect=RuntimeError("preempted")),
            pytest.raises(RuntimeError, match="preempted"),
        ):
            runner._finalize_and_gate(
                results, results_dict, experiment_id="abc123def456", progress=progress
            )
        assert has_progress(tmp_path)

    def test_a_run_that_trips_the_gate_keeps_progress_for_its_retry(self, tmp_path) -> None:
        progress, results = self._finished(tmp_path)
        results["local"].failure_budget_exceeded = True

        assert self._finalize(tmp_path, results, progress) is True
        assert has_progress(tmp_path)
        progress.close()

    def test_a_completed_run_deletes_only_its_own_records(self, tmp_path) -> None:
        other = tmp_path / PROGRESS_DIR / "other-model"
        other.mkdir(parents=True)
        (other / "run.jsonl").write_text("{}\n")
        (other / f"gsm8k_abcdef{FINAL_SUFFIX}").write_text("{}")
        progress, results = self._finished(tmp_path)
        mine = stream_base(tmp_path, "mock").parent
        assert mine.is_dir()

        self._finalize(tmp_path, results, progress)

        assert not mine.exists()
        assert sorted(p.name for p in other.iterdir()) == [
            f"gsm8k_abcdef{FINAL_SUFFIX}",
            "run.jsonl",
        ]

    def test_discard_removes_emptied_directories(self, tmp_path) -> None:
        progress, _ = self._finished(tmp_path)
        progress.discard()
        assert not (tmp_path / PROGRESS_DIR).exists()

    def test_a_run_without_resume_leaves_stale_progress_and_warns(
        self, tmp_path, monkeypatch
    ) -> None:
        progress, results = self._finished(tmp_path)
        progress.close()
        warnings: list[str] = []
        monkeypatch.setattr(runner_module.runner_logger, "warning", warnings.append)

        self._finalize(tmp_path, results, None, resume=False)

        assert has_progress(tmp_path)
        assert any("stale" in w for w in warnings)


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
    tmp_path: Path, results: Any, *metrics: Any
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Record a run of the local task, then restore it in full and finish it again.

    ``results`` maps each queue item to its result. Returns both runs' task results
    and the recorded instance entries.
    """

    def setup() -> tuple[dict[str, TaskTracker], list[QueueItem]]:
        trackers, items = _local_setup(undecidable=set())
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
    recorded = _instance_entries(tmp_path, "segments")

    trackers, items = setup()
    progress = _progress(tmp_path)
    restored = progress.restore(trackers, items, _harness())
    assert restored.remaining == []
    second = _process_results(trackers, [], progress, restored.responses)
    progress.close()
    return first, second, recorded


class TestCompactRecords:
    def test_generations_are_recorded_without_per_token_logprobs(self, full_run) -> None:
        full, layout = full_run
        entries = _entries(full, layout)
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
        store = _stream(tmp_path, "segments")
        entries = list(store.read())
        store.reset()
        for entry in entries:
            if "header" in entry:
                entry["header"]["version"] = RECORD_VERSION - 1
            store.append(entry)
        store.close()

        trackers, items = _local_setup(undecidable=set())
        with pytest.raises(ProgressMismatchError, match=r"record format 2.*--fresh"):
            _progress(tmp_path).restore(trackers, items, _harness())


# --- Results return per request while progress is recorded ---


class _SlowProvider:
    """Answers each request with its prompt; the request "slow" waits to be released."""

    max_concurrency = 8

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls: list[list[str]] = []

    def describe_request(self, request: LMRequest, params: Any) -> None:
        return None

    async def agenerate(self, requests: list[LMRequest], params: Any) -> list[list[LMOutput]]:
        from olmo_eval.inference.errors import TerminalProviderError

        self.calls.append([str(request.prompt) for request in requests])

        async def one(request: LMRequest) -> list[LMOutput]:
            if request.prompt == "slow":
                await self.release.wait()
            if request.prompt == "dead":
                raise TerminalProviderError("engine died")
            return [LMOutput(text=str(request.prompt))]

        return list(await asyncio.gather(*(one(request) for request in requests)))


class _FakeHarness:
    def __init__(self, provider: _SlowProvider) -> None:
        self.provider = provider
        self.config = HarnessConfig(name="default", provider=ProviderConfig(kind="mock"))

    def _apply_config(self, request: LMRequest) -> LMRequest:
        return request

    def flush_metrics(self, batch_hash: str) -> None:
        return None


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


async def _run_batched(
    prompts: list[str], per_request: bool, wait_for: int
) -> tuple[list[str], _SlowProvider, list[ResultItem]]:
    """Run one batch through the batched strategy; return what arrived before "slow" ended."""
    from olmo_eval.runners.asynq.batching import BatchConfig, get_strategy

    provider = _SlowProvider()
    results: queue.Queue[ResultItem] = queue.Queue()
    strategy = get_strategy(
        BatchConfig(chunk_size=len(prompts), chunk_timeout=0.05), per_request=per_request
    )
    run = asyncio.create_task(
        strategy.run(
            _queued(prompts),  # type: ignore[arg-type]
            _FakeHarness(provider),  # type: ignore[arg-type]
            results,  # type: ignore[arg-type]
            None,
            logging.getLogger(__name__),
            len(prompts),
        )
    )
    deadline = time.monotonic() + 1.0
    while results.qsize() < wait_for and time.monotonic() < deadline and not run.done():
        await asyncio.sleep(0.01)
    early = [str(results.get_nowait().request.prompt) for _ in range(results.qsize())]
    provider.release.set()
    try:
        await run
    finally:
        rest = [results.get_nowait() for _ in range(results.qsize())]
    return early, provider, rest


class TestPerRequestResults:
    def test_a_slow_request_does_not_hold_back_its_batch(self) -> None:
        early, provider, rest = asyncio.run(_run_batched(["slow", "a", "b", "c"], True, 3))

        # Each finished request reached the result queue while "slow" was still running.
        assert sorted(early) == ["a", "b", "c"]
        assert [result.request.prompt for result in rest] == ["slow"]
        assert sorted(provider.calls) == [["a"], ["b"], ["c"], ["slow"]]
        assert all(result.error is None for result in rest)

    def test_without_progress_a_batch_returns_whole(self) -> None:
        early, provider, rest = asyncio.run(_run_batched(["slow", "a", "b", "c"], False, 3))

        assert early == []
        assert provider.calls == [["slow", "a", "b", "c"]]
        assert sorted(str(result.request.prompt) for result in rest) == ["a", "b", "c", "slow"]

    def test_a_terminal_error_still_stops_the_worker(self) -> None:
        from olmo_eval.inference.errors import TerminalProviderError

        with pytest.raises(TerminalProviderError, match="engine died"):
            asyncio.run(_run_batched(["slow", "dead", "a"], True, 3))

    def test_only_providers_that_dispatch_one_by_one_split_batches(self) -> None:
        from olmo_eval.runners.asynq.batching import BatchConfig

        split = {kind for kind in ("vllm_server", "litellm", "mock")}
        engines = {"vllm", "sglang", "hf", "olmo_core", "olmo_core_vlm"}
        assert all(BatchConfig.dispatches_per_request(kind) for kind in split)
        assert not any(BatchConfig.dispatches_per_request(kind) for kind in engines)

    @pytest.mark.parametrize("record_progress", [True, False])
    def test_the_worker_asks_for_per_request_results_only_with_progress(
        self, monkeypatch, record_progress
    ) -> None:
        from olmo_eval.runners.asynq import batching

        chosen: list[bool] = []
        real = batching.get_strategy

        def spy(*args: Any, **kwargs: Any) -> Any:
            chosen.append(kwargs["per_request"])
            return real(*args, **kwargs)

        monkeypatch.setattr(batching, "get_strategy", spy)
        items: queue.Queue[QueueItem | None] = queue.Queue()
        items.put(None)
        # The real worker, in this process; the module fixture replaced the spawned one.
        real_inference_worker(
            "worker-0",
            [],
            items,  # type: ignore[arg-type]
            queue.Queue(),  # type: ignore[arg-type]
            _harness().to_dict(),
            0,
            record_progress=record_progress,
        )

        assert chosen == [record_progress]

    @pytest.mark.parametrize("resume", [True, False])
    def test_the_runner_tells_workers_whether_it_records_progress(
        self, tmp_path, monkeypatch, resume
    ) -> None:
        from olmo_eval.inference.provider_manager import ProviderManager

        sent: list[Any] = []

        class Context:
            def Process(self, target: Any, args: tuple[Any, ...]) -> Any:
                from inspect import signature

                parameters = signature(real_inference_worker).parameters
                sent.append(dict(zip(parameters, args, strict=False)))
                raise RuntimeError("stop before starting workers")

        manager = ProviderManager(
            harness_config=_harness(),
            num_inference_workers=1,
            gpu_ids=[],
            item_queue=queue.Queue(),  # type: ignore[arg-type]
            result_queue=queue.Queue(),  # type: ignore[arg-type]
            record_progress=resume,
        )
        with pytest.raises(RuntimeError, match="stop before"):
            manager.start(Context(), 0, queue.Queue())  # type: ignore[arg-type]
        assert sent[0]["record_progress"] is resume

        made: list[bool] = []
        real_init = ProviderManager.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            made.append(kwargs.get("record_progress", False))
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(ProviderManager, "__init__", init)
        run_eval(tmp_path, "segments", {"gsm8k": 2}, resume=resume)
        assert made == [resume]

    def test_the_batching_strategy_is_not_part_of_what_must_match(self, tmp_path) -> None:
        from olmo_eval.runners.asynq.batching import BatchConfig

        trackers, items = _local_setup(undecidable=set())
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        _process(trackers, restored.remaining, progress, restored.responses)
        progress.close()
        _drop_final(tmp_path)

        streaming = replace(_harness(), batching=BatchConfig.streaming())
        trackers, items = _local_setup(undecidable=set())
        progress = open_run_progress(str(tmp_path), "mock", streaming)
        restored = progress.restore(trackers, items, streaming)
        progress.close()

        assert sorted(restored.responses["local"]) == [0, 1, 2, 3]

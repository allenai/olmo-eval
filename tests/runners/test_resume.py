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
import queue
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
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
    ProgressMismatchError,
    RunProgress,
    decode_value,
    encode_value,
    open_run_progress,
    parse_exit_thresholds,
    store_class,
)
from olmo_eval.runners.asynq.results import process_results
from olmo_eval.runners.asynq.runner import AsyncEvalRunner
from olmo_eval.runners.asynq.types import QueueItem, ResultItem, TaskTracker
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


def _harness(**kwargs: Any) -> HarnessConfig:
    # HashProvider fails some requests on purpose; keep every task's metrics.
    return HarnessConfig(
        name="default",
        provider=ProviderConfig(kind="mock", model="mock", **kwargs),
        max_hard_failure_rate=1.0,
    )


def _runner(out: Path, limits: dict[str, int] = TASK_LIMITS, **kwargs: Any) -> AsyncEvalRunner:
    return AsyncEvalRunner(
        harness_config=_harness(),
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
    **kwargs: Any,
) -> RunStats:
    """Run the evaluation into ``out`` under the CLI's SIGTERM handling."""
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
                _runner(out, limits, **kwargs).run()
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
        return {k: _without_timing(v) for k, v in value.items() if k not in _TIMING_KEYS}
    if isinstance(value, list):
        return [_without_timing(v) for v in value]
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
    assert len(reference_files) == 2 * len(TASK_LIMITS)
    for name, content in reference_files.items():
        assert resumed_files[name] == content, f"{name} differs"
    assert _metrics(resumed) == _metrics(reference)


def _stores(out: Path, layout: str) -> list[Any]:
    cls = store_class(layout)
    paths = sorted((out / PROGRESS_DIR).glob(f"*/*{cls.suffix}"))
    return [cls(path.with_name(path.name[: -len(cls.suffix)])) for path in paths]


def _instance_entries(store: Any) -> list[dict[str, Any]]:
    return [entry for entry in store.read() if "idx" in entry]


def _recorded(out: Path, layout: str) -> int:
    return sum(len(_instance_entries(store)) for store in _stores(out, layout))


def keep_first_entries(out: Path, layout: str, keep: int) -> int:
    """Cut the records back to ``keep`` instance entries in all, as an interrupted run leaves.

    Finalized results go too. Returns the number of entries kept.
    """
    for final in (out / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"):
        final.unlink()
    kept_total = 0
    for store in _stores(out, layout):
        header, *entries = store.read()
        store.reset(header)
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
        mp.setattr(runner_module, "discard_progress", lambda output_dir: None)
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
        stores = _stores(full, layout)
        assert len(stores) == len(TASK_LIMITS)
        assert len(list((full / PROGRESS_DIR).glob(f"*/*{FINAL_SUFFIX}"))) == len(TASK_LIMITS)
        entries = [_instance_entries(store) for store in stores]
        failed = [e for rest in entries for e in rest if "failed" in e]
        assert failed and all(e["failed"] == resume_fakes.FAILED_REQUEST for e in failed)
        for store, rest in zip(stores, entries, strict=True):
            header = next(iter(store.read()))
            assert header["task_hash"] and header["model_hash"] and header["harness_hash"]
            assert "execution" in header
            assert len({entry["idx"] for entry in rest}) == len(rest)
            assert "<<say:" not in json.dumps(rest)
        # Instances whose judge was unavailable are not final, so they are not recorded.
        assert _recorded(full, layout) < TOTAL

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

    def test_sigterm_then_resume(self, full_run, tmp_path) -> None:
        full, layout = full_run
        out = tmp_path / "run"

        with pytest.raises(Terminated):
            run_eval(out, layout, env={DEBUG_EXIT_AT_ENV: "0.37"})
        recorded = _recorded(out, layout)
        assert recorded >= 0.37 * TOTAL
        first = read_manifest(out)
        assert first is not None and first["run"]["status"] == "running"
        secret = run_secret(out)

        # Without --resume the directory is refused rather than mixed into.
        with pytest.raises(ProgressMismatchError, match="Pass --resume"):
            run_eval(out, layout)

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

        stats = run_eval(out, layout, fresh=True)

        assert stats.queued == TOTAL and stats.restored == 0
        assert_same_results(out, full)


class TestStores:
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_a_torn_last_line_is_dropped(self, tmp_path, layout) -> None:
        store = store_class(layout)(tmp_path / "task_abcdef")
        store.reset({"header": True})
        store.append({"idx": 0})
        store.close()
        torn = b'{"idx": 1, "outputs": [{"te'
        if layout == "append":
            with open(store.path, "ab") as f:
                f.write(torn)
        else:
            (store.path / "000099.jsonl").write_bytes(b'{"idx": 1}\n' + torn)

        reopened = store_class(layout)(tmp_path / "task_abcdef")
        expected = [{"header": True}, {"idx": 0}] + ([] if layout == "append" else [{"idx": 1}])
        assert list(reopened.read()) == expected
        reopened.append({"idx": 2})
        reopened.close()
        assert list(store_class(layout)(tmp_path / "task_abcdef").read()) == [
            *expected,
            {"idx": 2},
        ]

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
    """Feed one result per item to process_results; ``failures`` maps idx -> retryable."""
    failures = failures or {}
    results: queue.Queue[ResultItem] = queue.Queue()
    for item in items:
        failed = item.instance_idx in failures
        results.put(
            ResultItem(
                model_name="mock",
                task_id=item.task_id,
                instance_idx=item.instance_idx,
                instance=item.instance,
                request=item.request,
                outputs=[] if failed else [LMOutput(text="a", metadata={"prompt_tokens": 3})],
                error="provider failed" if failed else None,
                retryable=failures.get(item.instance_idx, False),
            )
        )
    return asyncio.run(
        process_results(
            trackers=trackers,
            result_queue=results,  # type: ignore[arg-type]
            workers=[],
            scoring_context=None,  # type: ignore[arg-type]
            scoring_concurrency=2,
            total_tasks=len(trackers),
            total_instances=len(items),
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
        # 0: a request failure (final); 1: a failed call (retried); 2: undecidable
        # (final, unscored); 3: grader unreachable (retried).
        trackers, items = _local_setup(undecidable={2}, unreachable=frozenset({3}))
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())
        first = _process(
            trackers, restored.remaining, progress, restored.responses, {0: False, 1: True}
        )
        progress.close()
        _drop_final(tmp_path)

        trackers, items = _local_setup(undecidable={2}, unreachable=frozenset({3}))
        progress = _progress(tmp_path)
        restored = progress.restore(trackers, items, _harness())

        assert sorted(restored.responses["local"]) == [2]
        assert trackers["local"].failed_instances == {0: "provider failed"}
        assert [item.instance_idx for item in restored.remaining] == [1, 3]
        second = _process(trackers, restored.remaining, progress, restored.responses, {1: True})
        progress.close()
        assert second["local"].metrics == first["local"].metrics
        assert second["local"].instances_failed == first["local"].instances_failed == 2
        assert second["local"].scoring_failures == first["local"].scoring_failures

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

    def test_a_resumed_run_keeps_its_id(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("BEAKER_JOB_ID", "job-1")
        run_id = self._write_manifest(tmp_path, "failed", ["gsm8k"])
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

"""Tests for the replay provider that serves generations already saved to disk."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.common.types import LMRequest, RequestType, SamplingParams
from olmo_eval.inference.providers.replay import (
    ReplayCoverageError,
    ReplayInputError,
    StoredPredictionsProvider,
)

MODEL_DIR = "test-model"
TASK = "demo_task"


def chat_request(content: str) -> LMRequest:
    """Build the chat request a task would hand to the provider."""
    return LMRequest(
        request_type=RequestType.CHAT,
        messages=({"role": "user", "content": content},),
    )


def write_results_dir(
    root: Path,
    rows: list[tuple[Any, Any, Any, Any]],
    *,
    task: str = TASK,
    model_dir: str = MODEL_DIR,
    include_prediction: bool = True,
) -> Path:
    """Write a minimal results directory.

    Args:
        root: Directory that will hold `requests/` and `predictions/`.
        rows: (doc_id, native_id, request_context, model_output) tuples. A
            `model_output` of None omits the prediction row entirely.
        task: Task name used in the filenames.
        model_dir: Model subdirectory name.
        include_prediction: When False, no predictions file is written at all.
    """
    req_dir = root / "requests" / model_dir
    pred_dir = root / "predictions" / model_dir
    req_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    request_lines = []
    prediction_lines = []
    for doc_id, native_id, context, model_output in rows:
        request_lines.append(
            json.dumps(
                {
                    "request_type": "generate_until",
                    "doc": {"query": "q"},
                    "request": {"context": context, "stop_sequences": [], "generation_kwargs": {}},
                    "idx": 0,
                    "task_name": task,
                    "doc_id": doc_id,
                    "native_id": native_id,
                    "label": None,
                }
            )
        )
        if model_output is None:
            continue
        prediction_lines.append(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "native_id": native_id,
                    "model_output": model_output,
                    "instance_metrics": {},
                    "label": None,
                }
            )
        )

    (req_dir / f"{task}-requests.jsonl").write_text("\n".join(request_lines) + "\n")
    if include_prediction:
        text = "\n".join(prediction_lines)
        (pred_dir / f"{task}-predictions.jsonl").write_text(text + "\n" if text else "")
    return root


def simple_rows() -> list[tuple[Any, Any, Any, Any]]:
    """Three chat instances with distinct prompts and distinct answers."""
    return [
        (
            i,
            i + 1,
            [{"role": "user", "content": f"question {i}"}],
            [{"text": f"answer {i}", "extracted_answer": f"answer {i}", "num_chars": 8}],
        )
        for i in range(3)
    ]


class TestExactMatch:
    def test_replays_stored_text_for_every_request(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        requests = [chat_request(f"question {i}") for i in range(3)]
        outputs = provider.generate(requests)

        assert [out[0].text for out in outputs] == ["answer 0", "answer 1", "answer 2"]
        assert all(len(out) == 1 for out in outputs)

    def test_out_of_order_requests_get_their_own_answer(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        outputs = provider.generate([chat_request("question 2"), chat_request("question 0")])

        assert [out[0].text for out in outputs] == ["answer 2", "answer 0"]

    def test_key_ignores_tools_and_sampling_params(self, tmp_path: Path) -> None:
        """Tools and generation settings describe how to generate, not which instance."""
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        request = LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": "question 1"},),
            max_length=4096,
        )
        outputs = provider.generate([request], SamplingParams(temperature=0.7, max_tokens=99))

        assert outputs[0][0].text == "answer 1"

    def test_message_key_is_order_and_content_sensitive(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError):
            provider.generate([chat_request("question 1 ")])  # trailing space

    def test_completion_style_prompt_round_trips(self, tmp_path: Path) -> None:
        write_results_dir(
            tmp_path,
            [(0, 1, "raw prompt text", [{"text": "raw answer"}])],
        )
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        request = LMRequest(request_type=RequestType.COMPLETION, prompt="raw prompt text")
        assert provider.generate([request])[0][0].text == "raw answer"

    def test_prompt_key_does_not_collide_with_chat_key(self, tmp_path: Path) -> None:
        """A prompt string must never match a chat conversation."""
        write_results_dir(tmp_path, [(0, 1, "hello", [{"text": "text answer"}])])
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError):
            provider.generate([chat_request("hello")])

    def test_metadata_carries_provenance_and_no_logprob_fields(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        output = provider.generate([chat_request("question 1")])[0][0]

        assert output.metadata["replay_source"]["doc_id"] == 1
        assert output.metadata["replay_source"]["native_id"] == 2
        assert output.logprobs is None
        # Absent, not zero: downstream treats missing logprob fields as "unavailable".
        assert "sum_logits" not in output.metadata
        assert "num_tokens" not in output.metadata
        # The task recomputes extracted_answer during scoring.
        assert output.extracted_answer is None


class TestMissingEntry:
    def test_unknown_request_raises(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError) as excinfo:
            provider.generate([chat_request("question 0"), chat_request("never generated")])

        message = str(excinfo.value)
        assert "1 of 2" in message
        assert "no saved request matches this content" in message

    def test_saved_request_without_prediction_is_named(self, tmp_path: Path) -> None:
        """A known instance with no prediction reports its identity, not just 'unknown'."""
        rows = simple_rows()
        rows[1] = (1, 2, [{"role": "user", "content": "question 1"}], None)
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError) as excinfo:
            provider.generate([chat_request("question 1")])

        message = str(excinfo.value)
        assert "doc_id=1" in message
        assert "no prediction row with doc_id=1" in message

    def test_null_text_is_missing_not_empty(self, tmp_path: Path) -> None:
        rows = simple_rows()
        rows[1] = (1, 2, [{"role": "user", "content": "question 1"}], [{"text": None}])
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError) as excinfo:
            provider.generate([chat_request("question 1")])

        assert "is null (no generation recorded)" in str(excinfo.value)

    def test_empty_model_output_list_is_missing(self, tmp_path: Path) -> None:
        rows = simple_rows()
        rows[1] = (1, 2, [{"role": "user", "content": "question 1"}], [])
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError) as excinfo:
            provider.generate([chat_request("question 1")])

        assert "empty or missing 'model_output' list" in str(excinfo.value)

    def test_missing_request_does_not_poison_the_others(self, tmp_path: Path) -> None:
        """The whole batch fails; no partial result is handed back."""
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError):
            provider.generate([chat_request("question 0"), chat_request("missing")])

        assert provider.coverage_report()["missing"] == 1
        assert provider.coverage_report()["matched"] == 1


class TestEmptyButPresent:
    def test_empty_string_prediction_is_replayed_as_empty(self, tmp_path: Path) -> None:
        rows = simple_rows()
        rows[1] = (1, 2, [{"role": "user", "content": "question 1"}], [{"text": ""}])
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        outputs = provider.generate([chat_request("question 1")])

        assert outputs[0][0].text == ""
        assert provider.coverage_report()["missing"] == 0
        assert provider.coverage_report()["matched"] == 1
        assert provider.coverage_report()["empty_text_replayed"] == 1

    def test_empty_and_missing_are_reported_differently(self, tmp_path: Path) -> None:
        rows = [
            (0, 1, [{"role": "user", "content": "empty one"}], [{"text": ""}]),
            (1, 2, [{"role": "user", "content": "absent one"}], None),
        ]
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert provider.generate([chat_request("empty one")])[0][0].text == ""
        with pytest.raises(ReplayCoverageError):
            provider.generate([chat_request("absent one")])


class TestAmbiguousInput:
    def test_two_prediction_files_raise(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        extra = tmp_path / "predictions" / MODEL_DIR / "other_task-predictions.jsonl"
        extra.write_text("")

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "Ambiguous predictions input" in str(excinfo.value)
        assert "other_task-predictions.jsonl" in str(excinfo.value)

    def test_two_request_files_raise(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        extra = tmp_path / "requests" / "another-model" / f"{TASK}-requests.jsonl"
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text("")

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "Ambiguous requests input" in str(excinfo.value)

    def test_task_filter_disambiguates(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        write_results_dir(
            tmp_path,
            [(0, 1, [{"role": "user", "content": "other"}], [{"text": "other answer"}])],
            task="other_task",
        )

        provider = StoredPredictionsProvider(
            "stored", results_dir=str(tmp_path), task_filter="other_task"
        )
        assert provider.generate([chat_request("other")])[0][0].text == "other answer"

    def test_explicit_files_bypass_globbing(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        write_results_dir(tmp_path, simple_rows(), task="other_task")

        provider = StoredPredictionsProvider(
            "stored",
            results_dir=str(tmp_path),
            requests_file=f"requests/{MODEL_DIR}/{TASK}-requests.jsonl",
            predictions_file=f"predictions/{MODEL_DIR}/{TASK}-predictions.jsonl",
        )
        assert provider.generate([chat_request("question 0")])[0][0].text == "answer 0"

    def test_no_matching_file_raises(self, tmp_path: Path) -> None:
        (tmp_path / "requests").mkdir()
        (tmp_path / "predictions").mkdir()

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "No *-requests.jsonl file found" in str(excinfo.value)

    def test_missing_results_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ReplayInputError):
            StoredPredictionsProvider("stored", results_dir=str(tmp_path / "nope"))

    def test_duplicate_doc_id_in_predictions_raises(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        path = tmp_path / "predictions" / MODEL_DIR / f"{TASK}-predictions.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([*lines, lines[0]]) + "\n")

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "repeats doc_id=0" in str(excinfo.value)

    def test_identical_requests_with_different_text_raise(self, tmp_path: Path) -> None:
        """Two instances that look identical cannot be told apart; refuse to guess."""
        rows = [
            (0, 1, [{"role": "user", "content": "same"}], [{"text": "answer A"}]),
            (1, 2, [{"role": "user", "content": "same"}], [{"text": "answer B"}]),
        ]
        write_results_dir(tmp_path, rows)

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "different stored" in str(excinfo.value)

    def test_identical_requests_with_identical_text_are_accepted(self, tmp_path: Path) -> None:
        rows = [
            (0, 1, [{"role": "user", "content": "same"}], [{"text": "one answer"}]),
            (1, 2, [{"role": "user", "content": "same"}], [{"text": "one answer"}]),
        ]
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert provider.generate([chat_request("same")])[0][0].text == "one answer"

    def test_native_id_disagreement_raises(self, tmp_path: Path) -> None:
        """doc_id joins the files; native_id is the witness that they match."""
        write_results_dir(tmp_path, simple_rows())
        path = tmp_path / "predictions" / MODEL_DIR / f"{TASK}-predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["native_id"] = 999
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "do not describe the same run" in str(excinfo.value)

    def test_request_without_context_raises(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        path = tmp_path / "requests" / MODEL_DIR / f"{TASK}-requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        del rows[0]["request"]["context"]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        with pytest.raises(ReplayInputError) as excinfo:
            StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert "has no request.context" in str(excinfo.value)


class TestCoverageReporting:
    def test_load_time_counts(self, tmp_path: Path) -> None:
        rows = simple_rows()
        rows[2] = (2, 3, [{"role": "user", "content": "question 2"}], None)
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        report = provider.coverage_report()
        assert report["stored_requests"] == 3
        assert report["stored_predictions"] == 2
        assert report["replayable_entries"] == 2
        assert report["unusable_entries"] == 1
        assert report["requested"] == 0

    def test_counts_accumulate_across_calls(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        provider.generate([chat_request("question 0")])
        provider.generate([chat_request("question 1"), chat_request("question 2")])

        report = provider.coverage_report()
        assert report["requested"] == 3
        assert report["matched"] == 3
        assert report["missing"] == 0
        assert report["unused_entries"] == 0

    def test_has_stored_prediction_probes_without_raising(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        assert provider.has_stored_prediction(chat_request("question 0"))
        assert not provider.has_stored_prediction(chat_request("never generated"))

    def test_unused_entries_are_visible(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        provider.generate([chat_request("question 0")])

        assert provider.coverage_report()["unused_entries"] == 2

    def test_batch_coverage_is_logged(self, tmp_path: Path, caplog: Any) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with caplog.at_level("INFO"):
            provider.generate([chat_request("question 0")])

        assert any("Replay coverage" in record.message for record in caplog.records)


class TestSamplesAndLogprobs:
    def test_logprobs_raises(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(NotImplementedError, match="no logprobs"):
            provider.logprobs([chat_request("question 0")])

    @pytest.mark.anyio
    async def test_alogprobs_raises(self, tmp_path: Path, anyio_backend: str) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(NotImplementedError):
            await provider.alogprobs([chat_request("question 0")])

    @pytest.mark.anyio
    async def test_agenerate_matches_generate(self, tmp_path: Path, anyio_backend: str) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        outputs = await provider.agenerate([chat_request("question 0")])
        assert outputs[0][0].text == "answer 0"

    def test_more_samples_than_stored_raises(self, tmp_path: Path) -> None:
        write_results_dir(tmp_path, simple_rows())
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        with pytest.raises(ReplayCoverageError, match="never fabricates"):
            provider.generate([chat_request("question 0")], SamplingParams(num_samples=2))

    def test_multiple_stored_samples_are_replayed(self, tmp_path: Path) -> None:
        rows = [
            (
                0,
                1,
                [{"role": "user", "content": "q"}],
                [{"text": "sample A"}, {"text": "sample B"}],
            )
        ]
        write_results_dir(tmp_path, rows)
        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))

        outputs = provider.generate([chat_request("q")], SamplingParams(num_samples=2))
        assert [o.text for o in outputs[0]] == ["sample A", "sample B"]


class TestTrajectoryPreservation:
    def test_trajectory_is_reattached_to_metadata(self, tmp_path: Path) -> None:
        """The runner rebuilds Response.trajectory from metadata['trajectory']."""
        req_dir = tmp_path / "requests" / MODEL_DIR
        pred_dir = tmp_path / "predictions" / MODEL_DIR
        req_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        trajectory = {"turns": [{"role": "assistant", "content": "hi"}], "final_answer": "answer"}
        (req_dir / f"{TASK}-requests.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": 0,
                    "native_id": 1,
                    "request": {"context": [{"role": "user", "content": "q"}]},
                }
            )
            + "\n"
        )
        (pred_dir / f"{TASK}-predictions.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": 0,
                    "native_id": 1,
                    "model_output": [{"text": "answer"}],
                    "trajectory": trajectory,
                }
            )
            + "\n"
        )

        provider = StoredPredictionsProvider("stored", results_dir=str(tmp_path))
        output = provider.generate([chat_request("q")])[0][0]
        assert output.metadata["trajectory"] == trajectory

        off = StoredPredictionsProvider(
            "stored", results_dir=str(tmp_path), preserve_trajectory=False
        )
        assert "trajectory" not in off.generate([chat_request("q")])[0][0].metadata


class TestProviderWiring:
    def test_create_provider_python_kind(self, tmp_path: Path) -> None:
        """The provider is reachable through provider.kind=python + kwargs.class."""
        from olmo_eval.inference import create_provider

        write_results_dir(tmp_path, simple_rows())
        provider = create_provider(
            "python",
            "stored",
            **{
                "class": "olmo_eval.inference.providers.replay.StoredPredictionsProvider",
                "results_dir": str(tmp_path),
            },
        )
        assert isinstance(provider, StoredPredictionsProvider)
        assert provider.generate([chat_request("question 0")])[0][0].text == "answer 0"


# ─────────────────────────────────────────────────────────────
# Real data: DeepResearch Bench predictions saved on this host.
# ─────────────────────────────────────────────────────────────

DRB_EVIDENCE = Path(os.environ.get("DRB_EVIDENCE_DIR", "/tmp/drb-evidence"))
DRB_PREDICTIONS = [
    DRB_EVIDENCE / "fixed-drb-predictions.jsonl",
    DRB_EVIDENCE / "agent_disco-drb-predictions.jsonl",
]


def _real_drb_task_and_requests() -> tuple[Any, list[Any], list[LMRequest]]:
    """Load the real DeepResearch Bench task, its instances, and its live requests."""
    from olmo_eval.evals.tasks.common import get_task

    task = get_task("deepresearch_bench")
    instances = list(task.instances)
    return task, instances, [task.format_request(instance) for instance in instances]


def _build_real_drb_results_dir(tmp_path: Path, predictions_path: Path) -> Path:
    """Pair real saved DRB predictions with real DRB requests.

    The requests file is produced by the same `build_requests()` code the runner uses,
    over the real DeepResearch Bench instances, so `request.context` holds exactly what
    the original run recorded. The predictions come from disk verbatim.
    """
    from olmo_eval.runners.io.builders import build_requests

    task, instances, requests = _real_drb_task_and_requests()
    rows = build_requests(instances, requests, "deepresearch_bench", task.config.sampling_params)

    req_dir = tmp_path / "requests" / "real-model"
    pred_dir = tmp_path / "predictions" / "real-model"
    req_dir.mkdir(parents=True)
    pred_dir.mkdir(parents=True)
    (req_dir / "deepresearch_bench-requests.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (pred_dir / "deepresearch_bench-predictions.jsonl").write_bytes(predictions_path.read_bytes())
    return tmp_path


@pytest.mark.skipif(
    not all(p.exists() for p in DRB_PREDICTIONS),
    reason="real DeepResearch Bench predictions are not present on this host",
)
@pytest.mark.parametrize("predictions_path", DRB_PREDICTIONS, ids=lambda p: p.stem)
def test_real_deepresearch_bench_round_trip(tmp_path: Path, predictions_path: Path) -> None:
    """Every instance must come back with byte-identical stored text."""
    root = _build_real_drb_results_dir(tmp_path, predictions_path)
    provider = StoredPredictionsProvider("stored", results_dir=str(root))

    _, _, live_requests = _real_drb_task_and_requests()
    stored = {
        row["doc_id"]: row
        for row in (
            json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()
        )
    }

    outputs = provider.generate(live_requests)

    assert len(outputs) == len(live_requests)
    assert len(outputs) == len(stored)
    for doc_id, out in enumerate(outputs):
        assert out[0].text == stored[doc_id]["model_output"][0]["text"], f"doc_id={doc_id}"

    report = provider.coverage_report()
    assert report["missing"] == 0
    assert report["matched"] == len(live_requests)
    assert report["unused_entries"] == 0


@pytest.mark.skipif(
    not DRB_PREDICTIONS[0].exists(),
    reason="real DeepResearch Bench predictions are not present on this host",
)
def test_real_empty_predictions_replay_as_empty(tmp_path: Path) -> None:
    """The real `fixed` run contains genuinely empty reports; they must stay empty."""
    predictions_path = DRB_PREDICTIONS[0]
    root = _build_real_drb_results_dir(tmp_path, predictions_path)
    provider = StoredPredictionsProvider("stored", results_dir=str(root))

    _, _, live_requests = _real_drb_task_and_requests()
    stored_texts = [
        json.loads(line)["model_output"][0]["text"]
        for line in predictions_path.read_text().splitlines()
        if line.strip()
    ]
    expected_empty = sum(1 for text in stored_texts if text == "")
    assert expected_empty > 0, "fixture assumption: this run has empty reports"

    outputs = provider.generate(live_requests)

    assert sum(1 for out in outputs if out[0].text == "") == expected_empty
    assert provider.coverage_report()["empty_text_replayed"] == expected_empty
    assert provider.coverage_report()["missing"] == 0


@pytest.mark.skipif(
    not DRB_PREDICTIONS[0].exists(),
    reason="real DeepResearch Bench predictions are not present on this host",
)
def test_real_predictions_with_one_instance_dropped_fail_loudly(tmp_path: Path) -> None:
    """Deleting one real prediction row must surface as a named, loud failure."""
    predictions_path = DRB_PREDICTIONS[0]
    root = _build_real_drb_results_dir(tmp_path, predictions_path)
    pred_file = root / "predictions" / "real-model" / "deepresearch_bench-predictions.jsonl"
    lines = [line for line in pred_file.read_text().splitlines() if line.strip()]
    dropped = json.loads(lines[3])
    pred_file.write_text("\n".join(lines[:3] + lines[4:]) + "\n")

    provider = StoredPredictionsProvider("stored", results_dir=str(root))
    _, _, live_requests = _real_drb_task_and_requests()

    with pytest.raises(ReplayCoverageError) as excinfo:
        provider.generate(live_requests)

    message = str(excinfo.value)
    assert "1 of 100" in message
    assert f"doc_id={dropped['doc_id']}" in message

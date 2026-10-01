"""Tests for rebuilding saveable results from an output directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from olmo_eval.cli.results.ingest import local_reader
from olmo_eval.common.types import compute_model_hash
from olmo_eval.runners.io.writers import write_predictions_jsonl
from olmo_eval.runners.processing.metrics import write_metrics_json
from olmo_eval.storage.base import convert_runner_results
from olmo_eval.storage.ingest import load_output_dir

PROVIDER = {"kind": "vllm", "model": "allenai/Olmo-3-7B", "revision": "step100"}


def _runner_results() -> dict[str, Any]:
    return {
        "model": "olmo3-7b",
        "model_path": PROVIDER["model"],
        "provider": "vllm",
        "model_config": dict(PROVIDER),
        "harness_config": {"name": "default", "provider": dict(PROVIDER)},
        "timestamp": "2026-10-01T12:00:00",
        "errors": [{"task": "mbpp", "error": "hard-failure budget exceeded"}],
        "tasks": {
            "gsm8k": {
                "metrics": {"exact_match": {"exact_match": 0.5}},
                "task_hash": "gsm8k-hash-abcdef",
                "config": {"name": "gsm8k"},
                "num_instances": 2,
                "primary_metric": "exact_match:exact_match",
                "duration_seconds": 3.0,
                "instances_processed": 2,
                "instances_failed": 0,
                "predictions": [
                    {"native_id": "q1", "instance_metrics": {"exact_match": {"exact_match": 1}}},
                    {"native_id": "q2", "instance_metrics": {"exact_match": {"exact_match": 0}}},
                ],
            },
            "mbpp": {
                "num_instances": 4,
                "task_hash": "mbpp-hash-123456",
                "error": "hard-failure budget exceeded",
                "error_summary": "3 instances timed out",
                "instances_processed": 4,
                "instances_failed": 3,
            },
        },
    }


def _write_run(output_dir: Path, results: dict[str, Any]) -> None:
    write_metrics_json(
        output_dir=str(output_dir),
        results=results,
        experiment_id="exp-1",
        experiment_name="nightly",
        experiment_group="nightly-group",
        model_hash="model-hash-1",
        experiment_duration_seconds=12.0,
    )
    for spec, task in results["tasks"].items():
        if task.get("predictions"):
            write_predictions_jsonl(
                str(output_dir), spec, task["predictions"], results["model"], task["task_hash"]
            )


def test_round_trip_matches_what_the_runner_would_have_saved(tmp_path: Path) -> None:
    original = _runner_results()
    _write_run(tmp_path, original)

    loaded = load_output_dir(str(tmp_path), local_reader())

    assert loaded.experiment_id == "exp-1"
    assert loaded.model_hash == "model-hash-1"
    assert loaded.experiment_name == "nightly"
    assert loaded.experiment_group == "nightly-group"
    assert loaded.experiment_duration_seconds == 12.0
    assert loaded.missing_predictions == []

    expected = convert_runner_results(original, "exp-1", model_hash="model-hash-1")
    actual = convert_runner_results(loaded.results, "exp-1", model_hash="model-hash-1")
    assert actual.model_name == "olmo3-7b"
    assert actual.backend_name == expected.backend_name
    assert actual.timestamp == expected.timestamp
    assert actual.tasks == expected.tasks
    assert loaded.results["model_path"] == PROVIDER["model"]
    assert loaded.results["model_config"] == PROVIDER
    assert (
        loaded.results["tasks"]["gsm8k"]["predictions"] == original["tasks"]["gsm8k"]["predictions"]
    )


def test_model_name_option_overrides_the_recorded_name(tmp_path: Path) -> None:
    _write_run(tmp_path, _runner_results())

    loaded = load_output_dir(
        str(tmp_path), local_reader(), model_name="other", include_predictions=False
    )

    assert loaded.results["model"] == "other"
    assert "predictions" not in loaded.results["tasks"]["gsm8k"]


def test_falls_back_to_model_path_without_a_recorded_name(tmp_path: Path) -> None:
    _write_run(tmp_path, _runner_results())
    metrics_file = tmp_path / "metrics.json"
    metrics = json.loads(metrics_file.read_text())
    del metrics["model_name"]
    metrics_file.write_text(json.dumps(metrics))

    loaded = load_output_dir(str(tmp_path), local_reader())

    assert loaded.results["model"] == PROVIDER["model"]
    # Predictions were written under the alias, so they cannot be found by path.
    assert loaded.missing_predictions == ["gsm8k"]


def test_flat_config_without_hash_recomputes_the_model_hash(tmp_path: Path) -> None:
    config = {"kind": "litellm", "model": "gpt-4o"}
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-10-01T12:00:00",
                "config": config,
                "tasks": [{"task": "tau2", "metrics": {}, "num_instances": 0, "task_hash": "t"}],
                "summary": {},
                "experiment_id": "exp-2",
            }
        )
    )

    loaded = load_output_dir(str(tmp_path), local_reader())

    assert loaded.model_hash == compute_model_hash(config)
    assert loaded.results["model"] == "gpt-4o"
    assert loaded.results["provider"] == "litellm"
    assert loaded.missing_predictions == ["tau2"]


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (None, "No metrics.json"),
        ({"timestamp": "2026-10-01T12:00:00", "config": {}}, "no experiment_id"),
        ({"experiment_id": "e", "config": {}}, "no timestamp"),
        (
            {"experiment_id": "e", "timestamp": "2026-10-01T12:00:00", "config": {"models": {}}},
            "Multi-model",
        ),
        ({"experiment_id": "e", "timestamp": "2026-10-01T12:00:00", "config": {}}, "model name"),
    ],
)
def test_rejects_output_that_cannot_identify_the_run(
    tmp_path: Path, metrics: dict[str, Any] | None, message: str
) -> None:
    if metrics is not None:
        (tmp_path / "metrics.json").write_text(json.dumps(metrics))

    with pytest.raises(ValueError, match=message):
        load_output_dir(str(tmp_path), local_reader())

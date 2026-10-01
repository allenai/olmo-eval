"""Tests for saving runner results to storage backends."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from olmo_eval.runners.io.storage import ResultsSaveError, save_results


def _results(**task_overrides: Any) -> dict[str, Any]:
    task = {
        "metrics": {"accuracy": {"exact_match": 0.5}},
        "task_hash": "hash-1",
        "num_instances": 2,
        "primary_metric": "accuracy:exact_match",
        "predictions": [
            {"native_id": "a", "instance_metrics": {"accuracy": {"exact_match": 1.0}}},
            {"native_id": "b", "instance_metrics": {"accuracy": {"exact_match": 0.0}}},
        ],
    }
    task.update(task_overrides)
    return {
        "model": "model-a",
        "provider": "vllm",
        "timestamp": "2026-10-01T12:00:00",
        "model_config": {"kind": "vllm", "model": "org/model-a"},
        "tasks": {"gsm8k": task},
    }


def test_save_results_raises_when_a_backend_fails() -> None:
    failing = MagicMock()
    failing.save.side_effect = RuntimeError("connection refused")

    with pytest.raises(ResultsSaveError, match="connection refused") as excinfo:
        save_results(_results(), [failing], experiment_id="exp-1", model_hash="mh")

    assert len(excinfo.value.failures) == 1


def test_save_results_tries_every_backend_before_raising() -> None:
    failing = MagicMock()
    failing.save.side_effect = RuntimeError("boom")
    working = MagicMock()

    with pytest.raises(ResultsSaveError):
        save_results(_results(), [failing, working], experiment_id="exp-1", model_hash="mh")

    working.save.assert_called_once()
    eval_result, instances = working.save.call_args.args
    assert eval_result.experiment_id == "exp-1"
    assert [inst["native_id"] for inst in instances["gsm8k"]] == ["a", "b"]


def test_save_results_raises_when_results_cannot_be_converted() -> None:
    backend = MagicMock()

    with pytest.raises(ResultsSaveError, match="task_hash is required"):
        save_results(_results(task_hash=None), [backend], experiment_id="exp-1")

    backend.save.assert_not_called()


def test_save_results_succeeds_when_every_backend_saves() -> None:
    backend = MagicMock()

    save_results(_results(), [backend], experiment_id="exp-1", model_hash="mh")

    backend.save.assert_called_once()

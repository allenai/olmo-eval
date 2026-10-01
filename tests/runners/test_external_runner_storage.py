"""Tests for how the external runner hands results to storage."""

from typing import Any
from unittest.mock import MagicMock, patch

from olmo_eval.evals.external.result import ExternalEvalResult
from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.external.runner import ExternalEvalRunner
from olmo_eval.storage.base import convert_runner_results


def test_results_without_counts_still_convert_for_storage(tmp_path):
    runner = ExternalEvalRunner(
        provider_config=ProviderConfig(model="model"),
        external_eval_names=["scicode", "asta"],
        output_dir=str(tmp_path),
        storages=[MagicMock()],
    )
    results = {
        "scicode": ExternalEvalResult(name="scicode", metrics={"pass@1": 0.25}),
        "asta": ExternalEvalResult.from_error("asta", "container exited"),
    }
    saved: list[dict[str, Any]] = []

    with patch(
        "olmo_eval.runners.io.storage.save_results",
        side_effect=lambda results, **_: saved.append(results),
    ):
        runner._save_results(results, total_duration=1.0)

    stored = convert_runner_results(saved[0], experiment_id="exp-1")
    assert {task.task_name: task.num_instances for task in stored.tasks} == {
        "scicode": 0,
        "asta": 0,
    }

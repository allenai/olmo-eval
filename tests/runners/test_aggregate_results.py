"""Tests for aggregate_results output shape expected by storage and metrics consumers."""

from __future__ import annotations

from olmo_eval.inference.providers.config import ProviderConfig
from olmo_eval.runners.asynq.results import aggregate_results
from olmo_eval.runners.common.types import TaskResult


def test_aggregate_results_output_shape():
    """The dict returned by aggregate_results must contain every field that
    storage.py and metrics.py read downstream: predictions in each task,
    harness_config at the top level, and attention_backend in model_config."""

    provider = ProviderConfig(model="test-model", alias="test", kind="mock")
    harness_cfg = {"model": "test-model", "provider": "mock"}
    predictions = [{"input": "x", "output": "y"}]

    result = aggregate_results(
        results={
            "task:metric": TaskResult(
                spec="task:metric",
                config={"task": "task:metric"},
                num_instances=10,
                metrics={"accuracy": {"exact_match": 0.8}},
                primary_metric="accuracy:exact_match",
                duration_seconds=1.5,
                predictions=predictions,
            ),
        },
        expanded_tasks=["task:metric"],
        task_specs=["task:metric"],
        provider_config=provider,
        attention_backend="flash_attention_2",
        harness_config=harness_cfg,
    )

    # Top-level fields consumed by metrics.py
    assert result["harness_config"] == harness_cfg
    assert result["model_config"]["attention_backend"] == "flash_attention_2"

    # Task-level fields consumed by storage.py
    task = result["tasks"]["task:metric"]
    assert task["predictions"] == predictions

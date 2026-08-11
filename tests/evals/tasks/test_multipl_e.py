"""Tests for MULTIPL-E task configuration."""

import pytest

from olmo_eval.common.types import Instance
from olmo_eval.evals.constants.code import MULTIPL_E_STOP_TOKENS
from olmo_eval.evals.tasks.common import get_task


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_olmo3base_v2_uses_dataset_stop_sequences(benchmark: str) -> None:
    task = get_task(f"multipl_e_{benchmark}_java:olmo3base:v2")
    instance = Instance(
        question="class Problem {",
        metadata={"stop_tokens": ["\n    }\n"]},
    )

    sampling_params = task.get_sampling_params(instance)

    assert sampling_params is not None
    assert sampling_params.stop_sequences == ("\n    }\n",)


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_legacy_olmo3base_keeps_configured_stop_sequences(benchmark: str) -> None:
    task = get_task(f"multipl_e_{benchmark}_java:olmo3base")
    instance = Instance(
        question="class Problem {",
        metadata={"stop_tokens": ["\n    }\n"]},
    )

    sampling_params = task.get_sampling_params(instance)

    assert sampling_params is not None
    assert sampling_params.stop_sequences == MULTIPL_E_STOP_TOKENS["java"]

"""Tests for MULTIPL-E task configuration."""

import pytest

from olmo_eval.common.types import Instance
from olmo_eval.evals.constants.code import MULTIPL_E_STOP_TOKENS
from olmo_eval.evals.tasks.common import get_task


@pytest.mark.parametrize("benchmark", ["humaneval", "mbpp"])
def test_olmo3base_v2_uses_dataset_stops_with_prompt_cap(benchmark: str) -> None:
    task = get_task(f"multipl_e_{benchmark}_java:olmo3base:v2")
    instance = Instance(
        question="class Problem {",
        metadata={"stop_tokens": ["\n    }\n"]},
    )

    sampling_params = task.get_sampling_params(instance)

    assert sampling_params is not None
    assert sampling_params.stop_sequences == ("\n    }\n",)
    assert sampling_params.truncate_prompt_tokens == 1024


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


class _FailingExecutionEnv:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def execute_command(self, command: str, timeout: float | None = None):
        raise self.error


@pytest.mark.anyio
async def test_multipl_e_sandbox_failure_is_not_scored_as_wrong() -> None:
    from olmo_eval.common.scorers.code_execution.multipl_e import MultiplEScorer
    from olmo_eval.common.types import LMOutput
    from olmo_eval.harness.sandbox.errors import SandboxTransportError

    instance = Instance(question="Q", metadata={"test": "echo ok"})
    output = LMOutput(text="unused")
    output.extracted_answer = "true"
    env = _FailingExecutionEnv(SandboxTransportError("command execution failed"))

    with pytest.raises(SandboxTransportError):
        await MultiplEScorer(language="sh").ascore(instance, output, env)


@pytest.mark.anyio
async def test_multipl_e_other_execution_errors_score_zero() -> None:
    from olmo_eval.common.scorers.code_execution.multipl_e import MultiplEScorer
    from olmo_eval.common.types import LMOutput

    instance = Instance(question="Q", metadata={"test": "echo ok"})
    output = LMOutput(text="unused")
    output.extracted_answer = "true"

    score = await MultiplEScorer(language="sh").ascore(
        instance, output, _FailingExecutionEnv(RuntimeError("bad command"))
    )

    assert score == 0.0
    assert output.metadata["execution_result"]["status"] == "Error"

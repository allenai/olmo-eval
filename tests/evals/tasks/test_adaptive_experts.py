import pytest

from olmo_eval.common.constants.models import get_model_presets
from olmo_eval.evals.suites.registry import get_suite
from olmo_eval.evals.tasks.common import get_task


@pytest.fixture(autouse=True)
def _setup_registry() -> None:
    import olmo_eval.evals.tasks  # noqa: F401


@pytest.mark.parametrize(
    "task_spec",
    (
        "math500:chat",
        "gpqa_diamond:qwen3_thinking",
        "humaneval_plus:chat:pass_at_1:qwen3_thinking",
        "ifeval_ood:qwen3_thinking",
    ),
)
def test_qwen3_thinking_tasks_use_32k_outputs(task_spec: str) -> None:
    sampling_params = get_task(task_spec).config.sampling_params

    assert sampling_params is not None
    assert sampling_params.max_tokens == 32768
    assert sampling_params.temperature == 0.6
    assert sampling_params.top_p == 0.95
    assert sampling_params.top_k == 20
    assert hash(sampling_params)


def test_humaneval_thinking_does_not_stop_inside_reasoning() -> None:
    sampling_params = get_task(
        "humaneval_plus:chat:pass_at_1:qwen3_thinking"
    ).config.sampling_params

    assert sampling_params is not None
    assert sampling_params.stop_sequences is None


def test_qwen3_hybrid_context_accommodates_32k_output() -> None:
    preset = get_model_presets()["qwen3-30b-a3b"]

    assert preset.max_model_len == 40960


def test_hybrid_pilot_excludes_humaneval() -> None:
    assert get_suite("adaptive_experts:hybrid_pilot").expand() == (
        "math500:chat",
        "gpqa_diamond:qwen3_thinking",
        "ifeval_ood:qwen3_thinking",
    )

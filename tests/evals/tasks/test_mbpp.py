"""Tests for MBPP task fixtures."""

from olmo_eval.evals.tasks.common import TaskConfig
from olmo_eval.evals.tasks.constants.mbpp import (
    MBPP_FEWSHOT_SOURCES,
    MBPP_FEWSHOT_SOURCES_V2,
)
from olmo_eval.evals.tasks.mbpp import MBPPOlmo3Base, MBPPOlmo3BaseV2


def test_hardcoded_fewshot_code_is_valid_python() -> None:
    """Few-shot answers must demonstrate executable, properly indented Python."""
    for example in MBPP_FEWSHOT_SOURCES_V2:
        compile(example["code"], f"<mbpp-fewshot-{example['task_id']}>", "exec")


def test_hardcoded_fewshot_code_passes_its_tests() -> None:
    """Each answer must satisfy the assertions shown alongside it."""
    for example in MBPP_FEWSHOT_SOURCES_V2:
        namespace: dict[str, object] = {}
        exec(example["code"], namespace)
        exec("\n".join(example["test_list"]), namespace)


def test_olmo3base_prompt_preserves_fewshot_code_formatting() -> None:
    """The prompt builder must include multiline few-shot answers unchanged."""
    task = MBPPOlmo3BaseV2(TaskConfig(name="mbpp:olmo3base:v2", num_fewshot=3))
    target = task.process_doc(
        {
            "text": "Write a function that returns its input.",
            "code": "def identity(value):\n    return value",
            "test_list": ["assert identity(1) == 1"],
            "task_id": 999,
        }
    )

    prompt = task.format_request(target).prompt

    for example in MBPP_FEWSHOT_SOURCES_V2[:3]:
        assert example["code"] in prompt


def test_legacy_olmo3base_prompt_keeps_original_fewshot_bytes() -> None:
    """The unversioned task remains reproducible after introducing v2."""
    task = MBPPOlmo3Base(TaskConfig(name="mbpp:olmo3base", num_fewshot=3))
    target = task.process_doc(
        {
            "text": "Write a function that returns its input.",
            "code": "def identity(value):\n    return value",
            "test_list": ["assert identity(1) == 1"],
            "task_id": 999,
        }
    )

    prompt = task.format_request(target).prompt

    for example in MBPP_FEWSHOT_SOURCES[:3]:
        assert example["code"] in prompt
    assert MBPP_FEWSHOT_SOURCES[0]["code"] != MBPP_FEWSHOT_SOURCES_V2[0]["code"]

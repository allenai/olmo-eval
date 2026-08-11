"""Regression tests for code-task few-shot configuration."""

import pytest

from olmo_eval.common.formatters import CompletionFormatter
from olmo_eval.evals.tasks.common import get_task


@pytest.mark.parametrize(
    "task_spec",
    [
        "humaneval:3shot",
        "humaneval:olmo3base",
        "codex_humaneval:3shot",
        "codex_humaneval:olmo3base",
        "humaneval_plus:3shot",
        "bigcodebench:3shot",
        "bigcodebench:olmo3base",
        "ds1000:3shot",
        "ds1000:olmo3base",
        "mbpp:3shot",
        "mbpp_plus:3shot",
        "mbpp:olmo3base",
        "mt_mbpp_v2fix_python:3shot",
        "mt_mbpp_v2fix_python:olmo3base",
    ],
)
def test_answered_code_fewshot_variants_have_a_formatter(task_spec: str) -> None:
    """A configured few-shot count must affect the serialized request."""
    task = get_task(task_spec)

    assert task.config.num_fewshot > 0
    assert task.config.formatter is not None


def test_mbpp_completion_variants_use_function_body_as_fewshot_answer() -> None:
    """MBPP questions already contain the signature, so do not repeat it."""
    for task_spec in ("mbpp:3shot", "mbpp_plus:3shot"):
        formatter = get_task(task_spec).config.formatter

        assert isinstance(formatter, CompletionFormatter)
        assert formatter.fewshot_answer_key == "fewshot_answer"


def test_mbpp_fewshot_prompt_does_not_repeat_function_signature() -> None:
    """The completed demonstration should contain exactly one function header."""
    task = get_task("mbpp:3shot")
    example = task.process_doc(
        {
            "text": "Return the input.",
            "code": "def identity(value):\n    return value",
            "test_list": ["assert identity(1) == 1"],
            "task_id": 1,
        }
    )
    target = task.process_doc(
        {
            "text": "Return zero.",
            "code": "def zero():\n    return 0",
            "test_list": ["assert zero() == 0"],
            "task_id": 2,
        }
    )
    task._fewshot_cache = [example]

    prompt = task.format_request(target).prompt

    assert prompt.count("def identity(value):") == 1
    assert "def identity(value):\n    return value" in prompt


def test_deepseek_leetcode_does_not_claim_unavailable_fewshots() -> None:
    """The source dataset does not contain reference solutions for demonstrations."""
    task = get_task("deepseek_leetcode:olmo3base")

    assert task.config.num_fewshot == 0

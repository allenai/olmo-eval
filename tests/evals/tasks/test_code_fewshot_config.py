"""Regression tests for code-task few-shot configuration."""

import pytest

from olmo_eval.common.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.evals.tasks.common import get_task


@pytest.mark.parametrize(
    "task_spec",
    [
        "humaneval:3shot",
        "humaneval:olmo3base",
        "codex_humaneval:3shot",
        "codex_humaneval:olmo3base:v2",
        "humaneval_plus:3shot",
        "bigcodebench:3shot",
        "bigcodebench:olmo3base",
        "ds1000:3shot",
        "ds1000:olmo3base",
        "mbpp:3shot:v2",
        "mbpp_plus:3shot:v2",
        "mbpp:olmo3base:v2",
        "mt_mbpp_v2fix_python:3shot:v2",
        "mt_mbpp_v2fix_python:olmo3base:v2",
    ],
)
def test_answered_code_fewshot_variants_have_a_formatter(task_spec: str) -> None:
    """A configured few-shot count must affect the serialized request."""
    task = get_task(task_spec)

    assert task.config.num_fewshot > 0
    assert task.config.formatter is not None


def test_mbpp_completion_variants_use_function_body_as_fewshot_answer() -> None:
    """MBPP questions already contain the signature, so do not repeat it."""
    for task_spec in ("mbpp:3shot:v2", "mbpp_plus:3shot:v2"):
        formatter = get_task(task_spec).config.formatter

        assert isinstance(formatter, CompletionFormatter)
        assert formatter.fewshot_answer_key == "fewshot_answer"


def test_mbpp_fewshot_prompt_does_not_repeat_function_signature() -> None:
    """The completed demonstration should contain exactly one function header."""
    task = get_task("mbpp:3shot:v2")
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


def test_v2_deepseek_leetcode_does_not_claim_unavailable_fewshots() -> None:
    """The source dataset does not contain reference solutions for demonstrations."""
    task = get_task("deepseek_leetcode:olmo3base:v2")

    assert task.config.num_fewshot == 0


@pytest.mark.parametrize(
    ("legacy_spec", "v2_spec"),
    [
        ("codex_humaneval:olmo3base", "codex_humaneval:olmo3base:v2"),
        ("mbpp:3shot", "mbpp:3shot:v2"),
        ("mbpp_plus:3shot", "mbpp_plus:3shot:v2"),
        ("mt_mbpp_v2fix_python:3shot", "mt_mbpp_v2fix_python:3shot:v2"),
        ("mt_mbpp_v2fix_python:olmo3base", "mt_mbpp_v2fix_python:olmo3base:v2"),
    ],
)
def test_v2_formatter_fixes_do_not_mutate_legacy_tasks(legacy_spec: str, v2_spec: str) -> None:
    assert get_task(legacy_spec).config.formatter is None
    assert isinstance(get_task(v2_spec).config.formatter, CompletionFormatter)


def test_deepseek_v2_preserves_legacy_fewshot_configuration() -> None:
    assert get_task("deepseek_leetcode:olmo3base").config.num_fewshot == 3
    assert get_task("deepseek_leetcode:olmo3base:v2").config.num_fewshot == 0


def test_ds1000_v2_preserves_protocol_and_declares_prompt_limit() -> None:
    legacy = get_task("ds1000:olmo3base")
    v2 = get_task("ds1000:olmo3base:v2")

    assert v2.config.num_fewshot == legacy.config.num_fewshot == 3
    assert v2.config.sampling_params is not None
    assert legacy.config.sampling_params is not None
    assert v2.config.sampling_params.max_tokens == legacy.config.sampling_params.max_tokens == 1024
    assert legacy.config.sampling_params.truncate_prompt_tokens is None
    assert v2.config.sampling_params.truncate_prompt_tokens == 4096
    assert v2.config.to_dict()["sampling_params"]["truncate_prompt_tokens"] == 4096


@pytest.mark.parametrize(
    "task_spec",
    [
        "codex_humaneval:olmo3base:v2:bpb",
        "mt_mbpp_v2fix_python:olmo3base:v2:bpb",
    ],
)
def test_v2_bpb_compositions_keep_ppl_formatter(task_spec: str) -> None:
    """The few-shot fix must not overwrite BPB's likelihood formatter."""
    assert isinstance(get_task(task_spec).config.formatter, PPLFormatter)

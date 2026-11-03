"""DeepSeek LeetCode v3: programs carry the upstream imports; earlier specs are unchanged."""

from __future__ import annotations

import subprocess
import sys

import pytest

from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.deepseek_leetcode import DEEPSEEK_IMPORT_HEADER

PROMPT = (
    '"""\nReturn the sum of nums.\n"""\n'
    "class Solution:\n    def total(self, nums: List[int]) -> int:\n        "
)
TEST = "\nmy_solution = Solution()\nassert my_solution.total(**{'nums': [1, 2, 3]}) == 6\n"


def _extracted(spec: str, text: str) -> str | None:
    task = get_task(spec)
    instance = Instance(
        question=PROMPT, gold_answer="", metadata={"id": "1", "prompt": PROMPT, "test": TEST}
    )
    output = LMOutput(text=text)
    response = Response(instance=instance, request=task.format_request(instance), outputs=[output])
    task._extract_answers([response])
    return output.extracted_answer


def _passes(program: str) -> bool:
    return subprocess.run([sys.executable, "-c", program], capture_output=True).returncode == 0


@pytest.mark.parametrize(
    "spec", ["deepseek_leetcode", "deepseek_leetcode:olmo3base", "deepseek_leetcode:olmo3base:v2"]
)
def test_earlier_specs_run_without_the_header(spec: str) -> None:
    assert get_task(spec).config.to_dict().get("task_settings") is None
    assert _extracted(spec, "return sum(nums)\n") == PROMPT + "return sum(nums)\n"


def test_v3_hash_records_the_import_header() -> None:
    config = get_task("deepseek_leetcode:olmo3base:v3").config.to_dict()
    assert config["task_settings"] == {"import_header": "deepseek_coder"}


def test_v3_program_starts_with_the_upstream_header() -> None:
    program = _extracted("deepseek_leetcode:olmo3base:v3", "return sum(nums)\n")
    assert program == DEEPSEEK_IMPORT_HEADER + "\n" + PROMPT + "return sum(nums)\n"
    assert "from typing import *" in DEEPSEEK_IMPORT_HEADER


def test_a_correct_solution_passes_only_with_the_header() -> None:
    """Without the header the List annotation raises NameError before any test runs.

    Python 3.14 evaluates annotations lazily (PEP 649), so there the program without
    the header passes too; the sandbox runs an earlier Python.
    """
    pytest.importorskip("numpy")
    program = _extracted("deepseek_leetcode:olmo3base:v3", "return sum(nums)\n")
    assert program is not None
    assert _passes(program + "\n" + TEST)
    if sys.version_info < (3, 14):
        assert not _passes(PROMPT + "return sum(nums)\n" + "\n" + TEST)

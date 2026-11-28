import subprocess
import sys

from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.tasks.common import get_task

PROMPT = "class Solution:\n    def total(self, nums: List[int]) -> int:\n        "
TEST = "\nassert Solution().total([1, 2, 3]) == 6\n"


def test_v3_program_has_the_import_header_and_runs() -> None:
    task = get_task("deepseek_leetcode:olmo3base:v3")
    instance = Instance(question=PROMPT, gold_answer="", metadata={"prompt": PROMPT, "test": TEST})
    output = LMOutput(text="return sum(nums)\n")
    task._extract_answers([Response(instance=instance, request=None, outputs=[output])])

    assert output.extracted_answer == task.IMPORT_HEADER + PROMPT + "return sum(nums)\n"
    run = subprocess.run([sys.executable, "-c", output.extracted_answer + TEST])
    assert run.returncode == 0


def test_v3_has_its_own_task_hash() -> None:
    assert get_task("deepseek_leetcode:olmo3base:v3").config.task_settings is not None


def test_v4_prompt_ends_at_the_newline_and_the_model_indents() -> None:
    task = get_task("deepseek_leetcode:olmo3base:v4")
    instance = task.process_doc({"prompt": PROMPT, "test": TEST, "questionId": "1"})
    assert instance.question == PROMPT.rstrip(" ")
    assert instance.question.endswith("-> int:\n")
    assert task.format_request(instance).prompt == instance.question

    output = LMOutput(text="        return sum(nums)\n")
    task._extract_answers([Response(instance=instance, request=None, outputs=[output])])
    run = subprocess.run([sys.executable, "-c", output.extracted_answer + TEST])
    assert run.returncode == 0


def test_v4_has_its_own_task_hash() -> None:
    v3 = get_task("deepseek_leetcode:olmo3base:v3").config.task_settings
    v4 = get_task("deepseek_leetcode:olmo3base:v4").config.task_settings
    assert v4 is not None and v4 != v3


def test_code_v4_suite_swaps_only_leetcode() -> None:
    from olmo_eval.evals.suites.registry import get_suite

    v3, v4 = get_suite("olmobase:code:v3").tasks, get_suite("olmobase:code:v4").tasks
    assert [t for t in v3 if t not in v4] == ["deepseek_leetcode:olmo3base:v3"]
    assert [t for t in v4 if t not in v3] == ["deepseek_leetcode:olmo3base:v4"]

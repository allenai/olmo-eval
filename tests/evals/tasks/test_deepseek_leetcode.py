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

"""Tests for the LiveCodeBench task, its prompts, and its grader."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from typing import Any

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.scorers import SandboxRequiredError
from olmo_eval.common.scorers.code_execution.scripts import get_script
from olmo_eval.common.types import Instance, LMOutput, RequestType
from olmo_eval.evals.tasks.common import get_task
from olmo_eval.evals.tasks.livecodebench import (
    RELEASE_V3_FILES,
    RELEASE_V4_V6_FILES,
    SYSTEM_PROMPT,
    LiveCodeBenchScorer,
)

STDIN_DOC = {
    "question_id": "abc123_a",
    "question_content": "Print the sum of two integers.",
    "starter_code": "",
    "metadata": "{}",
    "platform": "atcoder",
    "difficulty": "easy",
    "contest_date": "2024-01-01T00:00:00",
    "public_test_cases": "[]",
    "private_test_cases": "[]",
}

STARTER_DOC = {
    **STDIN_DOC,
    "question_id": "2727",
    "starter_code": (
        "class Solution:\n    def countSeniors(self, details: List[str]) -> int:\n        "
    ),
    "metadata": json.dumps({"func_name": "countSeniors"}),
    "platform": "leetcode",
}


# ---------------------------------------------------------------------------
# Instances and prompts
# ---------------------------------------------------------------------------


def test_stdin_problem_asks_for_a_whole_program() -> None:
    task = get_task("livecodebench")
    instance = task.process_doc(STDIN_DOC, index=7)

    assert instance.question == "Print the sum of two integers."
    assert instance.metadata["id"] == "abc123_a"
    assert instance.metadata["fn_name"] is None
    assert "reads the inputs" in instance.metadata["format_instruction"]


def test_starter_code_problem_carries_its_function_name() -> None:
    task = get_task("livecodebench")
    instance = task.process_doc(STARTER_DOC, index=0)

    assert instance.metadata["fn_name"] == "countSeniors"
    assert "starter code" in instance.metadata["format_instruction"]
    assert "def countSeniors" in instance.metadata["format_instruction"]


def test_test_cases_are_referenced_not_carried() -> None:
    # Payloads reach tens of megabytes per problem, and instance metadata is
    # written into the run's request records.
    task = get_task("livecodebench")
    instance = task.process_doc(STDIN_DOC, index=7)

    assert instance.metadata["row"] == 7
    assert instance.metadata["test_files"] == RELEASE_V3_FILES
    assert "public_test_cases" not in instance.metadata
    assert "private_test_cases" not in instance.metadata


def test_default_prompt_requests_reasoning_in_think_tags() -> None:
    task = get_task("livecodebench")
    request = task.format_request(task.process_doc(STDIN_DOC))

    assert request.request_type == RequestType.CHAT
    assert request.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = request.messages[1]["content"]
    assert user.startswith("### Question:\nPrint the sum of two integers.\n\n### Format:\n")
    assert "<think> </think> tag" in user
    assert user.endswith("### Answer: (use the provided format with backticks)\n\n")


def test_tulu_variant_drops_the_reasoning_line() -> None:
    task = get_task("livecodebench:tulu")
    user = task.format_request(task.process_doc(STDIN_DOC)).messages[1]["content"]

    assert "<think>" not in user
    assert user.startswith(
        "### Question:\nPrint the sum of two integers.\n\n### Format:\nRead the inputs from stdin"
    )


def test_answer_extraction_takes_the_last_code_block() -> None:
    # A reasoning model drafts code as it thinks; the answer is the last block.
    task = get_task("livecodebench")
    output = LMOutput(
        text=(
            "Let me try:\n```python\nprint(0)  # draft\n```\n"
            "That is wrong. Actually:\n```python\nprint(1)\n```"
        )
    )

    assert task.extract_answer(output) == "print(1)"


def test_answer_extraction_survives_an_unopened_think_tag() -> None:
    # A chat template may supply the opening tag, so the reply closes one it
    # never opened. Extraction must not depend on seeing a matched pair.
    task = get_task("livecodebench")
    output = LMOutput(text="reasoning...</think>\n```python\nprint(1)\n```")

    assert task.extract_answer(output) == "print(1)"


def test_answer_extraction_without_a_complete_block_yields_nothing() -> None:
    task = get_task("livecodebench")
    assert task.extract_answer(LMOutput(text="no code here at all")) is None
    assert task.extract_answer(LMOutput(text="```python\nprint(1)")) is None


# ---------------------------------------------------------------------------
# Releases and variants
# ---------------------------------------------------------------------------


def test_releases_select_different_contest_windows() -> None:
    assert get_task("livecodebench").config.data_source.data_files == RELEASE_V3_FILES
    assert get_task("livecodebench_hidden").config.data_source.data_files == RELEASE_V4_V6_FILES
    assert not set(RELEASE_V3_FILES) & set(RELEASE_V4_V6_FILES)


def test_default_regime_samples_ten_completions() -> None:
    config = get_task("livecodebench").config

    assert config.sampling_params is not None
    assert config.sampling_params.num_samples == 10
    assert config.sampling_params.temperature == pytest.approx(0.6)
    assert config.sampling_params.top_p == pytest.approx(0.95)
    # Generation is bounded by the model's context, not a fixed budget.
    assert config.sampling_params.max_tokens is None
    assert [metric.name for metric in config.metrics] == ["pass_at_1", "pass_at_5", "pass_at_10"]
    assert config.primary_metric is not None
    assert config.primary_metric.name == "pass_at_1"


def test_lite_variant_scores_a_single_sample() -> None:
    config = get_task("livecodebench:lite").config

    assert config.sampling_params is not None
    assert config.sampling_params.num_samples == 1
    assert config.sampling_params.temperature == pytest.approx(0.6)
    assert [metric.name for metric in config.metrics] == ["pass_at_1"]


def test_grpo_variant_reports_pass_at_10() -> None:
    config = get_task("livecodebench:grpo").config

    assert config.sampling_params is not None
    assert config.sampling_params.temperature == pytest.approx(1.0)
    assert config.sampling_params.top_p == pytest.approx(1.0)
    assert config.sampling_params.max_tokens == 16384
    assert config.primary_metric is not None
    assert config.primary_metric.name == "pass_at_10"


def test_variants_are_registered_for_both_releases() -> None:
    for variant in ("tulu", "lite", "grpo"):
        assert get_task(f"livecodebench_hidden:{variant}") is not None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class _StubStagingEnv:
    """Execution environment that records staged files and returns a verdict."""

    def __init__(self, output: str = '{"passed": true}', success: bool = True) -> None:
        self.output = output
        self.success = success
        self.staged: dict[str, str] = {}
        self.commands: list[str] = []

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        return ExecutionResult(success=self.success, output=self.output)

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult:
        return ExecutionResult(success=self.success, output=self.output)

    async def execute_with_files(
        self, command: str, files: Mapping[str, str], timeout: float | None = None
    ) -> ExecutionResult:
        self.commands.append(command)
        self.staged.update(files)
        return ExecutionResult(success=self.success, output=self.output)


class _NonStagingEnv:
    """Execution environment that cannot stage files."""

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        return ExecutionResult(success=True)

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult:
        return ExecutionResult(success=True)


def _instance() -> Instance:
    return Instance(
        question="Print the sum.",
        metadata={
            "id": "abc123_a",
            "row": 3,
            "test_repo": "org/repo",
            "test_files": RELEASE_V3_FILES,
            "fn_name": None,
        },
    )


@pytest.fixture
def stub_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"question_id": f"q{index}", "public_test_cases": "[]", "private_test_cases": ""}
        for index in range(4)
    ]
    rows[3] = {
        "question_id": "abc123_a",
        "public_test_cases": '[{"input": "1", "output": "1"}]',
        "private_test_cases": "",
    }

    def fake_rows(repo: str, files: tuple[str, ...]) -> Any:
        del repo, files
        return rows

    monkeypatch.setattr(
        "olmo_eval.evals.tasks.livecodebench._test_case_rows",
        fake_rows,
    )


@pytest.mark.anyio
class TestLiveCodeBenchScorer:
    async def test_stages_grader_problem_and_solution(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        score = await scorer.ascore(_instance(), output, env)

        assert score == 1.0
        staged = sorted(os.path.basename(path) for path in env.staged)
        assert staged == ["grade.py", "problem.json", "solution.py"]

        work_dirs = {os.path.dirname(path) for path in env.staged}
        assert len(work_dirs) == 1, "all files belong to one working directory"
        work_dir = work_dirs.pop()
        assert work_dir in env.commands[0]

        problem = json.loads(env.staged[f"{work_dir}/problem.json"])
        assert problem["public_test_cases"] == '[{"input": "1", "output": "1"}]'
        assert problem["timeout"] == scorer.timeout
        assert env.staged[f"{work_dir}/solution.py"] == "print(1)"

    async def test_working_directory_is_cleaned_up(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        await scorer.ascore(_instance(), output, env)

        assert "rm -rf" in env.commands[0]

    async def test_failing_verdict_scores_zero(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv(
            output='{"passed": false, "error_code": -2, "error_message": "Wrong Answer"}'
        )
        output = LMOutput(text="unused")
        output.extracted_answer = "print(2)"

        score = await scorer.ascore(_instance(), output, env)

        assert score == 0.0
        assert output.metadata["execution_result"]["error"] == "Wrong Answer"

    async def test_unparseable_output_scores_zero(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv(output="container died", success=False)
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        assert await scorer.ascore(_instance(), output, env) == 0.0
        assert output.metadata["execution_result"]["success"] is False

    async def test_missing_answer_scores_zero_without_executing(self) -> None:
        scorer = LiveCodeBenchScorer()
        env = _StubStagingEnv()
        output = LMOutput(text="no code here")

        assert await scorer.ascore(_instance(), output, env) == 0.0
        assert env.commands == []

    async def test_environment_without_staging_is_rejected(self, stub_rows: None) -> None:
        scorer = LiveCodeBenchScorer()
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(SandboxRequiredError):
            await scorer.ascore(_instance(), output, _NonStagingEnv())

    async def test_row_pointing_at_another_problem_is_rejected(self, stub_rows: None) -> None:
        # Silent misalignment would grade every solution against the wrong tests.
        scorer = LiveCodeBenchScorer()
        instance = _instance()
        instance.metadata["row"] = 1
        output = LMOutput(text="unused")
        output.extracted_answer = "print(1)"

        with pytest.raises(RuntimeError, match="row order changed"):
            await scorer.ascore(instance, output, _StubStagingEnv())


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------


def _grade(
    solution: str,
    tests: list[dict[str, str]],
    fn_name: str | None = None,
    timeout: int = 5,
) -> dict[str, Any]:
    """Run the container-side grader the way the sandbox would."""
    with tempfile.TemporaryDirectory() as work_dir:
        with open(os.path.join(work_dir, "grade.py"), "w") as handle:
            handle.write(get_script("livecodebench_grader"))
        with open(os.path.join(work_dir, "problem.json"), "w") as handle:
            json.dump(
                {
                    "public_test_cases": json.dumps(tests),
                    "private_test_cases": "",
                    "fn_name": fn_name,
                    "timeout": timeout,
                },
                handle,
            )
        with open(os.path.join(work_dir, "solution.py"), "w") as handle:
            handle.write(solution)

        process = subprocess.run(
            [sys.executable, "grade.py"],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

    verdicts = [line for line in process.stdout.strip().splitlines() if line.startswith("{")]
    return json.loads(verdicts[-1])


SUM_TESTS = [
    {"input": "1 2\n", "output": "3\n"},
    {"input": "10 20\n", "output": "30\n"},
]


class TestGraderStdinMode:
    def test_correct_program_passes(self) -> None:
        solution = "a, b = map(int, input().split())\nprint(a + b)\n"
        assert _grade(solution, SUM_TESTS)["passed"] is True

    def test_wrong_program_fails(self) -> None:
        verdict = _grade("a, b = map(int, input().split())\nprint(a - b)\n", SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -2

    def test_main_guard_is_unwrapped(self) -> None:
        solution = (
            "def main():\n"
            "    a, b = map(int, input().split())\n"
            "    print(a + b)\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        assert _grade(solution, SUM_TESTS)["passed"] is True

    def test_extra_output_line_fails(self) -> None:
        solution = "a, b = map(int, input().split())\nprint(a + b)\nprint('done')\n"
        verdict = _grade(solution, SUM_TESTS)
        assert verdict["passed"] is False
        assert "length" in verdict["error_message"]

    def test_numerically_equal_output_passes(self) -> None:
        # Formatting differences must not fail an otherwise correct answer.
        tests = [{"input": "1\n", "output": "1.50\n"}]
        assert _grade("input()\nprint(1.5)\n", tests)["passed"] is True

    def test_runtime_error_fails(self) -> None:
        verdict = _grade("raise ValueError('boom')\n", SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4

    def test_endless_loop_times_out(self) -> None:
        verdict = _grade("while True:\n    pass\n", SUM_TESTS, timeout=2)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -3

    def test_syntax_error_is_reported_as_a_compilation_failure(self) -> None:
        verdict = _grade("  this is not python\n", SUM_TESTS)
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4
        assert "Compilation error" in verdict["error_message"]


DOUBLE_TESTS = [
    {"input": "[1, 2, 3]", "output": "[2, 4, 6]"},
    {"input": "[]", "output": "[]"},
]


class TestGraderCallMode:
    def test_correct_solution_class_passes(self) -> None:
        solution = (
            "class Solution:\n"
            "    def double(self, nums: List[int]) -> List[int]:\n"
            "        return [n * 2 for n in nums]\n"
        )
        assert _grade(solution, DOUBLE_TESTS, fn_name="double")["passed"] is True

    def test_wrong_solution_fails(self) -> None:
        solution = (
            "class Solution:\n"
            "    def double(self, nums: List[int]) -> List[int]:\n"
            "        return nums\n"
        )
        verdict = _grade(solution, DOUBLE_TESTS, fn_name="double")
        assert verdict["passed"] is False
        assert verdict["error_code"] == -2

    def test_tuple_result_is_accepted(self) -> None:
        # Ground truth is never a tuple, so a tuple answer is not a mismatch.
        solution = (
            "class Solution:\n"
            "    def double(self, nums: List[int]) -> List[int]:\n"
            "        return tuple(n * 2 for n in nums)\n"
        )
        assert _grade(solution, DOUBLE_TESTS, fn_name="double")["passed"] is True

    def test_missing_function_fails(self) -> None:
        verdict = _grade("class Solution:\n    pass\n", DOUBLE_TESTS, fn_name="double")
        assert verdict["passed"] is False
        assert verdict["error_code"] == -4

    def test_module_level_function_is_found(self) -> None:
        solution = "def double(nums):\n    return [n * 2 for n in nums]\n"
        assert _grade(solution, DOUBLE_TESTS, fn_name="double")["passed"] is True

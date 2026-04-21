from __future__ import annotations

import re

import pytest

from olmo_eval.common.execution.environment import ExecutionResult
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.tasks.bigcodebench import (
    BigCodeBench,
    BigCodeBenchScorer,
    _build_bcb_execution_script,
)


class _StubExecutionEnv:
    """Minimal execution environment stub for scorer tests."""

    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.result = result or ExecutionResult(success=True, output="", exit_code=0)
        self.command_calls: list[tuple[str, float | None]] = []
        self.code_calls: list[tuple[str, str, float | None]] = []

    @property
    def is_running(self) -> bool:
        return True

    async def execute(self, command: str, timeout: float | None = None) -> str:
        return ""

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        self.command_calls.append((command, timeout))
        return self.result

    async def execute_code(
        self, code: str, language: str = "python", timeout: float | None = None
    ) -> ExecutionResult:
        self.code_calls.append((code, language, timeout))
        return self.result


def _extract_tmp_dir(command: str) -> str:
    matches = set(re.findall(r"/tmp/[0-9a-f]{32}", command))
    assert len(matches) == 1
    return matches.pop()


@pytest.mark.anyio
class TestCodeExecutionScorerPythonScriptHelper:
    async def test_execute_python_script_uses_file_based_command(self) -> None:
        scorer = CodeExecutionScorer()
        env = _StubExecutionEnv()

        result = await scorer.execute_python_script(
            env,
            "print('hello')\n",
            timeout=2.2,
            python_executable="/custom/python3",
            filename="runner.py",
        )

        assert result.success is True
        assert env.code_calls == []
        assert len(env.command_calls) == 1

        command, timeout = env.command_calls[0]
        tmp_dir = _extract_tmp_dir(command)

        assert timeout == pytest.approx(3.2)
        assert "printf '%s'" in command
        assert f"mkdir -p {tmp_dir}" in command
        assert f"cd {tmp_dir}" in command
        assert f"rm -rf {tmp_dir}" in command
        assert f"{tmp_dir}/runner.py" in command
        assert "timeout 3 bash -c" in command
        assert "/custom/python3" in command
        assert "runner.py" in command
        assert "ulimit -v" not in command
        assert "ulimit -d" not in command
        assert "ulimit -s" not in command


@pytest.mark.anyio
class TestBigCodeBenchScorer:
    async def test_ascore_uses_execute_command_with_rlimits(self) -> None:
        scorer = BigCodeBenchScorer()
        env = _StubExecutionEnv()
        instance = Instance(
            question="Q",
            metadata={
                "id": "task-1",
                "test": (
                    "import unittest\n"
                    "class TestCases(unittest.TestCase):\n"
                    "    def test_answer(self):\n"
                    "        self.assertEqual(answer(), 42)\n"
                ),
                "code_prompt": "",
            },
        )
        output = LMOutput(text="unused")
        output.extracted_answer = "def answer():\n    return 42\n"

        score = await scorer.ascore(instance, output, env)

        assert score == 1.0
        assert env.code_calls == []
        assert len(env.command_calls) == 1

        command, timeout = env.command_calls[0]
        tmp_dir = _extract_tmp_dir(command)

        assert timeout == pytest.approx(4.0)
        assert f"mkdir -p {tmp_dir}" in command
        assert f"cd {tmp_dir}" in command
        assert "ulimit -v 31457280" in command
        assert "ulimit -d 31457280" in command
        assert "ulimit -s 10240" in command
        assert "/usr/local/bin/python3" in command

    async def test_ascore_returns_zero_on_nonzero_command_exit(self) -> None:
        scorer = BigCodeBenchScorer()
        env = _StubExecutionEnv(result=ExecutionResult(success=False, output="", exit_code=1))
        instance = Instance(
            question="Q",
            metadata={"test": "class TestCases: pass", "code_prompt": ""},
        )
        output = LMOutput(text="unused")
        output.extracted_answer = "def answer():\n    return 42\n"

        score = await scorer.ascore(instance, output, env)

        assert score == 0.0
        assert env.code_calls == []
        assert len(env.command_calls) == 1

    @pytest.mark.parametrize(
        ("metadata", "extracted_answer"),
        [
            ({"test": "", "code_prompt": ""}, "def answer():\n    return 42\n"),
            ({"test": "class TestCases: pass", "code_prompt": ""}, None),
        ],
    )
    async def test_ascore_returns_zero_without_required_inputs(
        self,
        metadata: dict[str, str],
        extracted_answer: str | None,
    ) -> None:
        scorer = BigCodeBenchScorer()
        env = _StubExecutionEnv()
        instance = Instance(question="Q", metadata=metadata)
        output = LMOutput(text="unused")
        output.extracted_answer = extracted_answer

        score = await scorer.ascore(instance, output, env)

        assert score == 0.0
        assert env.command_calls == []
        assert env.code_calls == []


class TestBuildBcbExecutionScript:
    def test_script_contains_expected_harness_and_env(self) -> None:
        script = _build_bcb_execution_script(
            "def answer():\n    return 42\n",
            (
                "import unittest\n"
                "class TestCases(unittest.TestCase):\n"
                "    def test_answer(self):\n"
                "        self.assertEqual(answer(), 42)\n"
            ),
        )

        assert "os.environ['TZ'] = 'UTC'" in script
        assert "time.tzset()" in script
        assert "os.environ['OMP_NUM_THREADS'] = '1'" in script
        assert "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'" in script
        assert "loader = unittest.TestLoader()" in script
        assert "suite.run(test_result)" in script
        assert "TestCases = getattr(new_module, 'TestCases')" in script
        # reliability_guard ported from bigcodebench/eval/utils.py
        assert "resource.setrlimit(resource.RLIMIT_AS" in script
        assert "resource.setrlimit(resource.RLIMIT_DATA" in script
        assert "resource.setrlimit(resource.RLIMIT_STACK" in script
        assert "faulthandler.disable()" in script
        assert "builtins.exit = None" in script
        assert "builtins.quit = None" in script
        assert "socket" not in script


class TestBigCodeBenchSandboxEnv:
    def test_sandbox_uses_preset_provided_environment(self) -> None:
        assert BigCodeBench.sandbox_env.dependencies == ()
        assert BigCodeBench.sandbox_env.dockerfile_extra == ()

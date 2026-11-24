"""Python solution timeouts are failed solutions, not broken transport."""

import asyncio
from pathlib import Path

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput


class CommandEnvironment:
    @property
    def is_running(self):
        return True

    async def execute(self, command, timeout=None):
        return (await self.execute_command(command, timeout)).output

    async def execute_with_files(self, command, files, timeout=None):
        assert len(command.encode()) < 4096
        for filename, content in files.items():
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        return await self.execute_command(command, timeout)

    async def execute_command(self, command, timeout=None):
        process = await asyncio.create_subprocess_shell(
            command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        return ExecutionResult(
            success=process.returncode == 0,
            exit_code=process.returncode,
            output=(stdout + stderr).decode(),
        )

    async def execute_code(self, *args, **kwargs):
        raise AssertionError("Python must be bounded inside the command")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("body", "expected"), [("return 42", 1.0), ("return 0", 0.0), ("while True: pass", 0.0)]
)
async def test_python_solution_verdict(body, expected):
    scorer = CodeExecutionScorer(timeout=0.1)
    instance = Instance(question="answer", metadata={"test": "assert answer() == 42"})
    output = LMOutput(text="", extracted_answer="def answer():\n    " + body + "\n")
    assert await scorer.ascore(instance, output, CommandEnvironment()) == expected
    if "while" in body:
        assert output.metadata["execution_result"]["exit_code"] == 124


@pytest.mark.anyio
async def test_transport_failure_still_propagates():
    class BrokenEnvironment(CommandEnvironment):
        async def execute_command(self, command, timeout=None):
            raise ConnectionError("transport unavailable")

    scorer = CodeExecutionScorer()
    instance = Instance(question="answer", metadata={"test": "assert answer() == 42"})
    output = LMOutput(text="", extracted_answer="def answer():\n    return 42\n")
    with pytest.raises(ConnectionError, match="transport unavailable"):
        await scorer.ascore(instance, output, BrokenEnvironment())


@pytest.mark.anyio
async def test_large_quoted_payload_uses_file_staging():
    scorer = CodeExecutionScorer(timeout=1)
    tests = "# 'quoted payload'\n" * 20000 + "assert answer() == 42"
    instance = Instance(question="answer", metadata={"test": tests})
    output = LMOutput(text="", extracted_answer="def answer():\n    return 42\n")
    assert await scorer.ascore(instance, output, CommandEnvironment()) == 1.0

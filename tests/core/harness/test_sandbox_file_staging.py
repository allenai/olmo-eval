"""Tests for staging files into a sandbox before running a command."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.harness.sandbox import Capability, SandboxConfig, SandboxManager, SandboxMode


class _StagingExecutor:
    """Executor that records the files written to it and the commands it ran."""

    def __init__(self, name: str, *, max_concurrency: int = 2) -> None:
        self.name = name
        self.config = SandboxConfig(
            image="test",
            mode=SandboxMode.MODAL,
            capabilities=Capability.DEFAULT,
            max_concurrency=max_concurrency,
        )
        self.running = True
        self.files: dict[str, str] = {}
        self.commands: list[str] = []
        #: Files present when each command ran, to catch a command that
        #: executes somewhere its files were never written.
        self.files_at_command: list[set[str]] = []

    @property
    def is_running(self) -> bool:
        return self.running

    async def write_files(self, files: Mapping[str, str]) -> None:
        await asyncio.sleep(0.01)
        self.files.update(files)

    async def execute_command(self, command: str, timeout: float | None = None) -> ExecutionResult:
        del timeout
        self.commands.append(command)
        self.files_at_command.append(set(self.files))
        return ExecutionResult(success=True, output='{"passed": true}')


def _manager(*executors: _StagingExecutor) -> SandboxManager:
    manager = SandboxManager([])
    manager._executors = list(executors)  # type: ignore[assignment]
    manager._active_operations = {id(executor): 0 for executor in executors}
    return manager


@pytest.mark.anyio
async def test_files_are_written_before_the_command_runs() -> None:
    executor = _StagingExecutor("sb-0")
    manager = _manager(executor)

    result = await manager.execute_with_files(
        "python3 grade.py",
        {"/tmp/work/grade.py": "print(1)", "/tmp/work/problem.json": "{}"},
    )

    assert result.success
    assert executor.commands == ["python3 grade.py"]
    assert executor.files_at_command == [{"/tmp/work/grade.py", "/tmp/work/problem.json"}]


@pytest.mark.anyio
async def test_command_runs_on_the_executor_that_received_the_files() -> None:
    first = _StagingExecutor("sb-0")
    second = _StagingExecutor("sb-1")
    manager = _manager(first, second)

    # Enough concurrent calls to spread across both executors.
    await asyncio.gather(
        *(
            manager.execute_with_files(
                f"python3 grade.py {index}",
                {f"/tmp/work-{index}/solution.py": f"# {index}"},
            )
            for index in range(6)
        )
    )

    assert first.commands and second.commands, "expected work on both executors"
    for executor in (first, second):
        for command, staged in zip(executor.commands, executor.files_at_command, strict=True):
            index = command.rsplit(" ", 1)[-1]
            assert f"/tmp/work-{index}/solution.py" in staged


@pytest.mark.anyio
async def test_staging_respects_executor_capacity() -> None:
    executor = _StagingExecutor("sb-0", max_concurrency=1)
    manager = _manager(executor)

    await asyncio.gather(
        *(manager.execute_with_files(f"cmd {index}", {f"/tmp/{index}": "x"}) for index in range(4))
    )

    assert len(executor.commands) == 4

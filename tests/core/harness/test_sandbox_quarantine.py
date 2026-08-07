"""Tests for quarantining unresponsive sandbox executors."""

from __future__ import annotations

import aiohttp
import pytest

from olmo_eval.harness.sandbox.config import Capability, SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.executor import SandboxExecutor
from olmo_eval.harness.sandbox.manager import SandboxManager


class _FailingRuntime:
    async def execute(self, command: object) -> None:
        raise aiohttp.ConnectionTimeoutError("Connection timeout to host")


class _BrokenPipeRuntime:
    async def execute(self, command: object) -> None:
        raise BrokenPipeError("broken pipe")


def _running_executor(name: str) -> SandboxExecutor:
    executor = SandboxExecutor(
        SandboxConfig(image="test", mode=SandboxMode.MODAL),
        name=name,
    )
    executor._deployment = object()
    executor._runtime = object()
    return executor


@pytest.mark.anyio
async def test_transport_failure_quarantines_executor() -> None:
    executor = _running_executor("broken")
    executor._runtime = _FailingRuntime()

    with pytest.raises(aiohttp.ConnectionTimeoutError):
        await executor.execute_command("true")

    assert executor.is_running is False


@pytest.mark.anyio
async def test_execute_code_propagates_transport_failure_after_quarantine() -> None:
    executor = _running_executor("broken")
    executor._runtime = _BrokenPipeRuntime()

    with pytest.raises(BrokenPipeError):
        await executor.execute_code("print('hello')")

    assert executor.is_running is False


def test_manager_skips_quarantined_executor() -> None:
    manager = SandboxManager([])
    broken = _running_executor("broken")
    healthy = _running_executor("healthy")
    broken._mark_unhealthy(BrokenPipeError("broken pipe"))
    manager._executors = [broken, healthy]

    assert manager.is_running is True
    assert manager.get_executor(Capability.DEFAULT) is healthy

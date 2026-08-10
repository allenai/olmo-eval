"""Tests for strict per-executor sandbox capacity and routing."""

from __future__ import annotations

import asyncio

import pytest

from olmo_eval.common.execution import ExecutionResult
from olmo_eval.harness.sandbox import Capability, SandboxConfig, SandboxManager, SandboxMode
from olmo_eval.harness.sandbox.errors import SandboxTransportError


class _TrackingExecutor:
    def __init__(
        self,
        name: str,
        *,
        capabilities: frozenset[str] = Capability.DEFAULT,
        max_concurrency: int = 2,
        fail_and_quarantine: bool = False,
        fail_transport: bool = False,
    ) -> None:
        self.name = name
        self.config = SandboxConfig(
            image="test",
            mode=SandboxMode.MODAL,
            capabilities=capabilities,
            max_concurrency=max_concurrency,
        )
        self.running = True
        self.fail_and_quarantine = fail_and_quarantine
        self.fail_transport = fail_transport
        self.active = 0
        self.peak = 0
        self.calls = 0

    @property
    def is_running(self) -> bool:
        return self.running

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        timeout: float | None = None,
    ) -> ExecutionResult:
        del code, language, timeout
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            if self.fail_and_quarantine:
                self.running = False
                raise ConnectionResetError("sandbox disconnected")
            if self.fail_transport:
                raise SandboxTransportError("request retries exhausted")
            return ExecutionResult(success=True)
        finally:
            self.active -= 1


def _manager(*executors: _TrackingExecutor) -> SandboxManager:
    manager = SandboxManager([])
    manager._executors = list(executors)  # type: ignore[assignment]
    manager._active_operations = {id(executor): 0 for executor in executors}
    return manager


@pytest.mark.anyio
async def test_each_executor_enforces_its_own_capacity() -> None:
    executors = [_TrackingExecutor(f"sandbox-{idx}") for idx in range(3)]
    manager = _manager(*executors)
    environment = manager.for_capabilities(Capability.DEFAULT)

    await asyncio.gather(*(environment.execute_code("pass") for _ in range(18)))

    assert sum(executor.calls for executor in executors) == 18
    assert all(executor.calls > 0 for executor in executors)
    assert all(executor.peak <= 2 for executor in executors)


@pytest.mark.anyio
async def test_capability_environment_routes_each_operation_across_pool() -> None:
    capability = frozenset({"sandbox:bigcodebench"})
    executors = [
        _TrackingExecutor(f"bcb-{idx}", capabilities=capability, max_concurrency=1)
        for idx in range(5)
    ]
    manager = _manager(*executors)
    environment = manager.for_capabilities(capability)

    await asyncio.gather(*(environment.execute_code("pass") for _ in range(5)))

    assert [executor.calls for executor in executors] == [1, 1, 1, 1, 1]


@pytest.mark.anyio
async def test_scoring_environment_fails_over_after_quarantine() -> None:
    broken = _TrackingExecutor("broken", fail_and_quarantine=True)
    healthy = _TrackingExecutor("healthy")
    manager = _manager(broken, healthy)
    environment = manager.for_capabilities(Capability.DEFAULT)

    result = await environment.execute_code("pass")

    assert result.success is True
    assert broken.calls == 1
    assert healthy.calls == 1


@pytest.mark.anyio
async def test_scoring_environment_fails_over_after_exhausted_transport_retries() -> None:
    broken = _TrackingExecutor("broken", fail_transport=True)
    healthy = _TrackingExecutor("healthy")
    manager = _manager(broken, healthy)
    environment = manager.for_capabilities(Capability.DEFAULT)

    result = await environment.execute_code("pass")

    assert result.success is True
    assert broken.is_running is True
    assert broken.calls == 1
    assert healthy.calls == 1

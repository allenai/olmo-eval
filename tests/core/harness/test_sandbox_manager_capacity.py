"""Tests for strict per-executor sandbox capacity and routing."""

from __future__ import annotations

import asyncio

import pytest

from olmo_eval.common.scoring_time import ScoringTimes, measure_instance_scoring
from olmo_eval.harness.sandbox import Capability
from olmo_eval.harness.sandbox.errors import SandboxTransportError
from tests.conftest import ScoringClock
from tests.core.harness.sandbox_stubs import TrackingExecutor, make_manager


@pytest.mark.anyio
async def test_each_executor_enforces_its_own_capacity() -> None:
    executors = [TrackingExecutor(f"sandbox-{idx}") for idx in range(3)]
    manager = make_manager(*executors)
    environment = manager.for_capabilities(Capability.DEFAULT)

    await asyncio.gather(*(environment.execute_code("pass") for _ in range(18)))

    assert sum(executor.calls for executor in executors) == 18
    assert all(executor.calls > 0 for executor in executors)
    assert all(executor.peak <= 2 for executor in executors)


@pytest.mark.anyio
async def test_capability_environment_routes_each_operation_across_pool() -> None:
    capability = frozenset({"sandbox:bigcodebench"})
    executors = [
        TrackingExecutor(f"bcb-{idx}", capabilities=capability, max_concurrency=1)
        for idx in range(5)
    ]
    manager = make_manager(*executors)
    environment = manager.for_capabilities(capability)

    await asyncio.gather(*(environment.execute_code("pass") for _ in range(5)))

    assert [executor.calls for executor in executors] == [1, 1, 1, 1, 1]


@pytest.mark.anyio
async def test_scoring_environment_fails_over_after_quarantine() -> None:
    broken = TrackingExecutor("broken", fail_and_quarantine=True)
    healthy = TrackingExecutor("healthy")
    manager = make_manager(broken, healthy)
    environment = manager.for_capabilities(Capability.DEFAULT)

    result = await environment.execute_code("pass")

    assert result.success is True
    assert broken.calls == 1
    assert healthy.calls == 1


@pytest.mark.anyio
async def test_scoring_environment_does_not_fail_over_from_healthy_sandbox() -> None:
    broken = TrackingExecutor("broken", fail_transport=True)
    healthy = TrackingExecutor("healthy")
    manager = make_manager(broken, healthy)
    environment = manager.for_capabilities(Capability.DEFAULT)

    with pytest.raises(SandboxTransportError):
        await environment.execute_code("pass")

    assert broken.is_running is True
    assert broken.calls == 1
    assert healthy.calls == 0


@pytest.mark.anyio
async def test_scoring_records_sandbox_time_but_not_waits_for_a_slot(
    scoring_clock: ScoringClock,
) -> None:
    class OneSecondExecutor(TrackingExecutor):
        async def _run(self) -> None:
            await super()._run()
            scoring_clock.t += 1.0

    manager = make_manager(OneSecondExecutor("only", max_concurrency=1))
    environment = manager.for_capabilities(Capability.DEFAULT)
    totals = ScoringTimes()

    async def score() -> None:
        await asyncio.gather(*(environment.execute_code("pass") for _ in range(3)))

    await measure_instance_scoring(score(), totals)

    # The second and third calls waited 1 and 2 seconds for the one slot.
    assert totals.sandbox == pytest.approx(3.0)

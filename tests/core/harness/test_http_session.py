"""Persistent shell concurrency and deadline regressions using real pexpect."""

import asyncio

import httpx
import pytest
from swerex.exceptions import CommandTimeoutError, SessionDoesNotExistError
from swerex.runtime import local
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    CloseSessionRequest,
    CreateBashSessionRequest,
)

from olmo_eval.harness.sandbox._http_session import ResponsiveBashSession, ResponsiveSessionRuntime


@pytest.mark.anyio
async def test_http_session_keeps_file_requests_responsive(monkeypatch, tmp_path):
    import swerex.server as server

    monkeypatch.setattr(local, "BashSession", ResponsiveBashSession)
    runtime = local.LocalRuntime()
    monkeypatch.setattr(server, "runtime", runtime)
    marker = tmp_path / "started"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://sandbox"
    ) as client:
        created = await client.post(
            "/create_session", json=CreateBashSessionRequest(session="test").model_dump()
        )
        assert created.status_code == 200
        task = asyncio.create_task(
            client.post(
                "/run_in_session",
                json={
                    "session": "test",
                    "command": f"touch {marker}; sleep 1",
                    "timeout": 3,
                    "check": "silent",
                    "action_type": "bash",
                },
            )
        )
        try:
            async with asyncio.timeout(3):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            response = await client.post(
                "/write_file", json={"path": str(tmp_path / "file"), "content": "ready"}
            )
            assert response.status_code == 200 and not task.done()
            assert (await task).status_code == 200
        finally:
            await task
            await runtime.close()


@pytest.mark.anyio
async def test_session_serializes_commands_and_retains_environment(tmp_path):
    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    try:
        first = asyncio.create_task(
            session.run(
                BashAction(
                    command=f"sleep 0.1; cd {tmp_path}; export PERSIST=retained",
                    timeout=3,
                )
            )
        )
        await asyncio.sleep(0.01)
        second = asyncio.create_task(
            session.run(BashAction(command='printf "%s:%s" "$PERSIST" "$PWD"', timeout=3))
        )
        await first
        assert (await second).output == f"retained:{tmp_path}"
    finally:
        await session.close()


@pytest.mark.anyio
async def test_native_deadline_and_interrupt_recover_session():
    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    try:
        await session.run(BashAction(command="echo ready", timeout=3, check="silent"))
        with pytest.raises(CommandTimeoutError):
            await session.run(BashAction(command="sleep 30", timeout=0.5, check="silent"))
        await session.run(BashInterruptAction())
        assert (
            "recovered"
            in (await session.run(BashAction(command="echo recovered", timeout=3))).output
        )
    finally:
        await session.close()


@pytest.mark.anyio
async def test_cancellation_settles_reader_before_next_command(tmp_path):
    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    marker = tmp_path / "started"
    try:
        task = asyncio.create_task(
            session.run(BashAction(command=f"touch {marker}; sleep 30", timeout=3))
        )
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert (
            "recovered"
            in (await session.run(BashAction(command="echo recovered", timeout=3))).output
        )
    finally:
        await session.close()


@pytest.mark.anyio
async def test_interrupt_can_signal_an_active_serialized_reader(tmp_path):
    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    marker = tmp_path / "started"
    try:
        task = asyncio.create_task(
            session.run(
                BashAction(
                    command=f"touch {marker}; sleep 30",
                    timeout=3,
                    check="silent",
                )
            )
        )
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        await asyncio.wait_for(session.run(BashInterruptAction()), 3)
        assert task.done()
        await task
        response = await session.run(BashAction(command="echo recovered", timeout=3))
        assert "recovered" in response.output
    finally:
        await session.close()


@pytest.mark.anyio
@pytest.mark.parametrize("operation", ["cancel", "interrupt", "close"])
async def test_ignored_interrupt_without_deadline_does_not_lock_session(tmp_path, operation):
    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    marker = tmp_path / "started"
    task = asyncio.create_task(
        session.run(
            BashAction(
                command=f"trap '' INT; touch {marker}; sleep 30",
                timeout=None,
                check="silent",
            )
        )
    )
    try:
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        if operation == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 4)
        elif operation == "interrupt":
            await asyncio.wait_for(session.run(BashInterruptAction()), 4)
            await task
        else:
            await asyncio.wait_for(session.close(), 4)
            await task
        assert task.done()
        if operation != "close":
            response = await session.run(
                BashAction(command="trap - INT; echo recovered", timeout=3)
            )
            assert "recovered" in response.output
    finally:
        await asyncio.wait_for(session.close(), 4)


@pytest.mark.anyio
async def test_concurrent_close_uses_domain_error(monkeypatch):
    monkeypatch.setattr(local, "BashSession", ResponsiveBashSession)
    runtime = ResponsiveSessionRuntime()
    await runtime.create_session(CreateBashSessionRequest(session="test"))
    responses = await asyncio.gather(
        runtime.close_session(CloseSessionRequest(session="test")),
        runtime.close_session(CloseSessionRequest(session="test")),
        return_exceptions=True,
    )
    assert sum(isinstance(value, SessionDoesNotExistError) for value in responses) == 1
    assert sum(not isinstance(value, Exception) for value in responses) == 1


@pytest.mark.anyio
async def test_runtime_close_and_create_do_not_mutate_registry_during_iteration(monkeypatch):
    monkeypatch.setattr(local, "BashSession", ResponsiveBashSession)
    runtime = ResponsiveSessionRuntime()
    await runtime.create_session(CreateBashSessionRequest(session="first"))
    try:
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        creating = asyncio.create_task(
            runtime.create_session(CreateBashSessionRequest(session="second"))
        )
        await asyncio.gather(closing, creating)
        assert "second" in runtime.sessions
    finally:
        await runtime.close()


@pytest.mark.anyio
async def test_close_prevents_a_queued_unbounded_command_from_starting(tmp_path):
    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    marker = tmp_path / "started"
    first = asyncio.create_task(
        session.run(BashAction(command=f"touch {marker}; sleep 30", timeout=None, check="silent"))
    )
    try:
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        queued = asyncio.create_task(session.run(BashAction(command="sleep 30", timeout=None)))
        await asyncio.sleep(0)
        await asyncio.wait_for(session.close(), 4)
        await first
        with pytest.raises(RuntimeError, match="closing"):
            await queued
    finally:
        await asyncio.wait_for(session.close(), 4)

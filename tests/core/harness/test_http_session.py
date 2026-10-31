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
        await asyncio.wait_for(task, 1)
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


@pytest.mark.anyio
@pytest.mark.parametrize(
    "operation",
    [
        "interrupt",
        "cancel",
        "both",
        "cancel-during-cleanup",
        "natural-exit",
        "late-cleanup",
        "grace-boundary",
        "terminal-close-boundary",
    ],
)
async def test_deadline_during_interrupt_grace_keeps_queued_action_safe(
    monkeypatch, tmp_path, operation
):
    import contextlib
    import os
    import signal
    import threading
    import time

    import pexpect

    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    for _ in range(3):
        ready = await session.run(BashAction(command="echo warmed-up", timeout=3, check="silent"))
        if "warmed-up" in ready.output:
            break
    assert "warmed-up" in ready.output
    marker = tmp_path / "started"
    finish = tmp_path / "finish"
    finished = tmp_path / "finished"
    second_started = tmp_path / "second-started"
    expire = threading.Event()
    interrupted = threading.Event()
    draining = threading.Event()
    cancel_requested = threading.Event()
    readers_lock = threading.Lock()
    readers = 0
    peak_readers = 0
    original_expect = session.shell.expect
    original_sendintr = session.shell.sendintr

    def expect(pattern, timeout=-1, **kwargs):
        nonlocal readers, peak_readers
        with readers_lock:
            readers += 1
            peak_readers = max(readers, peak_readers)
        try:
            if timeout == 0.1:
                # Start the real deadline after SIGINT, inside its grace period.
                assert interrupted.wait(3)
                if operation == "terminal-close-boundary":
                    assert expire.wait(3)
            elif timeout == 1.0 and interrupted.is_set():
                draining.set()
                if operation == "cancel-during-cleanup":
                    assert cancel_requested.wait(3)
            try:
                return original_expect(pattern, timeout=timeout, **kwargs)
            except pexpect.TIMEOUT:
                if operation == "natural-exit" and timeout == 0.1:
                    finish.touch()
                    until = time.monotonic() + 3
                    while not finished.exists():
                        assert time.monotonic() < until
                        time.sleep(0.01)
                raise
        finally:
            with readers_lock:
                readers -= 1

    def sendintr():
        original_sendintr()
        interrupted.set()

    monkeypatch.setattr(session.shell, "expect", expect)
    monkeypatch.setattr(session.shell, "sendintr", sendintr)
    if operation == "late-cleanup":
        original_stop = session._stop_active

        async def stop(*args):
            # Let the real reader expire before its scheduled cleanup starts.
            interrupted.set()
            while not args[0].done():
                await asyncio.sleep(0.01)
            await original_stop(*args)

        monkeypatch.setattr(session, "_stop_active", stop)
    if operation in ("grace-boundary", "terminal-close-boundary"):
        original_wait_for = asyncio.wait_for

        async def wait_for(awaitable, timeout):
            if operation == "terminal-close-boundary" and timeout in (0.5, 1.0):
                awaitable.cancel()
                raise TimeoutError
            try:
                return await original_wait_for(awaitable, timeout)
            except CommandTimeoutError:
                if timeout == 0.5:
                    # The grace timer and reader deadline can become ready together.
                    raise TimeoutError from None
                raise

        monkeypatch.setattr(asyncio, "wait_for", wait_for)
    if operation == "terminal-close-boundary":
        original_to_thread = asyncio.to_thread

        async def to_thread(function, *args, **kwargs):
            if function.__name__ == "close_terminal":
                async with asyncio.timeout(3):
                    while not second_started.exists():
                        await asyncio.sleep(0.01)
                expire.set()
                while not session._active_worker.done():
                    await asyncio.sleep(0.01)
            return await original_to_thread(function, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", to_thread)
    work = (
        f"while [ ! -f {finish} ]; do sleep 0.01; done; touch {finished}"
        if operation == "natural-exit"
        else "sleep 30"
    )
    command = f"trap '' INT; /bin/sh -c 'touch {marker}; {work}'"
    if operation == "terminal-close-boundary":
        command += f"; /bin/sh -c 'echo $$ > {second_started}; sleep 30'"
    first = asyncio.create_task(
        session.run(BashAction(command=command, timeout=0.1, check="silent"))
    )
    foreground = None
    queued = None
    interrupt_task = None
    try:
        async with asyncio.timeout(3):
            while not marker.exists():
                await asyncio.sleep(0.01)
        foreground = os.tcgetpgrp(session.shell.child_fd)
        assert foreground != os.getpgrp()
        queued = asyncio.create_task(
            session.run(
                BashAction(
                    command="trap - INT; echo queued-recovered",
                    timeout=2,
                    check="silent",
                )
            )
        )
        await asyncio.sleep(0)
        if operation in (
            "interrupt",
            "natural-exit",
            "late-cleanup",
            "grace-boundary",
            "terminal-close-boundary",
        ):
            await asyncio.wait_for(session.run(BashInterruptAction()), 4)
            with pytest.raises(CommandTimeoutError):
                await first
        else:
            if operation != "cancel":
                interrupt_task = asyncio.create_task(session.run(BashInterruptAction()))
                signal_seen = interrupted if operation == "both" else draining
                async with asyncio.timeout(3):
                    while not signal_seen.is_set():
                        await asyncio.sleep(0.01)
            first.cancel()
            cancel_requested.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(first, 4)
            if interrupt_task is not None:
                await asyncio.wait_for(interrupt_task, 4)
        response = await asyncio.wait_for(queued, 4)
        assert response.output.strip() == "queued-recovered"
        assert peak_readers == 1
    finally:
        interrupted.set()
        cancel_requested.set()
        expire.set()
        finish.touch()
        if second_started.exists():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(int(second_started.read_text()), signal.SIGKILL)
        if foreground is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(foreground, signal.SIGKILL)
        for task in (first, queued, interrupt_task):
            if task is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task
        await session.close()


@pytest.mark.anyio
@pytest.mark.parametrize("concurrent_close", [False, True])
async def test_failed_prompt_drain_terminates_compound_action(
    monkeypatch, tmp_path, concurrent_close
):
    import contextlib
    import os
    import signal
    import subprocess
    import threading

    import pexpect

    session = ResponsiveBashSession(CreateBashSessionRequest(session="test"))
    await session.start()
    for _ in range(3):
        ready = await session.run(BashAction(command="echo warmed-up", timeout=3, check="silent"))
        if "warmed-up" in ready.output:
            break
    assert "warmed-up" in ready.output
    started = tmp_path / "started"
    second_pid = tmp_path / "second-pid"
    interrupted = threading.Event()
    draining = threading.Event()
    release = threading.Event()
    original_expect = session.shell.expect
    original_sendintr = session.shell.sendintr

    def expect(pattern, timeout=-1, **kwargs):
        if timeout == 0.1:
            assert interrupted.wait(3)
        elif timeout == 1.0 and interrupted.is_set():
            draining.set()
            assert release.wait(3)
        return original_expect(pattern, timeout=timeout, **kwargs)

    def sendintr():
        original_sendintr()
        interrupted.set()

    monkeypatch.setattr(session.shell, "expect", expect)
    monkeypatch.setattr(session.shell, "sendintr", sendintr)
    action = asyncio.create_task(
        session.run(
            BashAction(
                command=f"trap '' INT; /bin/sh -c 'touch {started}; sleep 30'; "
                f"/bin/sh -c 'echo $$ > {second_pid}; sleep 30'",
                timeout=0.1,
                check="silent",
            )
        )
    )
    interrupt = None
    closing = None
    try:
        async with asyncio.timeout(3):
            while not started.exists():
                await asyncio.sleep(0.01)
        interrupt = asyncio.create_task(session.run(BashInterruptAction()))
        async with asyncio.timeout(3):
            while not draining.is_set() or not second_pid.exists():
                await asyncio.sleep(0.01)
        if concurrent_close:
            closing = asyncio.create_task(session.close())
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(pexpect.TIMEOUT):
            await asyncio.wait_for(interrupt, 4)
        with pytest.raises(pexpect.TIMEOUT):
            await action
        if closing is not None:
            await asyncio.wait_for(closing, 3)
        else:
            assert not session.shell.isalive()
            await session.close()
        with pytest.raises(RuntimeError, match="shell not initialized"):
            _ = session.shell
        state = subprocess.run(
            ["ps", "-p", second_pid.read_text().strip(), "-o", "stat="],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        interrupted.set()
        release.set()
        if second_pid.exists():
            with contextlib.suppress(ProcessLookupError):
                os.killpg(int(second_pid.read_text()), signal.SIGKILL)
        for task in (action, interrupt, closing):
            if task is not None:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await task
        with contextlib.suppress(Exception):
            await session.close()
        with contextlib.suppress(RuntimeError):
            await asyncio.to_thread(session.shell.close, force=True)

"""Owned attached-process output and deployment cleanup regressions."""

import asyncio
import gc
import logging
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("swerex")

from olmo_eval.harness.sandbox.docker_deployment import OwnedDockerDeployment, _CapturedPipe
from olmo_eval.harness.sandbox.executor import _run_with_progress


@pytest.mark.anyio
async def test_attached_output_is_drained_before_readiness_and_bounded(monkeypatch):
    from swerex.deployment.docker import DockerDeployment

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; os.write(1,b'START'+b'x'*8388608+b'END'); "
            "os.write(2,b'ERROR'+b'y'*8388608+b'LAST')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deployment = OwnedDockerDeployment(image="unused", port=58123)
    deployment._container_process = process

    async def ready(self, timeout):
        await asyncio.to_thread(process.wait, timeout=5)

    monkeypatch.setattr(DockerDeployment, "_wait_until_alive", ready)
    try:
        await deployment._wait_until_alive(timeout=5)
        for capture in deployment._captures:
            await asyncio.to_thread(capture._thread.join, 1)
        for stream, prefix, suffix in (
            (process.stdout, b"START", b"END"),
            (process.stderr, b"ERROR", b"LAST"),
        ):
            output = stream.read()
            assert output.startswith(prefix) and output.endswith(suffix)
            assert b"[Output truncated]" in output
            assert len(output) <= 512 * 1024 + 100
    finally:
        await deployment.stop()
    assert all(not capture._thread.is_alive() for capture in (process.stdout, process.stderr))


def test_attached_pipe_drains_a_high_file_descriptor():
    fcntl = pytest.importorskip("fcntl")
    resource = pytest.importorskip("resource")
    minimum_fd = 1200
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft <= minimum_fd:
        if hard != resource.RLIM_INFINITY and hard <= minimum_fd:
            pytest.skip("File descriptor limit cannot support a high descriptor")
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (minimum_fd + 1, hard))
        except (OSError, ValueError) as exc:
            pytest.skip(f"Cannot raise the file descriptor limit: {exc}")
    read_fd = write_fd = high_fd = None
    try:
        read_fd, write_fd = os.pipe()
        high_fd = fcntl.fcntl(read_fd, fcntl.F_DUPFD, minimum_fd)
        assert high_fd >= minimum_fd
        payload = b"high-descriptor startup diagnostics"
        os.write(write_fd, payload)
        os.close(write_fd)
        write_fd = None
        stream = os.fdopen(high_fd, "rb", buffering=0)
        high_fd = None
        capture = _CapturedPipe(stream)
        try:
            capture._thread.join(timeout=2)
            assert not capture._thread.is_alive()
            assert capture.error is None
            assert capture.read() == payload
        finally:
            capture.close()
        assert stream.closed
    finally:
        for descriptor in (read_fd, write_fd, high_fd):
            if descriptor is not None:
                os.close(descriptor)
        if soft <= minimum_fd:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_attached_pipe_records_unexpected_reader_errors(monkeypatch, tmp_path):
    error = ValueError("selector setup failed")

    def failed_selector():
        raise error

    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.docker_deployment.selectors.DefaultSelector", failed_selector
    )
    stream = (tmp_path / "capture").open("w+b")
    capture = _CapturedPipe(stream)
    capture._thread.join(timeout=2)
    assert not capture._thread.is_alive()
    assert capture.error is error
    assert stream.closed


@pytest.mark.anyio
async def test_startup_failure_keeps_diagnostics_and_stops_process(monkeypatch, caplog):
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; os.write(2,b'startup-fixture-error'); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deployment = OwnedDockerDeployment(
        image="unused", port=58123, logger=logging.getLogger("startup-fixture")
    )
    deployment._container_process = process

    async def failed(*args, **kwargs):
        for _ in range(100):
            if b"startup-fixture-error" in process.stderr.read():
                raise TimeoutError("readiness fixture")
            await asyncio.sleep(0.01)
        pytest.fail("Attached stderr was not drained")

    monkeypatch.setattr("swerex.deployment.docker._wait_until_alive", failed)
    with pytest.raises(TimeoutError, match="readiness fixture"):
        await deployment._wait_until_alive(timeout=1)
    assert process.poll() is not None
    assert "startup-fixture-error" in caplog.text
    await deployment.stop()


class Remover:
    def __init__(self, mode):
        self.mode = mode
        self.returncode = None
        self.killed = False

    async def wait(self):
        if self.mode == "timeout" and not self.killed:
            await asyncio.sleep(60)
        self.returncode = 1 if self.mode == "fail" else (-9 if self.killed else 0)
        return self.returncode

    def kill(self):
        self.killed = True


@pytest.mark.anyio
@pytest.mark.parametrize("close_mode", ["ok", "raise", "timeout"])
@pytest.mark.parametrize("remove_mode", ["ok", "fail", "timeout"])
async def test_cleanup_is_bounded_idempotent_and_finalized(monkeypatch, close_mode, remove_mode):
    calls = []

    async def close():
        calls.append("close")
        if close_mode == "raise":
            raise ConnectionRefusedError("closed server")
        if close_mode == "timeout":
            await asyncio.sleep(60)

    deployment = OwnedDockerDeployment(image="unused", port=58123)
    deployment._runtime = SimpleNamespace(close=close)
    deployment._container_name = "owned-fixture"
    process = SimpleNamespace(poll=lambda: None, kill=Mock(), wait=Mock(return_value=0))
    deployment._container_process = process
    remover = Remover(remove_mode)

    async def create(*args, **kwargs):
        assert args == ("docker", "rm", "-f", "owned-fixture")
        return remover

    monkeypatch.setattr("asyncio.create_subprocess_exec", create)
    monkeypatch.setattr("olmo_eval.harness.sandbox.docker_deployment._CLOSE_TIMEOUT", 0.01)
    monkeypatch.setattr("olmo_eval.harness.sandbox.docker_deployment._REMOVE_TIMEOUT", 0.01)
    for _ in range(2):
        if remove_mode == "ok":
            await asyncio.wait_for(deployment.stop(), timeout=1)
        else:
            with pytest.raises(RuntimeError, match="confirm removal"):
                await asyncio.wait_for(deployment.stop(), timeout=1)
    assert calls == ["close"]
    process.kill.assert_called_once()
    assert deployment._runtime is None and deployment._container_process is None
    if remove_mode == "timeout":
        assert remover.killed
    del deployment
    gc.collect()
    await asyncio.sleep(0)
    assert calls == ["close"]


@pytest.mark.anyio
async def test_cleanup_waits_for_slow_successful_container_removal(monkeypatch):
    class DelayedRemover(Remover):
        async def wait(self):
            if not self.killed:
                await asyncio.sleep(3.1)
            return await super().wait()

    remover = DelayedRemover("ok")

    async def create(*args, **kwargs):
        assert args == ("docker", "rm", "-f", "owned-fixture")
        return remover

    monkeypatch.setattr("asyncio.create_subprocess_exec", create)
    deployment = OwnedDockerDeployment(image="unused", port=58123)
    deployment._container_name = "owned-fixture"
    await deployment.stop()
    assert remover.returncode == 0
    assert not remover.killed
    assert deployment._container_name is None


@pytest.mark.anyio
async def test_cancellation_waits_for_owned_cleanup_and_gc_creates_no_task(monkeypatch):
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def close():
        calls.append("close")
        entered.set()
        await finish.wait()

    deployment = OwnedDockerDeployment(image="unused", port=58123)
    deployment._runtime = SimpleNamespace(close=close)
    task = asyncio.create_task(deployment.stop())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    await deployment.stop()
    before = asyncio.all_tasks()
    del deployment
    gc.collect()
    await asyncio.sleep(0)
    assert asyncio.all_tasks() == before
    assert calls == ["close"]


@pytest.mark.anyio
async def test_progress_cancellation_awaits_the_startup_task():
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def start():
        entered.set()
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.set()

    task = asyncio.create_task(_run_with_progress(start(), "fixture", interval=0.01))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.anyio
async def test_deployment_can_restart_after_completed_cleanup(monkeypatch):
    from swerex.deployment.docker import DockerDeployment

    calls = []

    async def close():
        calls.append("close")

    async def start(self):
        self._runtime = SimpleNamespace(close=close)

    monkeypatch.setattr(DockerDeployment, "start", start)
    deployment = OwnedDockerDeployment(image="unused", port=58123)
    for _ in range(2):
        await deployment.start()
        await deployment.stop()
    assert calls == ["close", "close"]


@pytest.mark.anyio
async def test_cleanup_preserves_container_and_image_options(monkeypatch):
    deployment = OwnedDockerDeployment(
        image="unused", port=58123, remove_container=False, remove_images=True
    )
    deployment._container_name = "owned-fixture"
    commands = []

    async def create(*args, **kwargs):
        commands.append(args)
        return Remover("ok")

    monkeypatch.setattr("asyncio.create_subprocess_exec", create)
    await deployment.stop()
    assert commands == [("docker", "kill", "owned-fixture"), ("docker", "rmi", "unused")]


@pytest.mark.anyio
async def test_executor_waits_for_owned_cleanup_without_a_competing_deadline(monkeypatch, caplog):
    from olmo_eval.harness.sandbox import SandboxConfig, SandboxExecutor, SandboxMode

    completed = False

    async def close():
        nonlocal completed
        await asyncio.sleep(0.03)
        completed = True

    real_wait_for = asyncio.wait_for

    async def shortened_generic_deadline(awaitable, timeout):
        return await real_wait_for(awaitable, 0.005 if timeout == 10.0 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", shortened_generic_deadline)
    caplog.set_level(logging.DEBUG, logger="olmo_eval.harness.sandbox.executor")
    deployment = OwnedDockerDeployment(image="unused", port=58123)
    runtime = SimpleNamespace(close=close)
    deployment._runtime = runtime
    executor = SandboxExecutor(SandboxConfig(image="unused", mode=SandboxMode.DOCKER))
    executor._deployment = deployment
    executor._runtime = runtime
    await executor.stop()
    assert completed
    assert "Failed to stop deployment" not in caplog.text
    assert executor._deployment is None and executor._runtime is None


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["docker", "local", "modal"])
async def test_executor_preserves_generic_deployment_deadlines(monkeypatch, caplog, mode):
    from olmo_eval.harness.sandbox import SandboxConfig, SandboxExecutor, SandboxMode

    cancelled = False
    completed = False

    async def stop():
        nonlocal cancelled, completed
        try:
            await asyncio.sleep(0.03)
            completed = True
        except asyncio.CancelledError:
            cancelled = True
            raise

    real_wait_for = asyncio.wait_for

    async def shortened_generic_deadline(awaitable, timeout):
        return await real_wait_for(awaitable, 0.005 if timeout == 10.0 else timeout)

    monkeypatch.setattr(asyncio, "wait_for", shortened_generic_deadline)
    caplog.set_level(logging.DEBUG, logger="olmo_eval.harness.sandbox.executor")
    executor = SandboxExecutor(SandboxConfig(image="unused", mode=SandboxMode(mode)))
    executor._deployment = SimpleNamespace(stop=stop)
    executor._runtime = SimpleNamespace()
    await executor.stop()
    assert cancelled and not completed
    assert "Failed to stop deployment" in caplog.text

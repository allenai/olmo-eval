"""Attached sandbox log output must not block Docker or file staging."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time

import pytest

from olmo_eval.harness.sandbox import SandboxConfig, SandboxMode
from olmo_eval.harness.sandbox.executor import SandboxExecutor

pytestmark = pytest.mark.integration


@pytest.fixture
async def docker_sandbox():
    try:
        available = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("Docker is not available")
    if available.returncode:
        pytest.skip("Docker is not available")
    config = SandboxConfig(
        image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        mode=SandboxMode.DOCKER,
        container_runtime="docker",
        startup_timeout=120.0,
        command_timeout=10.0,
        inject_swerex=True,
    )
    async with SandboxExecutor(config) as sandbox:
        yield sandbox


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [1, 2])
async def test_runtime_log_flood_does_not_block_file_staging(docker_sandbox, stream):
    code = (
        "import os\n"
        f"fd = os.open('/proc/1/fd/{stream}', os.O_WRONLY)\n"
        "try:\n"
        "    for _ in range(128): os.write(fd, b'x' * 65536)\n"
        "finally: os.close(fd)\n"
    )
    flooded = await asyncio.to_thread(
        subprocess.run,
        [
            "docker",
            "exec",
            docker_sandbox._deployment.container_name,
            "python",
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert flooded.returncode == 0, flooded.stderr
    await asyncio.wait_for(
        docker_sandbox.write_files({"/tmp/after-log-flood": "responsive"}, timeout=3),
        timeout=10,
    )
    staged = await asyncio.wait_for(
        docker_sandbox.execute_command("cat /tmp/after-log-flood", timeout=3), timeout=10
    )
    assert staged.success and staged.output == "responsive"


@pytest.mark.anyio
@pytest.mark.parametrize("runtime", ["docker", "podman"])
async def test_force_removal_stops_a_sigterm_ignoring_pid_one(monkeypatch, runtime):
    if shutil.which(runtime) is None:
        pytest.skip(f"{runtime} is not available")
    try:
        available = await asyncio.to_thread(
            subprocess.run, [runtime, "info"], capture_output=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip(f"{runtime} is not available")
    if available.returncode:
        pytest.skip(f"{runtime} is not available")

    from swerex.deployment.docker import DockerDeployment, RemoteRuntime
    from swerex.runtime.abstract import IsAliveResponse

    from olmo_eval.harness.sandbox.docker_deployment import OwnedDockerDeployment

    deployment = OwnedDockerDeployment(
        image="ghcr.io/astral-sh/uv:python3.12-bookworm-slim",
        container_runtime=runtime,
        startup_timeout=120,
        docker_args=["--stop-timeout", "30"],
    )
    code = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('READY', flush=True); time.sleep(600)"
    )
    monkeypatch.setattr(
        DockerDeployment,
        "_get_swerex_start_cmd",
        lambda *args: ["python3", "-u", "-c", code],
    )

    class Readiness:
        async def is_alive(self, **kwargs):
            return IsAliveResponse(is_alive=b"READY" in deployment._container_process.stdout.read())

        async def close(self):
            pass

    monkeypatch.setattr(RemoteRuntime, "from_config", lambda *args: Readiness())
    name = None
    try:
        await deployment.start()
        name = deployment.container_name
        attached = deployment._container_process
        assert attached is not None and attached.poll() is None
        started = time.monotonic()
        await deployment.stop()
        assert time.monotonic() - started < 5
        assert attached.poll() is not None
        listed = await asyncio.to_thread(
            subprocess.run,
            [runtime, "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert listed.returncode == 0, listed.stderr
        assert name not in listed.stdout.splitlines()
        inspected = await asyncio.to_thread(
            subprocess.run, [runtime, "inspect", name], capture_output=True, timeout=10
        )
        assert inspected.returncode != 0
    finally:
        name = name or deployment.container_name
        if name is not None:
            await asyncio.to_thread(
                subprocess.run,
                [runtime, "kill", name],
                capture_output=True,
                timeout=10,
            )
            await asyncio.to_thread(
                subprocess.run,
                [runtime, "rm", "-f", name],
                capture_output=True,
                timeout=10,
            )
        await deployment.stop()

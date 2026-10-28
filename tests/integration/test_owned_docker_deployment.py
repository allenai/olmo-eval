"""Attached sandbox log output must not block Docker or file staging."""

from __future__ import annotations

import asyncio
import subprocess

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

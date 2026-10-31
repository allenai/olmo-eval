"""SWE-ReX command adapter and deployment bootstrap regressions."""

import asyncio
import os
import subprocess
import sys
import threading

import httpx
import pytest

pytest.importorskip("swerex")

from swerex.exceptions import CommandTimeoutError, NonZeroExitCodeError
from swerex.runtime.abstract import Command
from swerex.runtime.local import LocalRuntime

from olmo_eval.harness.sandbox._http_command import execute
from olmo_eval.harness.sandbox.server_command import server_command


@pytest.mark.anyio
async def test_http_command_keeps_health_and_file_requests_responsive(monkeypatch, tmp_path):
    import swerex.server as server

    monkeypatch.setattr(LocalRuntime, "execute", execute)
    marker = tmp_path / "started"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://sandbox"
    ) as client:
        task = asyncio.create_task(
            client.post(
                "/execute",
                json={
                    "command": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); "
                        "time.sleep(1)",
                    ],
                    "timeout": 5,
                },
            )
        )
        try:
            async with asyncio.timeout(3):
                while not marker.exists():
                    await asyncio.sleep(0.01)
            health = await client.get("/is_alive")
            written = await client.post(
                "/write_file", json={"path": str(tmp_path / "file"), "content": "ready"}
            )
            assert health.status_code == written.status_code == 200
            assert not task.done()
            assert (tmp_path / "file").read_text() == "ready"
            assert (await task).json()["exit_code"] == 0
        finally:
            await task


@pytest.mark.anyio
@pytest.mark.parametrize("shell", [False, True])
async def test_command_preserves_environment_cwd_streams_and_status(tmp_path, shell):
    code = 'printf "%s:%s" "$TOKEN" "$PWD"; printf error >&2; exit 7'
    command = Command(
        command=code if shell else ["/bin/sh", "-c", code],
        shell=shell,
        env={"TOKEN": "space value"},
        cwd=str(tmp_path),
        merge_output_streams=True,
    )
    response = await execute(LocalRuntime(), command)
    assert response.exit_code == 7
    assert response.stdout == f"space value:{tmp_path}error"
    assert response.stderr == ""
    command.check = True
    command.error_msg = "custom"
    with pytest.raises(NonZeroExitCodeError, match="custom"):
        await execute(LocalRuntime(), command)


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_deadline_and_cancellation_kill_descendants_holding_output(tmp_path, cancel):
    marker = tmp_path / "orphan"
    started = tmp_path / "started"
    script = f"(sleep 0.7; touch {marker}) & touch {started}; sleep 30"
    task = asyncio.create_task(
        execute(
            LocalRuntime(),
            Command(
                command=["/bin/sh", "-c", script],
                timeout=None if cancel else 0.1,
            ),
        )
    )
    if cancel:
        async with asyncio.timeout(3):
            while not started.exists():
                await asyncio.sleep(0.01)
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else CommandTimeoutError):
        await asyncio.wait_for(task, 3)
    await asyncio.sleep(0.8)
    assert not marker.exists()
    response = await execute(LocalRuntime(), Command(command=["/bin/echo", "recovered"]))
    assert response.stdout == "recovered\n"


@pytest.mark.parametrize("use_path", [False, True])
def test_bootstrap_uses_installed_shebang_and_preserves_shell_environment(tmp_path, use_path):
    entrypoint = tmp_path / "swerex-remote"
    entrypoint.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "from swerex.runtime.local import LocalRuntime\n"
        "print(LocalRuntime.execute.__module__, os.environ['BOOT_ENV'], sys.argv[1:])\n"
    )
    entrypoint.chmod(0o700)
    executable = "swerex-remote" if use_path else str(entrypoint)
    result = subprocess.run(
        ["/bin/bash", "-c", server_command(executable, "token with ' quote", 8123)],
        env=os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}", "BOOT_ENV": "preserved"},
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("__main__ preserved")
    assert "token with" in result.stdout
    assert "--port', '8123'" in result.stdout
    assert entrypoint.read_text().startswith(f"#!{sys.executable}\nimport os")


def test_bootstrap_installs_with_module_when_pipx_is_not_on_path(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    entrypoint = install / "swerex-remote"
    entrypoint.write_text(f"#!{sys.executable}\nprint('bootstrapped')\n")
    calls = tmp_path / "calls"
    python = tmp_path / "python3"
    python.write_text(
        f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {calls}\n"
        f"if [ \"$2 $3\" = 'pipx environment' ]; then echo {install}; fi\n"
    )
    python.chmod(0o700)
    result = subprocess.run(
        ["/bin/bash", "-c", server_command("swerex-remote", "token", 8123, install_pipx=True)],
        env=os.environ | {"PATH": f"{tmp_path}:/usr/bin:/bin"},
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bootstrapped"
    assert calls.read_text().splitlines() == [
        "-m pip install pipx",
        "-m pipx ensurepath",
        "-m pipx install swe-rex",
        "-m pipx environment --value PIPX_BIN_DIR",
    ]


@pytest.mark.anyio
async def test_background_child_does_not_hold_command_output_open(tmp_path):
    child = tmp_path / "done"
    release = tmp_path / "release"
    response = await asyncio.wait_for(
        execute(
            LocalRuntime(),
            Command(
                command=[
                    "/bin/sh",
                    "-c",
                    f"(while [ ! -f {release} ]; do sleep 0.05; done; touch {child}) & printf foreground",
                ]
            ),
        ),
        3,
    )
    assert response.stdout == "foreground"
    release.touch()
    async with asyncio.timeout(3):
        while not child.exists():
            await asyncio.sleep(0.01)


@pytest.mark.anyio
async def test_cancellation_waits_for_output_reader_before_closing_files(monkeypatch):
    import olmo_eval.harness.sandbox._http_command as adapter

    started = threading.Event()
    release = threading.Event()
    files = []
    original_file = adapter.tempfile.TemporaryFile
    original_thread = asyncio.to_thread

    def temporary_file():
        file = original_file()
        files.append(file)
        return file

    async def to_thread(function, *args, **kwargs):
        def blocked_read():
            started.set()
            assert release.wait(5)
            assert all(not file.closed for file in files)
            return function(*args, **kwargs)

        return await original_thread(blocked_read)

    monkeypatch.setattr(adapter.tempfile, "TemporaryFile", temporary_file)
    monkeypatch.setattr(adapter.asyncio, "to_thread", to_thread)
    task = asyncio.create_task(execute(LocalRuntime(), Command(command=["/bin/echo", "output"])))
    try:
        async with asyncio.timeout(3):
            while not started.is_set():
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert all(not file.closed for file in files)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(file.closed for file in files)

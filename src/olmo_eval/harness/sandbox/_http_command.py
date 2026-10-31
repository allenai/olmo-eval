"""Asynchronous SWE-ReX command execution, embedded into remote server startup."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import os
import signal
import tempfile
from typing import Any

from swerex.exceptions import CommandTimeoutError, NonZeroExitCodeError
from swerex.runtime import local
from swerex.runtime.abstract import Command, CommandResponse
from swerex.runtime.local import LocalRuntime


async def execute(self: LocalRuntime, command: Command) -> CommandResponse:
    """Execute independent commands without occupying the HTTP server loop."""
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        # Keep inherited output append-only even if a descriptor's offset changes.
        for output in (stdout_file, stderr_file):
            flags = fcntl.fcntl(output, fcntl.F_GETFL)
            fcntl.fcntl(output, fcntl.F_SETFL, flags | os.O_APPEND)
        kwargs: dict[str, Any] = {
            "env": command.env,
            "cwd": command.cwd,
            "stdout": stdout_file,
            "stderr": stdout_file if command.merge_output_streams else stderr_file,
            "start_new_session": True,
        }
        args = [command.command] if isinstance(command.command, str) else command.command
        if command.shell:
            # Match subprocess.run's sequence handling with shell=True.
            args = ["/bin/sh", "-c", *args]
        spawning = asyncio.create_task(asyncio.create_subprocess_exec(*args, **kwargs))
        cancelled: asyncio.CancelledError | None = None
        while True:
            try:
                process = await asyncio.shield(spawning)
                break
            except asyncio.CancelledError as exc:
                if spawning.cancelled():
                    raise
                cancelled = exc
        try:
            if cancelled is not None:
                raise cancelled
            await asyncio.wait_for(process.wait(), command.timeout)
        # SWE-ReX can run on Python 3.10, where this is not built-in TimeoutError.
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:  # noqa: UP041
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            cleanup = asyncio.create_task(asyncio.wait_for(process.wait(), 2.0))
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    pass
                except asyncio.TimeoutError:  # noqa: UP041
                    break
            with contextlib.suppress(asyncio.TimeoutError):  # noqa: UP041
                cleanup.result()
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise CommandTimeoutError(
                f"Timeout ({command.timeout}s) exceeded while running command"
            ) from exc

        def read_output() -> tuple[bytes, bytes]:
            return (
                os.pread(stdout_file.fileno(), os.fstat(stdout_file.fileno()).st_size, 0),
                os.pread(stderr_file.fileno(), os.fstat(stderr_file.fileno()).st_size, 0),
            )

        # Regular files keep a detached child from holding communicate() open.
        reader = asyncio.create_task(asyncio.to_thread(read_output))
        cancelled: asyncio.CancelledError | None = None
        while True:
            try:
                stdout, stderr = await asyncio.shield(reader)
                break
            except asyncio.CancelledError as exc:
                if reader.cancelled():
                    raise
                cancelled = exc
        if cancelled is not None:
            raise cancelled
        response = CommandResponse(
            stdout=stdout.decode(errors="backslashreplace"),
            stderr=stderr.decode(errors="backslashreplace"),
            exit_code=process.returncode,
        )
    if command.check and response.exit_code != 0:
        msg = (
            f"Command {command.command!r} failed with exit code {response.exit_code}. "
            f"Stdout:\n{response.stdout!r}\nStderr:\n{response.stderr!r}"
        )
        if command.error_msg:
            msg = f"{command.error_msg}: {msg}"
        raise NonZeroExitCodeError(msg)
    return response


class ResponsiveLocalRuntime(LocalRuntime):
    async def execute(self, command: Command) -> CommandResponse:
        return await execute(self, command)


def install() -> None:
    """Install command execution before the server constructs its runtime."""
    vars(local).update(LocalRuntime=ResponsiveLocalRuntime)

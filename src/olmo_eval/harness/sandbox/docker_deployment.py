"""Owned SWE-ReX Docker deployment with bounded attached output and cleanup."""

from __future__ import annotations

import asyncio
import contextlib
import os
import selectors
import shutil
import subprocess
import threading
from collections import deque
from typing import Any, BinaryIO

from swerex.deployment.docker import DockerDeployment

_CAPTURE_BYTES = 256 * 1024
_CLOSE_TIMEOUT = 2.0
_REMOVE_TIMEOUT = 10.0
_PROCESS_TIMEOUT = 2.0


def _is_podman_runtime(runtime: str) -> bool:
    """Recognize Podman and Docker compatibility symlinks to its executable."""
    if runtime == "podman":
        return True
    executable = shutil.which(runtime)
    podman = shutil.which("podman")
    if executable is not None and podman is not None:
        with contextlib.suppress(OSError):
            return os.path.samefile(executable, podman)
    return False


class _CapturedPipe:
    """Drain an attached process pipe while retaining bounded diagnostics."""

    def __init__(self, stream: BinaryIO):
        self._stream = stream
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._prefix = bytearray()
        self._suffix: deque[bytes] = deque()
        self._suffix_size = 0
        self._truncated = False
        self.error: Exception | None = None
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        try:
            descriptor = self._stream.fileno()
            os.set_blocking(descriptor, False)
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_READ)
                while not self._stop.is_set():
                    if not selector.select(timeout=0.05):
                        continue
                    try:
                        chunk = os.read(descriptor, 64 * 1024)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        break
                    with self._lock:
                        take = min(len(chunk), _CAPTURE_BYTES - len(self._prefix))
                        self._prefix.extend(chunk[:take])
                        chunk = chunk[take:]
                        if chunk:
                            self._suffix.append(chunk)
                            self._suffix_size += len(chunk)
                            while self._suffix_size > _CAPTURE_BYTES:
                                first = self._suffix.popleft()
                                discard = min(len(first), self._suffix_size - _CAPTURE_BYTES)
                                self._suffix_size -= discard
                                if discard < len(first):
                                    self._suffix.appendleft(first[discard:])
                                self._truncated = True
        except Exception as exc:
            self.error = exc
        finally:
            try:
                self._stream.close()
            except Exception as exc:
                self.error = self.error or exc

    def read(self, size: int = -1) -> bytes:
        with self._lock:
            marker = b"\n[Output truncated]\n" if self._truncated else b""
            output = bytes(self._prefix) + marker + b"".join(self._suffix)
        return output if size < 0 else output[:size]

    def close(self) -> None:
        self._thread.join(timeout=0.25)
        self._stop.set()
        self._thread.join(timeout=0.25)


class OwnedDockerDeployment(DockerDeployment):
    """Keep Docker runtime output and explicit cleanup owned by the executor."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._captures: list[_CapturedPipe] = []
        self._shutdown_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._shutdown_task is not None:
            if not self._shutdown_task.done():
                raise RuntimeError("Docker deployment cleanup is still running")
            self._shutdown_task = None
        await super().start()

    async def _wait_until_alive(self, timeout: float = 10.0) -> None:
        # The SDK calls this immediately after Popen, before its first readiness await.
        process = self._container_process
        assert process is not None
        for name in ("stdout", "stderr"):
            stream = getattr(process, name)
            assert stream is not None
            capture = _CapturedPipe(stream)
            self._captures.append(capture)
            setattr(process, name, capture)
        await super()._wait_until_alive(timeout)

    async def stop(self) -> None:
        """Finish one bounded cleanup even when its caller is cancelled."""
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        task = self._shutdown_task
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        try:
            task.result()
        finally:
            if cancelled:
                raise asyncio.CancelledError

    async def _runtime_command(self, *arguments: str) -> None:
        process = await asyncio.create_subprocess_exec(
            self._config.container_runtime,
            *arguments,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            code = await asyncio.wait_for(process.wait(), timeout=_REMOVE_TIMEOUT)
            if code:
                raise RuntimeError(f"Container cleanup exited with status {code}")
        finally:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=_PROCESS_TIMEOUT)

    async def _shutdown(self) -> None:
        process = self._container_process
        name = self._container_name
        removal_error: Exception | None = None
        try:
            if self._runtime is not None:
                try:
                    await asyncio.wait_for(self._runtime.close(), timeout=_CLOSE_TIMEOUT)
                except (Exception, asyncio.CancelledError) as exc:
                    self.logger.debug("Runtime close failed: %r", exc)
            if name is not None:
                try:
                    if self._config.remove_container:
                        # Podman force-removal otherwise waits the container's stop grace.
                        options = (
                            ("--time", "0")
                            if _is_podman_runtime(self._config.container_runtime)
                            else ()
                        )
                        await self._runtime_command("rm", "-f", *options, name)
                    else:
                        await self._runtime_command("kill", name)
                except Exception as exc:
                    removal_error = exc
            if self._config.remove_images:
                try:
                    await self._runtime_command("rmi", self._config.image)
                except Exception as exc:
                    removal_error = removal_error or exc
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        with contextlib.suppress(ProcessLookupError):
                            process.kill()
                    try:
                        await asyncio.to_thread(process.wait, timeout=_PROCESS_TIMEOUT)
                    except subprocess.TimeoutExpired as exc:
                        removal_error = removal_error or exc
            finally:
                for capture in self._captures:
                    await asyncio.to_thread(capture.close)
                self._captures.clear()
                self._runtime = None
                self._container_process = None
                self._container_name = None
        if removal_error is not None:
            self.logger.warning("Could not confirm removal of %s: %r", name, removal_error)
            raise RuntimeError(f"Could not confirm removal of {name}") from removal_error

    def __del__(self) -> None:
        # SandboxExecutor owns cleanup, including startup failures. The SDK's
        # destructor would otherwise launch an unbounded, unobserved HTTP retry.
        pass

"""Serialize persistent shell operations without blocking the HTTP server."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from swerex.exceptions import CommandTimeoutError
from swerex.runtime import local
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    BashObservation,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    CreateBashSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
)
from swerex.runtime.local import BashSession

_T = TypeVar("_T")


class ResponsiveBashSession(BashSession):
    """Keep one pexpect reader per session while other HTTP requests run."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._operation_lock = asyncio.Lock()
        self._active_worker: asyncio.Task[Any] | None = None
        self._closing = False
        self._interrupt_task: asyncio.Task[None] | None = None

    def _kill_foreground(self, *, allow_shell: bool = True) -> None:
        with contextlib.suppress(RuntimeError, OSError):
            foreground = os.tcgetpgrp(self.shell.child_fd)
            if (
                foreground > 0
                and foreground != os.getpgrp()
                and (allow_shell or foreground != self.shell.pid)
            ):
                os.killpg(foreground, signal.SIGKILL)

    async def _finish_reader_timeout(self) -> None:
        # Ownership is retained, so a non-shell foreground group still belongs
        # to this action, even if its command advanced to another external job.
        self._kill_foreground(allow_shell=False)

        def drain_prompt() -> None:
            self.shell.expect(self._ps1, timeout=1.0)

        # Another SIGINT could flush the completed job's prompt without creating
        # a replacement when the persistent shell ignores SIGINT.
        try:
            await asyncio.to_thread(drain_prompt)
        except Exception as error:
            self._closing = True
            self._kill_foreground(allow_shell=False)

            def close_terminal() -> None:
                shell = self.shell
                try:
                    shell.close(force=True)
                finally:
                    if shell.isalive() and not shell.terminate(force=True):
                        raise RuntimeError("Session terminal could not be terminated")

            try:
                await asyncio.to_thread(close_terminal)
            except Exception as cleanup_error:
                raise error from cleanup_error
            raise

    async def _interrupt_active(self, task: asyncio.Task[Any]) -> None:
        if self._active_worker is not task:
            return
        if self._interrupt_task is None:
            self._interrupt_task = asyncio.create_task(self._stop_active(task))
        await asyncio.shield(self._interrupt_task)

    async def _stop_active(self, task: asyncio.Task[Any]) -> None:
        if self._active_worker is not task:
            return
        if task.done():
            if not task.cancelled() and isinstance(task.exception(), CommandTimeoutError):
                await self._finish_reader_timeout()
            return
        with contextlib.suppress(RuntimeError, OSError):
            self.shell.sendintr()
        try:
            await asyncio.wait_for(asyncio.shield(task), 0.5)
            return
        except CommandTimeoutError:
            await self._finish_reader_timeout()
            return
        except asyncio.TimeoutError:  # noqa: UP041 -- remote Python 3.10
            pass
        except Exception:
            return
        if self._active_worker is not task:
            return
        if task.done():
            if not task.cancelled() and isinstance(task.exception(), CommandTimeoutError):
                await self._finish_reader_timeout()
            return
        # A foreground job may ignore SIGINT. Signal only this terminal's
        # foreground group, leaving the server and other sessions untouched.
        self._kill_foreground()
        try:
            await asyncio.wait_for(asyncio.shield(task), 1.0)
            return
        except CommandTimeoutError:
            await self._finish_reader_timeout()
            return
        except asyncio.TimeoutError:  # noqa: UP041 -- remote Python 3.10
            pass
        except Exception:
            return

        # Closing an unresponsive terminal wakes its remaining pexpect reader.
        def close_terminal() -> None:
            if self._active_worker is task and not task.done():
                self.shell.close(force=True)

        await asyncio.to_thread(close_terminal)
        try:
            await asyncio.shield(task)
        except CommandTimeoutError:
            await self._finish_reader_timeout()
        except Exception:
            pass

    async def _run_worker(
        self, operation: Callable[[], Awaitable[_T]], *, allow_closing: bool = False
    ) -> _T:
        async with self._operation_lock:
            if self._closing and not allow_closing:
                raise RuntimeError("Session is closing")

            async def run_operation() -> _T:
                return await operation()

            def run() -> _T:
                return asyncio.run(run_operation())

            task = asyncio.create_task(asyncio.to_thread(run))
            self._active_worker = task
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                    # Keep the lock until the sole reader has settled.
                    try:
                        await self._interrupt_active(task)
                    except asyncio.CancelledError:
                        continue
                except Exception:
                    break
            # A reader deadline does not stop its command. Keep ownership until
            # the shared interrupt cleanup and its prompt read have settled.
            control = self._interrupt_task
            if control is not None:
                while not control.done():
                    try:
                        await asyncio.shield(control)
                    except asyncio.CancelledError:
                        cancelled = True
                    except Exception:
                        break
                error = control.exception()
                if error is not None:
                    self._closing = True
            else:
                error = None
            self._interrupt_task = None
            self._active_worker = None
            if cancelled:
                with contextlib.suppress(Exception):
                    task.result()
                raise asyncio.CancelledError
            if error is not None:
                raise error
            return task.result()

    async def start(self) -> CreateBashSessionResponse:
        return await self._run_worker(super().start)

    async def run(self, action: BashAction | BashInterruptAction) -> BashObservation:
        if isinstance(action, BashInterruptAction) and self._active_worker is not None:
            await self._interrupt_active(self._active_worker)
            # The existing reader consumed the prompt. Reading it again could
            # block when the shell itself ignores SIGINT.
            return BashObservation(output="", exit_code=0)
        return await self._run_worker(lambda: super(ResponsiveBashSession, self).run(action))

    async def close(self) -> CloseSessionResponse:
        self._closing = True
        if self._active_worker is not None:
            try:
                await self._interrupt_active(self._active_worker)
            except Exception as exc:
                self.logger.warning("Session interrupt cleanup failed during close: %s", exc)
        return await self._run_worker(super().close, allow_closing=True)


class ResponsiveSessionRuntime(local.LocalRuntime):
    """Serialize registry mutations across offloaded shell lifecycles."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._lifecycle_lock = asyncio.Lock()

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        async with self._lifecycle_lock:
            return await super().create_session(request)

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        async with self._lifecycle_lock:
            return await super().close_session(request)

    async def close(self) -> CloseResponse:
        async with self._lifecycle_lock:
            return await super().close()


def install() -> None:
    """Install persistent sessions before the runtime creates any shell."""
    vars(local).update(BashSession=ResponsiveBashSession, LocalRuntime=ResponsiveSessionRuntime)

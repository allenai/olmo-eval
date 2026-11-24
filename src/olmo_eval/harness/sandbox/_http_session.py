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

    # How long an interrupt waits for a job to take the terminal before it
    # signals the shell itself (a builtin such as ``read`` or a shell loop).
    shell_interrupt_grace = 0.25

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

    async def _send_interrupt(self, task: asyncio.Task[Any]) -> None:
        """Send SIGINT to the action's foreground job, or to the shell if it keeps the terminal.

        A ^C written to the terminal reaches whichever group is in the foreground
        at that moment. Between two commands of one line that group is the shell.
        If the shell has already forked the next command, it abandons the line and
        prints a prompt while the child still takes the terminal and runs, so the
        next action queues behind it. Signaling the job's group directly avoids
        that. Bash 4 and later end the rest of the line when its foreground job
        dies of SIGINT, as they do for a terminal ^C.

        Only one job is signaled. A later job may be swerex's own terminator line,
        which must run for the action to finish. Returns when the action finishes,
        a grace period after signaling the job, or after signaling the shell once
        it has held the terminal for the grace period. The caller then escalates.

        One race remains on the shell path: SIGINT sent to the shell just as it
        forks after holding the terminal for the whole grace period.
        """
        loop = asyncio.get_running_loop()
        grace = self.shell_interrupt_grace
        give_up = loop.time() + 2 * grace
        signaled_at: float | None = None
        shell_since: float | None = None
        while not task.done():
            now = loop.time()
            if signaled_at is not None:
                if now - signaled_at >= grace:
                    return
            elif now >= give_up:
                return
            else:
                try:
                    foreground = os.tcgetpgrp(self.shell.child_fd)
                except (RuntimeError, OSError):
                    return
                if foreground > 0 and foreground not in (self.shell.pid, os.getpgrp()):
                    shell_since = None
                    try:
                        os.killpg(foreground, signal.SIGINT)
                    except OSError:
                        pass  # The job already exited; wait for the next one.
                    else:
                        signaled_at = now
                elif shell_since is None:
                    shell_since = now
                elif now - shell_since >= grace:
                    # Only the shell held the terminal, so a builtin or shell loop
                    # is running and only the shell can stop it. A ^C written to
                    # the terminal would also discard unread input, including the
                    # terminator line, so signal the shell's group instead.
                    shell_pid = self.shell.pid
                    if shell_pid is not None:
                        with contextlib.suppress(OSError):
                            os.killpg(shell_pid, signal.SIGINT)
                    return
            await asyncio.sleep(0.005)

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
        await self._send_interrupt(task)
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
                        # Failed cleanup leaves the terminal unusable; the
                        # control-task handling below surfaces the error.
                        self._closing = True
                        break
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

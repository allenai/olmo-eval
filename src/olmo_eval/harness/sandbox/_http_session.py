"""Serialize persistent shell operations without blocking the HTTP server."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from swerex.runtime import local
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    BashObservation,
    CloseSessionResponse,
    CreateBashSessionResponse,
)
from swerex.runtime.local import BashSession

_T = TypeVar("_T")


class ResponsiveBashSession(BashSession):
    """Keep one pexpect reader per session while other HTTP requests run."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._operation_lock = asyncio.Lock()

    async def _run_worker(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        async with self._operation_lock:

            def run() -> _T:
                async def wait() -> _T:
                    return await operation()

                return asyncio.run(wait())

            task = asyncio.create_task(asyncio.to_thread(run))
            cancelled = False
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
                    # Sending a signal does not consume pexpect output. Only the
                    # worker may read the shell until it reaches its deadline.
                    with contextlib.suppress(RuntimeError, OSError):
                        self.shell.sendintr()
                except Exception:
                    break
            if cancelled:
                # Retrieve any worker exception before propagating cancellation.
                with contextlib.suppress(Exception):
                    task.result()
                raise asyncio.CancelledError
            return task.result()

    async def start(self) -> CreateBashSessionResponse:
        return await self._run_worker(super().start)

    async def run(self, action: BashAction | BashInterruptAction) -> BashObservation:
        if isinstance(action, BashInterruptAction) and self._operation_lock.locked():
            self.shell.sendintr()
        return await self._run_worker(lambda: super(ResponsiveBashSession, self).run(action))

    async def close(self) -> CloseSessionResponse:
        return await self._run_worker(super().close)


def install() -> None:
    """Install persistent sessions before the runtime creates any shell."""
    vars(local).update(BashSession=ResponsiveBashSession)

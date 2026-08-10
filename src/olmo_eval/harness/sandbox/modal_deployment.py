"""Modal deployment adapter with reliable HTTP transport and shutdown."""

from __future__ import annotations

import asyncio
import contextlib
import time

import modal  # type: ignore[import-untyped]
from swerex.deployment.modal import ModalDeployment

from .remote_runtime import ReliableRemoteRuntime


class ReliableModalDeployment(ModalDeployment):
    """SWE-ReX Modal deployment with isolated request transport."""

    def __init__(self, *, max_connections: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_connections = max_connections

    async def start(self) -> None:
        if self._runtime is not None and self._sandbox is not None:
            self.logger.warning("Deployment is already started. Ignoring duplicate start() call.")
            return

        self.logger.info("Starting modal sandbox")
        self._hooks.on_custom_step("Starting modal sandbox")
        started_at = time.time()
        token = self._get_token()
        modal_kwargs = dict(self._modal_kwargs)
        modal_kwargs.pop("unencrypted_ports", None)
        modal_kwargs.pop("encrypted_ports", None)
        self._sandbox = await modal.Sandbox.create.aio(
            "/usr/bin/env",
            "bash",
            "-c",
            self._start_swerex_cmd(token),
            image=self._image,
            timeout=int(self._deployment_timeout),
            unencrypted_ports=[self._port],
            app=self._app,
            **modal_kwargs,
        )
        tunnels = await self._sandbox.tunnels.aio()
        tunnel = tunnels[self._port]
        creation_seconds = time.time() - started_at
        self.logger.info(
            "Sandbox (%s) created in %.2fs",
            self._sandbox.object_id,
            creation_seconds,
        )
        self.logger.info("Check sandbox logs at %s", await self.get_modal_log_url())
        await asyncio.sleep(1)
        self.logger.info("Starting runtime at %s", tunnel.url)
        self._hooks.on_custom_step("Starting runtime")
        self._runtime = ReliableRemoteRuntime(
            host=tunnel.url,
            timeout=self._runtime_timeout,
            auth_token=token,
            logger=self.logger,
            max_connections=self._max_connections,
        )
        remaining_timeout = max(0, self._startup_timeout - creation_seconds)
        runtime_started_at = time.time()
        await self._wait_until_alive(timeout=remaining_timeout)
        self.logger.info("Runtime started in %.2fs", time.time() - runtime_started_at)

    async def stop(self) -> None:
        """Terminate Modal directly without calling the runtime over HTTP."""
        # SandboxExecutor already force-terminates first. Keep this idempotent
        # for callers that stop the deployment directly, and never call /close
        # on a server that has already been terminated.
        if self._sandbox is not None:
            with contextlib.suppress(Exception):
                await self._sandbox.terminate.aio()
        self._runtime = None
        self._sandbox = None
        self._app = None

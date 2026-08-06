"""Tests for sandbox executor deployment creation."""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

import pytest

from olmo_eval.harness.sandbox import SandboxConfig, SandboxExecutor, SandboxMode


class _AioMethod:
    def __init__(self, result: Any) -> None:
        self._result = result

    async def aio(self, *args: Any, **kwargs: Any) -> Any:
        if callable(self._result):
            return self._result(*args, **kwargs)
        return self._result


class _FakeHooks:
    def __init__(self) -> None:
        self.steps: list[str] = []

    def on_custom_step(self, step: str) -> None:
        self.steps.append(step)


class _FakeModalDeployment:
    def __init__(
        self,
        *,
        image: Any,
        logger: logging.Logger | None = None,
        startup_timeout: float = 0.4,
        runtime_timeout: float = 3600.0,
        modal_sandbox_kwargs: dict[str, Any] | None = None,
        install_pipx: bool = True,
        deployment_timeout: float = 3600.0,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._image = image
        self._startup_timeout = startup_timeout
        self._runtime_timeout = runtime_timeout
        self._modal_kwargs = modal_sandbox_kwargs or {}
        self._deployment_timeout = deployment_timeout
        self._runtime = None
        self._sandbox = None
        self._port = 8000
        self._app = object()
        self._hooks = _FakeHooks()
        self.wait_until_alive_timeout: float | None = None

    @property
    def runtime(self) -> Any:
        return self._runtime

    def _get_token(self) -> str:
        return "runtime-token"

    def _start_swerex_cmd(self, token: str) -> str:
        return f"start-runtime {token}"

    async def get_modal_log_url(self) -> str:
        return "https://modal.example/logs"

    async def _wait_until_alive(self, *, timeout: float) -> None:
        self.wait_until_alive_timeout = timeout


class _FakeImage:
    @staticmethod
    def from_registry(image: str) -> str:
        return f"registry:{image}"


class _FakeTunnel:
    url = "https://runtime.example"


class _FakeSandbox:
    object_id = "sandbox-123"

    def __init__(self) -> None:
        self.tunnels = _AioMethod({8000: _FakeTunnel()})


class _FakeSandboxCreate:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def aio(self, *args: Any, **kwargs: Any) -> _FakeSandbox:
        self.calls.append((args, kwargs))
        return _FakeSandbox()


class _FakeSandboxApi:
    create = _FakeSandboxCreate()


class _FakeRemoteRuntime:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_modal_fakes(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake_modal = types.ModuleType("modal")
    fake_modal.Image = _FakeImage
    fake_modal.Sandbox = _FakeSandboxApi

    fake_swerex = types.ModuleType("swerex")
    fake_deployment = types.ModuleType("swerex.deployment")
    fake_deployment_modal = types.ModuleType("swerex.deployment.modal")
    fake_deployment_modal.ModalDeployment = _FakeModalDeployment
    fake_runtime = types.ModuleType("swerex.runtime")
    fake_runtime_remote = types.ModuleType("swerex.runtime.remote")
    fake_runtime_remote.RemoteRuntime = _FakeRemoteRuntime

    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "swerex", fake_swerex)
    monkeypatch.setitem(sys.modules, "swerex.deployment", fake_deployment)
    monkeypatch.setitem(sys.modules, "swerex.deployment.modal", fake_deployment_modal)
    monkeypatch.setitem(sys.modules, "swerex.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "swerex.runtime.remote", fake_runtime_remote)

    return fake_modal


@pytest.mark.anyio
async def test_modal_deployment_uses_encrypted_runtime_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_modal = _install_modal_fakes(monkeypatch)
    fake_modal.Sandbox.create.calls.clear()
    executor = SandboxExecutor(
        SandboxConfig(
            image="python:3.12",
            mode=SandboxMode.MODAL,
            startup_timeout=10.0,
            runtime_timeout=20.0,
            modal_sandbox_kwargs={
                "encrypted_ports": [1234],
                "unencrypted_ports": [5678],
                "cloud": "aws",
            },
        )
    )

    deployment = executor.get_deployment()
    await deployment.start()

    assert len(fake_modal.Sandbox.create.calls) == 1
    _, kwargs = fake_modal.Sandbox.create.calls[0]
    assert "unencrypted_ports" not in kwargs
    assert kwargs["encrypted_ports"] == [1234, 8000]
    assert kwargs["cloud"] == "aws"
    assert kwargs["image"] == "registry:python:3.12"
    assert deployment.runtime.kwargs["host"] == "https://runtime.example"
    assert deployment.runtime.kwargs["auth_token"] == "runtime-token"

"""Tests for the reliable Modal deployment adapter."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from olmo_eval.harness.sandbox.modal_deployment import ReliableModalDeployment


@pytest.mark.anyio
async def test_modal_deployment_uses_encrypted_tunnel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    tunnel = SimpleNamespace(url="https://sandbox.example")
    sandbox = SimpleNamespace(
        object_id="sb-1",
        tunnels=SimpleNamespace(aio=_return_value({8880: tunnel})),
        poll=SimpleNamespace(aio=_return_value(None)),
    )

    async def create(*args: object, **kwargs: Any) -> object:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return sandbox

    class _Runtime:
        def __init__(self, **kwargs: Any) -> None:
            captured["runtime"] = kwargs

        async def close(self) -> None:
            pass

    deployment = object.__new__(ReliableModalDeployment)
    deployment._runtime = None
    deployment._sandbox = None
    deployment._port = 8880
    deployment._image = object()
    deployment._deployment_timeout = 600.0
    deployment._runtime_timeout = 60.0
    deployment._startup_timeout = 30.0
    deployment._modal_kwargs = {"unencrypted_ports": [9999], "cpu": 2}
    deployment._app = object()
    deployment._max_connections = 4
    deployment._hooks = SimpleNamespace(on_custom_step=lambda _step: None)
    deployment.logger = logging.getLogger(__name__)
    monkeypatch.setattr(deployment, "_get_token", lambda: "token")
    monkeypatch.setattr(deployment, "_start_swerex_cmd", lambda _token: "start")
    monkeypatch.setattr(deployment, "get_modal_log_url", _return_value("https://logs"))
    monkeypatch.setattr(deployment, "_wait_until_alive", _return_value(None))
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.modal.Sandbox",
        SimpleNamespace(create=SimpleNamespace(aio=create)),
    )
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.ReliableRemoteRuntime",
        _Runtime,
    )
    monkeypatch.setattr(
        "olmo_eval.harness.sandbox.modal_deployment.asyncio.sleep",
        _return_value(None),
    )

    await deployment.start()

    kwargs = captured["kwargs"]
    assert kwargs["encrypted_ports"] == [8880]
    assert "unencrypted_ports" not in kwargs
    assert kwargs["cpu"] == 2
    assert captured["runtime"]["host"] == "https://sandbox.example"
    assert captured["runtime"]["max_connections"] == 4
    await deployment.stop()


def _return_value(value: Any):
    async def inner(*_args: object, **_kwargs: object) -> Any:
        return value

    return inner

"""Tests for the reliable SWE-ReX remote transport."""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from olmo_eval.harness.sandbox.remote_runtime import ReliableRemoteRuntime


class _Result:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    status = 200

    async def json(self) -> dict[str, str]:
        return {"value": "ok"}


class _RequestContext:
    def __init__(self, outcome: Exception | _Response) -> None:
        self.outcome = outcome

    async def __aenter__(self) -> _Response:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def __aexit__(self, *args: object) -> None:
        pass


class _Session:
    closed = False

    def __init__(self, outcomes: list[Exception | _Response]) -> None:
        self.outcomes = outcomes
        self.headers: list[dict[str, str]] = []

    def post(self, _url: str, **kwargs: Any) -> _RequestContext:
        self.headers.append(kwargs["headers"])
        return _RequestContext(self.outcomes.pop(0))


@pytest.mark.anyio
async def test_retries_reuse_request_id_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session(
        [
            aiohttp.ServerDisconnectedError(),
            ConnectionResetError("reset"),
            _Response(),
        ]
    )
    runtime = ReliableRemoteRuntime(
        host="https://sandbox.example",
        auth_token="token",
        timeout=30.0,
        max_connections=2,
    )
    monkeypatch.setattr(runtime, "_get_session", lambda: session)
    monkeypatch.setattr(runtime, "_handle_response_errors", _no_response_error)
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("olmo_eval.harness.sandbox.remote_runtime.asyncio.sleep", record_sleep)

    result = await runtime._request("execute", None, _Result)

    assert result.value == "ok"
    assert delays == [0.25, 0.5]
    assert len({headers["X-Request-ID"] for headers in session.headers}) == 1


@pytest.mark.anyio
async def test_default_transport_retries_is_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session([aiohttp.ServerDisconnectedError() for _ in range(4)])
    runtime = ReliableRemoteRuntime(
        host="https://sandbox.example",
        auth_token="token",
        timeout=30.0,
        max_connections=2,
    )
    monkeypatch.setattr(runtime, "_get_session", lambda: session)
    monkeypatch.setattr(runtime, "_handle_response_errors", _no_response_error)
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("olmo_eval.harness.sandbox.remote_runtime.asyncio.sleep", record_sleep)

    with pytest.raises(aiohttp.ServerDisconnectedError):
        await runtime._request("execute", None, _Result)

    assert len(session.headers) == 4
    assert delays == [0.25, 0.5, 1.0]
    assert len({headers["X-Request-ID"] for headers in session.headers}) == 1


async def _no_response_error(_response: _Response) -> None:
    pass

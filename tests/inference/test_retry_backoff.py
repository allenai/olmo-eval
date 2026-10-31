"""Waits between retries in retry_with_backoff and the judge's retry settings."""

import pytest

from olmo_eval.common.scorers import llm_judge
from olmo_eval.inference import retry


def _record_sleeps(monkeypatch) -> list[float]:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    return sleeps


def _failing(times: int):
    calls = {"n": 0}

    async def call() -> str:
        calls["n"] += 1
        if calls["n"] <= times:
            raise TimeoutError("request timed out")
        return "ok"

    return call, calls


@pytest.mark.anyio
async def test_waits_double_without_a_cap(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    call, _ = _failing(3)

    assert await retry.retry_with_backoff(call, max_retries=3, retry_delay=1.0) == "ok"
    assert sleeps == [1.0, 2.0, 4.0]


@pytest.mark.anyio
async def test_max_delay_caps_each_wait(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    call, _ = _failing(5)

    await retry.retry_with_backoff(call, max_retries=5, retry_delay=5.0, max_delay=30.0)

    assert sleeps == [5.0, 10.0, 20.0, 30.0, 30.0]


@pytest.mark.anyio
async def test_jitter_scales_each_wait_by_half_either_way(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    factors = iter([0.0, 0.999])
    monkeypatch.setattr(retry.random, "random", lambda: next(factors))
    call, _ = _failing(2)

    await retry.retry_with_backoff(call, max_retries=2, retry_delay=4.0, jitter=True)

    assert sleeps == [pytest.approx(2.0), pytest.approx(8 * 1.499)]


@pytest.mark.anyio
async def test_gives_up_after_max_retries(monkeypatch):
    _record_sleeps(monkeypatch)
    call, calls = _failing(10)

    with pytest.raises(TimeoutError):
        await retry.retry_with_backoff(call, max_retries=2, retry_delay=1.0)
    assert calls["n"] == 3


@pytest.mark.anyio
async def test_a_judge_call_waits_out_a_long_stretch_of_failures(monkeypatch):
    sleeps = _record_sleeps(monkeypatch)
    monkeypatch.setattr(retry.random, "random", lambda: 0.5)
    replies: list[str | Exception] = [TimeoutError("request timed out")] * 6 + ["PASS"]

    async def judge(prompt: str) -> str:
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    assert await llm_judge.call_judge(judge, "grade this", label="test judge") == "PASS"
    assert sleeps == [5.0, 10.0, 20.0, 40.0, 80.0, 120.0]


@pytest.mark.anyio
async def test_a_judge_call_that_never_recovers_is_unscored(monkeypatch):
    _record_sleeps(monkeypatch)

    async def judge(prompt: str) -> str:
        raise TimeoutError("request timed out")

    with pytest.raises(llm_judge.ScoringIncompleteError, match="test judge call failed"):
        await llm_judge.call_judge(judge, "grade this", label="test judge")

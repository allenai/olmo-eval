"""Tests for terminal inference-provider error handling."""

from __future__ import annotations

import asyncio
import logging

import pytest

from olmo_eval.inference.dispatch import dispatch_concurrent
from olmo_eval.inference.errors import (
    TerminalProviderError,
    classify_terminal_provider_error,
)


def _engine_dead_error(message: str = "engine died") -> Exception:
    error_type = type(
        "EngineDeadError",
        (Exception,),
        {"__module__": "vllm.v1.engine.exceptions"},
    )
    return error_type(message)


def test_classifies_only_vllm_engine_death_as_terminal() -> None:
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = _engine_dead_error()

    for error in (_engine_dead_error(), wrapped):
        classified = classify_terminal_provider_error(error)
        assert classified is not None
        assert "EngineDeadError" in str(classified)
    assert classify_terminal_provider_error(RuntimeError("request failed")) is None


def test_dispatch_propagates_terminal_error_and_cancels_siblings() -> None:
    async def run() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def process(item: int) -> None:
            if item == 0:
                await sibling_started.wait()
                raise TerminalProviderError("engine died")

            sibling_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        with pytest.raises(TerminalProviderError, match="engine died"):
            await dispatch_concurrent([0, 1], process, max_in_flight=2, max_retries=3)

        assert sibling_cancelled.is_set()

    asyncio.run(run())


def test_dispatch_logs_one_summary_naming_the_first_non_terminal_error(caplog) -> None:
    async def process(item: int) -> int:
        if item % 2:
            raise ValueError(f"request {item} rejected")
        return item

    with caplog.at_level(logging.ERROR, logger="olmo_eval.inference.dispatch"):
        results = asyncio.run(dispatch_concurrent([0, 1, 2, 3], process, max_in_flight=1))

    assert results == [0, None, 2, None]
    summaries = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert summaries == [
        "dispatch: 2/4 requests failed; first error: ValueError: request 1 rejected"
    ]


def test_dispatch_logs_no_summary_when_nothing_failed(caplog) -> None:
    async def process(item: int) -> int:
        return item

    with caplog.at_level(logging.ERROR, logger="olmo_eval.inference.dispatch"):
        asyncio.run(dispatch_concurrent([0, 1], process))

    assert not [r for r in caplog.records if r.levelno == logging.ERROR]

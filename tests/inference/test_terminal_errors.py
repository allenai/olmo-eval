"""Tests for terminal inference-provider error handling."""

from __future__ import annotations

import asyncio

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


def test_classifies_vllm_engine_death_as_terminal() -> None:
    classified = classify_terminal_provider_error(_engine_dead_error())

    assert classified is not None
    assert classified.provider == "vLLM"
    assert classified.cause_type == "EngineDeadError"
    assert "engine died" in str(classified)


def test_classifies_terminal_error_through_exception_chain() -> None:
    wrapped = RuntimeError("provider call failed")
    wrapped.__cause__ = _engine_dead_error()

    classified = classify_terminal_provider_error(wrapped)

    assert classified is not None
    assert classified.cause_type == "EngineDeadError"


def test_does_not_classify_recoverable_provider_error() -> None:
    assert classify_terminal_provider_error(RuntimeError("request failed")) is None


def test_dispatch_propagates_terminal_provider_error() -> None:
    async def process(_item: int) -> int:
        raise TerminalProviderError("vLLM", "EngineDeadError", "engine died")

    with pytest.raises(TerminalProviderError, match="engine died"):
        asyncio.run(dispatch_concurrent([1], process, max_retries=3))


def test_dispatch_cancels_siblings_after_terminal_provider_error() -> None:
    async def run() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def process(item: int) -> None:
            if item == 0:
                await sibling_started.wait()
                raise TerminalProviderError("vLLM", "EngineDeadError", "engine died")

            sibling_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                sibling_cancelled.set()
                raise

        with pytest.raises(TerminalProviderError, match="engine died"):
            await dispatch_concurrent([0, 1], process, max_in_flight=2)

        assert sibling_cancelled.is_set()

    asyncio.run(run())

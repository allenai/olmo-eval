"""Inference provider error classification."""

from __future__ import annotations


class TerminalProviderError(RuntimeError):
    """A provider failure after which its worker cannot serve more requests."""


# Avoid importing optional GPU-only provider packages just to classify errors.
_TERMINAL_PROVIDER_ERRORS = {"vllm.v1.engine.exceptions.EngineDeadError"}


def classify_terminal_provider_error(exc: BaseException) -> TerminalProviderError | None:
    """Return a terminal provider error found in an exception chain, if any."""
    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, TerminalProviderError):
            return current

        error_type = type(current)
        error_name = f"{error_type.__module__}.{error_type.__qualname__}"
        if error_name in _TERMINAL_PROVIDER_ERRORS:
            return TerminalProviderError(f"{error_name}: {current}")

        current = current.__cause__ or current.__context__

    return None


__all__ = ["TerminalProviderError", "classify_terminal_provider_error"]

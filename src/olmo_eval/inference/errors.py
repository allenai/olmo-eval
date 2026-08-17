"""Inference provider error classification."""

from __future__ import annotations


class TerminalProviderError(RuntimeError):
    """A provider failure after which its worker cannot serve more requests."""

    def __init__(
        self,
        provider: str,
        cause_type: str,
        detail: str = "",
    ) -> None:
        self.provider = provider
        self.cause_type = cause_type
        self.detail = detail

        message = f"{provider} provider became unavailable ({cause_type})"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


# Keep optional provider imports out of this module. Classification by fully
# qualified name lets CPU-only installations recognize terminal failures
# without importing CUDA-only packages such as vLLM.
_TERMINAL_PROVIDER_ERRORS = {
    ("vllm.v1.engine.exceptions", "EngineDeadError"): "vLLM",
}


def classify_terminal_provider_error(exc: BaseException) -> TerminalProviderError | None:
    """Return a terminal provider error found in an exception chain, if any."""
    current: BaseException | None = exc
    seen: set[int] = set()

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, TerminalProviderError):
            return current

        error_type = type(current)
        provider = _TERMINAL_PROVIDER_ERRORS.get((error_type.__module__, error_type.__qualname__))
        if provider is not None:
            return TerminalProviderError(
                provider=provider,
                cause_type=error_type.__qualname__,
                detail=str(current).strip(),
            )

        current = current.__cause__ or current.__context__

    return None


__all__ = ["TerminalProviderError", "classify_terminal_provider_error"]

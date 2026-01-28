"""Storage backend implementations."""

from __future__ import annotations

__all__ = [
    "PostgresBackend",
]


def __getattr__(name: str):
    """Lazy import storage backends to avoid heavy dependencies."""
    if name == "PostgresBackend":
        from olmo_eval.storage.backends.postgres import PostgresBackend

        return PostgresBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

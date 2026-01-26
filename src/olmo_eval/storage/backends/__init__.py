"""Storage backend implementations."""

from __future__ import annotations

__all__ = [
    "PostgresBackend",
    "S3Backend",
]


def __getattr__(name: str):
    """Lazy import storage backends to avoid heavy dependencies."""
    if name == "PostgresBackend":
        from olmo_eval.storage.backends.postgres import PostgresBackend

        return PostgresBackend
    elif name == "S3Backend":
        from olmo_eval.storage.backends.s3 import S3Backend

        return S3Backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

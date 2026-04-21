"""Evaluation runners."""

from olmo_eval.runners.common.base import BaseEvalRunner
from olmo_eval.runners.common.constants import ValidationError

__all__ = [
    "AsyncEvalRunner",
    "BaseEvalRunner",
    "ValidationError",
]


def __getattr__(name: str) -> object:
    """Lazily import runner implementations to avoid package import cycles."""
    if name == "AsyncEvalRunner":
        from olmo_eval.runners.asynq import AsyncEvalRunner

        return AsyncEvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

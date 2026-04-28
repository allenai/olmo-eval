"""Async evaluation runner with instance-level queuing."""

__all__ = [
    "AsyncEvalRunner",
]


def __getattr__(name: str) -> object:
    """Lazily import AsyncEvalRunner to avoid package import cycles."""
    if name == "AsyncEvalRunner":
        from olmo_eval.runners.asynq.runner import AsyncEvalRunner

        return AsyncEvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

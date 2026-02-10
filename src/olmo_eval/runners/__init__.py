"""Evaluation runners."""

from olmo_eval.runners.asynq import AsyncEvalRunner
from olmo_eval.runners.common.base import BaseEvalRunner
from olmo_eval.runners.common.constants import ValidationError

# Backwards-compatible alias
EvalRunner = AsyncEvalRunner

__all__ = [
    "AsyncEvalRunner",
    "BaseEvalRunner",
    "EvalRunner",
    "ValidationError",
]

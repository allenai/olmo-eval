"""Evaluation runners."""

from olmo_eval.runners.agent import AgentEvalRunner
from olmo_eval.runners.base import BaseEvalRunner
from olmo_eval.runners.constants import ValidationError
from olmo_eval.runners.simple import AsyncEvalRunner, StreamingEvalRunner, SyncEvalRunner

# Backwards-compatible alias
EvalRunner = SyncEvalRunner

__all__ = [
    "AgentEvalRunner",
    "AsyncEvalRunner",
    "BaseEvalRunner",
    "EvalRunner",
    "StreamingEvalRunner",
    "SyncEvalRunner",
    "ValidationError",
]

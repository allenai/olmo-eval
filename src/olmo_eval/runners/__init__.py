"""Evaluation runners."""

from olmo_eval.runners.asynchronous import AsyncEvalRunner, StreamingEvalRunner
from olmo_eval.runners.synchronous import SyncEvalRunner

# Backwards-compatible alias
EvalRunner = SyncEvalRunner

__all__ = ["SyncEvalRunner", "EvalRunner", "AsyncEvalRunner", "StreamingEvalRunner"]

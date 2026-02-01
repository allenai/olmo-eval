"""Shared task execution utilities for sync and async runners.

This module re-exports symbols from the decomposed modules for backward compatibility.
New code should import directly from the specific modules:
- types: TaskResult, PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
- common: ID generation, sanitization, metadata extraction
- aggregation: compute_suite_aggregations
- builders: build_predictions, build_requests
- writers: write_predictions_jsonl, write_requests_jsonl
- execution: run_task_impl, run_agent_task_impl
"""

from __future__ import annotations

# Re-export all public symbols for backward compatibility
from olmo_eval.runners.aggregation import compute_suite_aggregations
from olmo_eval.runners.builders import build_predictions, build_requests
from olmo_eval.runners.common import (
    compute_task_hash,
    extract_score_from_metrics,
    generate_experiment_id,
    get_author,
    get_git_ref,
    get_metric_metadata,
    get_primary_metric,
    log_task_metrics,
    make_metric_key,
    parse_metric_key,
    sanitize_spec_for_filename,
    serialize_sampling_params,
)
from olmo_eval.runners.execution import run_agent_task_impl, run_task_impl
from olmo_eval.runners.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX, TaskResult
from olmo_eval.runners.writers import write_predictions_jsonl, write_requests_jsonl

__all__ = [
    "PREDICTIONS_SUFFIX",
    "REQUESTS_SUFFIX",
    "TaskResult",
    "build_predictions",
    "build_requests",
    "compute_suite_aggregations",
    "compute_task_hash",
    "extract_score_from_metrics",
    "generate_experiment_id",
    "get_author",
    "get_git_ref",
    "get_metric_metadata",
    "get_primary_metric",
    "log_task_metrics",
    "make_metric_key",
    "parse_metric_key",
    "run_agent_task_impl",
    "run_task_impl",
    "sanitize_spec_for_filename",
    "serialize_sampling_params",
    "write_predictions_jsonl",
    "write_requests_jsonl",
]

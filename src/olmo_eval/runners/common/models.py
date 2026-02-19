"""Dataclasses for runner configuration and metrics output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.utils import Serializable


@dataclass
class S3Config:
    """Configuration for S3 uploads.

    The S3 path structure is:
    s3://{bucket}/{prefix}/{group}/{model_name}_{model_hash_last_6}/{experiment_id}/
        - metrics.json
        - predictions/{task}-predictions.jsonl
        - requests/{task}-requests.jsonl
    """

    bucket: str
    prefix: str  # Base prefix, e.g., "olmo-eval"
    group: str  # Experiment group, e.g., "baseline", "ablation-lr"
    endpoint_url: str | None = None
    region: str = "us-east-1"


@dataclass
class ModelMetadata(Serializable):
    """Model metadata for metrics.json output format."""

    model: str
    provider: str
    dtype: str = "auto"
    tokenizer: str | None = None
    revision: str | None = None
    attention_backend: str | None = None


@dataclass
class TaskMetricsEntry(Serializable):
    """A task entry in the metrics output.

    Metrics are stored in a nested structure: {metric_name: {scorer_name: score}}.
    The primary_metric uses "metric_name:scorer_name" format.
    """

    task: str
    metrics: dict[str, dict[str, float]]
    num_instances: int
    model: str | None = None  # Only set for multi-model format
    primary_metric: str | None = None  # Format: "metric_name:scorer_name"
    config: dict[str, Any] | None = None
    duration_seconds: float | None = None
    task_hash: str | None = None


@dataclass
class ScoreSummary(Serializable):
    """Summary entry with metric:scorer identifier and score.

    The metric field uses "metric_name:scorer_name" format to uniquely
    identify which metric and scorer produced this score.
    """

    metric: str  # Format: "metric_name:scorer_name"
    score: float


@dataclass
class MetricsOutput(Serializable):
    """Top-level metrics.json output structure."""

    timestamp: str
    config: dict[str, Any]  # ProviderConfig.to_dict() or {"models": {name: config}}
    tasks: list[dict[str, Any]]  # List of TaskMetricsEntry.to_dict()
    summary: dict[str, Any]  # task_name -> ScoreSummary or model -> task -> ScoreSummary
    errors: list[dict[str, Any]] = field(default_factory=list)
    # Experiment identification fields for querying results
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    # Duration metrics
    experiment_duration_seconds: float | None = None
    provider_init_seconds: dict[str, float] | None = None  # model_name -> init_time

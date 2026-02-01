"""Dataclasses for runner configuration and metrics output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
class ModelConfig:
    """Model configuration for metrics.json output format.

    Note: There are multiple ModelConfig classes with different purposes:
    - core/configs.py:ModelConfig - Core model config for inference
    - launch/config.py:ModelConfig - Beaker launch config with resource settings
    - runners/models.py:ModelConfig (this one) - Metrics output format for JSON serialization
    """

    model: str
    provider: str
    dtype: str = "auto"
    tokenizer: str | None = None
    revision: str | None = None
    attention_backend: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "dtype": self.dtype,
        }
        if self.tokenizer:
            result["tokenizer"] = self.tokenizer
        if self.revision:
            result["revision"] = self.revision
        if self.attention_backend:
            result["attention_backend"] = self.attention_backend
        return result


@dataclass
class TaskMetricsEntry:
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, excluding None values."""
        result: dict[str, Any] = {
            "task": self.task,
            "metrics": self.metrics,
            "num_instances": self.num_instances,
        }
        if self.model is not None:
            result["model"] = self.model
        if self.primary_metric is not None:
            result["primary_metric"] = self.primary_metric
        if self.config is not None:
            result["config"] = self.config
        if self.duration_seconds is not None:
            result["duration_seconds"] = self.duration_seconds
        if self.task_hash is not None:
            result["task_hash"] = self.task_hash
        return result


@dataclass
class ScoreSummary:
    """Summary entry with metric:scorer identifier and score.

    The metric field uses "metric_name:scorer_name" format to uniquely
    identify which metric and scorer produced this score.
    """

    metric: str  # Format: "metric_name:scorer_name"
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict."""
        return {"metric": self.metric, "score": self.score}


@dataclass
class MetricsOutput:
    """Top-level metrics.json output structure."""

    timestamp: str
    config: dict[str, Any]  # ModelConfig.to_dict() or {"models": {name: config}}
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization, excluding None values."""
        result: dict[str, Any] = {
            "timestamp": self.timestamp,
            "config": self.config,
            "tasks": self.tasks,
            "summary": self.summary,
            "errors": self.errors,
        }
        if self.experiment_id is not None:
            result["experiment_id"] = self.experiment_id
        if self.experiment_name is not None:
            result["experiment_name"] = self.experiment_name
        if self.experiment_group is not None:
            result["experiment_group"] = self.experiment_group
        if self.experiment_duration_seconds is not None:
            result["experiment_duration_seconds"] = self.experiment_duration_seconds
        if self.provider_init_seconds is not None:
            result["provider_init_seconds"] = self.provider_init_seconds
        return result

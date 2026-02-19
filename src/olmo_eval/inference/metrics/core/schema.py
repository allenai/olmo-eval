"""Metrics data schemas.

Frozen dataclasses representing collected metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class RequestMetrics:
    """Metrics for a single inference request."""

    request_id: str
    prompt_tokens: int
    completion_tokens: int
    end_to_end_latency_s: float
    tokens_per_second: float
    time_to_first_token_s: float | None = None
    time_per_output_token_s: float | None = None
    finish_reason: str | None = None
    model: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "end_to_end_latency_s": self.end_to_end_latency_s,
            "tokens_per_second": self.tokens_per_second,
            "time_to_first_token_s": self.time_to_first_token_s,
            "time_per_output_token_s": self.time_per_output_token_s,
            "finish_reason": self.finish_reason,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class BatchMetrics:
    """Aggregated metrics for a batch of requests.

    Core metadata fields mirror the evaluation database schema for join-ability.
    """

    # Aggregate statistics
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    wall_clock_time_s: float
    output_tokens_per_second: float
    mean_latency_s: float
    p50_latency_s: float
    p95_latency_s: float
    p99_latency_s: float

    # Core metadata (mirrors evaluation schema)
    experiment_id: str | None = None
    experiment_name: str | None = None
    experiment_group: str | None = None
    model_name: str | None = None
    model_hash: str | None = None
    task_name: str | None = None
    task_hash: str | None = None
    workspace: str | None = None
    author: str | None = None

    # User-defined tags
    tags: dict[str, str] = field(default_factory=dict)

    # Detailed data
    requests: tuple[RequestMetrics, ...] = ()
    gpu_snapshots: tuple[GPUSnapshot, ...] = ()
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "wall_clock_time_s": self.wall_clock_time_s,
            "output_tokens_per_second": self.output_tokens_per_second,
            "mean_latency_s": self.mean_latency_s,
            "p50_latency_s": self.p50_latency_s,
            "p95_latency_s": self.p95_latency_s,
            "p99_latency_s": self.p99_latency_s,
            "timestamp": self.timestamp.isoformat(),
            "requests": [r.to_dict() for r in self.requests],
            "gpu_snapshots": [g.to_dict() for g in self.gpu_snapshots],
        }
        # Include non-None metadata
        if self.experiment_id is not None:
            d["experiment_id"] = self.experiment_id
        if self.experiment_name is not None:
            d["experiment_name"] = self.experiment_name
        if self.experiment_group is not None:
            d["experiment_group"] = self.experiment_group
        if self.model_name is not None:
            d["model_name"] = self.model_name
        if self.model_hash is not None:
            d["model_hash"] = self.model_hash
        if self.task_name is not None:
            d["task_name"] = self.task_name
        if self.task_hash is not None:
            d["task_hash"] = self.task_hash
        if self.workspace is not None:
            d["workspace"] = self.workspace
        if self.author is not None:
            d["author"] = self.author
        if self.tags:
            d["tags"] = dict(self.tags)
        return d


@dataclass(frozen=True)
class GPUSnapshot:
    """GPU utilization snapshot at a point in time."""

    device_id: int
    name: str
    utilization_pct: float
    memory_used_mb: float
    memory_total_mb: float
    temperature_c: float | None = None
    power_watts: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "utilization_pct": self.utilization_pct,
            "memory_used_mb": self.memory_used_mb,
            "memory_total_mb": self.memory_total_mb,
            "temperature_c": self.temperature_c,
            "power_watts": self.power_watts,
            "timestamp": self.timestamp.isoformat(),
        }

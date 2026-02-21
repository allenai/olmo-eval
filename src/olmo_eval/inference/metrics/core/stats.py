"""Statistics computation for metrics."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .schema import BatchMetrics, GPUSnapshot, RequestMetrics

if TYPE_CHECKING:
    from .config import MetricsConfig


def compute_batch_hash(native_ids: list[str]) -> str:
    """Compute a hash of the batch based on native instance IDs.

    Same batch composition with same order produces same hash.
    Deterministic across runs when using stable native_ids.
    """
    if not native_ids:
        return "empty"
    id_string = "|".join(native_ids)
    return hashlib.sha256(id_string.encode()).hexdigest()[:12]


def compute_batch_metrics(
    requests: list[RequestMetrics],
    wall_clock_s: float,
    batch_hash: str,
    config: MetricsConfig | None = None,
    gpu_snapshots: tuple[GPUSnapshot, ...] = (),
) -> BatchMetrics:
    """Compute aggregate metrics from a list of request metrics.

    Args:
        requests: List of RequestMetrics from individual requests.
        wall_clock_s: Total wall clock time for the batch.
        batch_hash: Batch hash computed from native instance IDs.
        config: Optional MetricsConfig to extract metadata from.
        gpu_snapshots: Optional GPU utilization snapshots.

    Returns:
        BatchMetrics with aggregated statistics.
    """
    # Extract metadata from config if provided
    experiment_id = config.experiment_id if config else None
    experiment_name = config.experiment_name if config else None
    experiment_group = config.experiment_group if config else None
    model_name = config.model_name if config else None
    model_hash = config.model_hash if config else None
    task_name = config.task_name if config else None
    task_hash = config.task_hash if config else None
    workspace = config.workspace if config else None
    author = config.author if config else None
    provider_kind = config.provider_kind if config else None
    tags = config.tags if config else {}

    if not requests:
        return BatchMetrics(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            wall_clock_time_s=wall_clock_s,
            output_tokens_per_second=0.0,
            mean_latency_s=0.0,
            batch_hash=batch_hash,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            model_name=model_name,
            model_hash=model_hash,
            task_name=task_name,
            task_hash=task_hash,
            workspace=workspace,
            author=author,
            provider_kind=provider_kind,
            tags=tags,
            gpu_snapshots=gpu_snapshots,
        )

    total_requests = len(requests)
    # Consider requests with completion_tokens > 0 as successful
    successful = [r for r in requests if r.completion_tokens > 0]
    successful_requests = len(successful)
    failed_requests = total_requests - successful_requests

    total_prompt_tokens = sum(r.prompt_tokens for r in requests)
    total_completion_tokens = sum(r.completion_tokens for r in requests)

    latencies = [r.end_to_end_latency_s for r in requests]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # If wall_clock_s not provided, use sum of latencies as approximation
    effective_wall_clock = wall_clock_s if wall_clock_s > 0 else sum(latencies)
    output_tps = total_completion_tokens / effective_wall_clock if effective_wall_clock > 0 else 0.0

    return BatchMetrics(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_clock_time_s=effective_wall_clock,
        output_tokens_per_second=output_tps,
        mean_latency_s=mean_latency,
        batch_hash=batch_hash,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        model_name=model_name,
        model_hash=model_hash,
        task_name=task_name,
        task_hash=task_hash,
        workspace=workspace,
        author=author,
        provider_kind=provider_kind,
        tags=tags,
        gpu_snapshots=gpu_snapshots,
    )

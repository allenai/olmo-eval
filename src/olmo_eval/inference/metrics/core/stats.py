"""Statistics computation for metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import BatchMetrics, RequestMetrics

if TYPE_CHECKING:
    from .config import MetricsConfig


def percentile(values: list[float], p: float) -> float:
    """Compute percentile of a list of values.

    Args:
        values: List of numeric values (must be non-empty).
        p: Percentile to compute (0-100).

    Returns:
        The p-th percentile value.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    # Linear interpolation method
    k = (n - 1) * (p / 100.0)
    f = int(k)
    c = f + 1

    if c >= n:
        return sorted_values[-1]

    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


def compute_batch_metrics(
    requests: list[RequestMetrics],
    wall_clock_s: float,
    config: MetricsConfig | None = None,
) -> BatchMetrics:
    """Compute aggregate metrics from a list of request metrics.

    Args:
        requests: List of RequestMetrics from individual requests.
        wall_clock_s: Total wall clock time for the batch.
        config: Optional MetricsConfig to extract metadata from.

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
            p50_latency_s=0.0,
            p95_latency_s=0.0,
            p99_latency_s=0.0,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            experiment_group=experiment_group,
            model_name=model_name,
            model_hash=model_hash,
            task_name=task_name,
            task_hash=task_hash,
            workspace=workspace,
            author=author,
            tags=tags,
            requests=tuple(requests),
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

    output_tps = total_completion_tokens / wall_clock_s if wall_clock_s > 0 else 0.0

    return BatchMetrics(
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        wall_clock_time_s=wall_clock_s,
        output_tokens_per_second=output_tps,
        mean_latency_s=mean_latency,
        p50_latency_s=percentile(latencies, 50),
        p95_latency_s=percentile(latencies, 95),
        p99_latency_s=percentile(latencies, 99),
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        experiment_group=experiment_group,
        model_name=model_name,
        model_hash=model_hash,
        task_name=task_name,
        task_hash=task_hash,
        workspace=workspace,
        author=author,
        tags=tags,
        requests=tuple(requests),
    )

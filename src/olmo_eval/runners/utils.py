"""Shared task execution utilities for sync and async runners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.backends import Backend
from olmo_eval.core import Response
from olmo_eval.evals.tasks import get_task


def get_primary_metric(metrics: dict[str, float]) -> tuple[str, float] | None:
    """Get the primary metric name and value from a metrics dict.

    Priority:
    1. "accuracy" if present (most common metric)
    2. First metric alphabetically (for determinism)

    Args:
        metrics: Dictionary of metric names to values

    Returns:
        Tuple of (metric_name, metric_value), or None if metrics is empty
    """
    if not metrics:
        return None

    if "accuracy" in metrics:
        return ("accuracy", metrics["accuracy"])

    # Fallback: first metric alphabetically for determinism
    name = sorted(metrics.keys())[0]
    return (name, metrics[name])


@dataclass
class TaskResult:
    """Result from executing a single task."""

    spec: str
    config: dict[str, Any]
    num_instances: int
    metrics: dict[str, float]
    error: str | None = None
    duration_seconds: float = 0.0


def run_task_impl(
    spec: str,
    backend: Backend,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> TaskResult:
    """Execute a single task and return results.

    This is the core task execution logic shared by both EvalRunner and AsyncEvalRunner.

    Args:
        spec: Task specification (e.g., "mmlu_history" or "mmlu_history::olmes")
        backend: Backend instance to use for generation
        overrides: Optional overrides for task config (num_fewshot, limit)
        progress_callback: Optional callback for progress messages

    Returns:
        TaskResult with metrics and metadata

    Raises:
        Exception: Any error during task execution (should be caught by caller)
    """
    import time

    start_time = time.time()

    try:
        # Get task
        task = get_task(spec)

        # Apply overrides
        if overrides:
            task.config = replace(task.config, **overrides)

        # Collect instances
        instances = list(task.instances)
        if task.config.limit:
            instances = instances[: task.config.limit]

        if progress_callback:
            progress_callback(f"Evaluating {len(instances)} instances...")

        # Format requests
        requests = [task.format_request(inst) for inst in instances]

        # Generate outputs
        outputs = backend.generate(requests, task.config.sampling_params)

        # Build responses
        responses = [
            Response(instance=inst, request=req, outputs=out)
            for inst, req, out in zip(instances, requests, outputs, strict=True)
        ]

        # Score and compute metrics
        scored = task.score_responses(responses)
        metrics = task.compute_metrics(scored)

        duration = time.time() - start_time

        return TaskResult(
            spec=spec,
            config={
                "name": task.config.name,
                "split": task.config.split.value,
                "num_fewshot": task.config.num_fewshot,
                "limit": task.config.limit,
            },
            num_instances=len(instances),
            metrics=metrics,
            duration_seconds=duration,
        )

    except Exception as e:
        duration = time.time() - start_time
        return TaskResult(
            spec=spec,
            config={},
            num_instances=0,
            metrics={},
            error=str(e),
            duration_seconds=duration,
        )

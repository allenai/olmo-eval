"""Shared task execution utilities for sync and async runners."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.backends import Backend
from olmo_eval.core import Response, SamplingParams
from olmo_eval.evals.tasks import get_task

# Re-export build_predictions for parallel runner
__all__ = [
    "TaskResult",
    "build_predictions",
    "compute_suite_aggregations",
    "get_primary_metric",
    "run_task_impl",
]


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


def compute_suite_aggregations(
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute aggregated metrics for suites in the task specs.

    For each suite in task_specs, computes the average of primary metrics
    across all tasks in that suite.

    Handles specs with inline overrides (::key=value) and priority suffixes (@priority).
    When a suite has these suffixes, they are propagated to expanded task lookups.

    Args:
        task_specs: Original task specs (may include suite names with overrides/priority)
        task_results: Dict mapping task spec -> {"metrics": {...}, ...}

    Returns:
        Dict mapping suite name -> {"metrics": {...}, "tasks": [...], "aggregation": ...}
    """
    from olmo_eval.evals.suites import get_suite, suite_exists
    from olmo_eval.evals.suites.registry import AggregationStrategy

    suite_aggregations: dict[str, dict[str, Any]] = {}

    for spec in task_specs:
        # Parse out priority suffix first (e.g., "suite::temp=0@high" -> "suite::temp=0", "@high")
        priority_suffix = ""
        spec_without_priority = spec
        if "@" in spec:
            spec_without_priority, priority = spec.rsplit("@", 1)
            priority_suffix = f"@{priority}"

        # Parse out overrides (e.g., "suite:variant::temp=0" -> "suite:variant", "::temp=0")
        override_suffix = ""
        base_spec = spec_without_priority
        if "::" in spec_without_priority:
            base_spec, overrides = spec_without_priority.split("::", 1)
            override_suffix = f"::{overrides}"

        # Check if the base spec (without suffixes) is a suite
        if not suite_exists(base_spec):
            continue

        suite = get_suite(base_spec)
        if suite.aggregation == AggregationStrategy.NONE:
            continue

        # Get results for all tasks in this suite
        # Note: suite.expand() returns task specs without suffixes
        suite_tasks = suite.expand()
        suite_metrics: dict[str, list[float]] = {}
        tasks_included = []

        for task_spec in suite_tasks:
            # Build the full task spec with the same suffixes as the suite
            full_task_spec = f"{task_spec}{override_suffix}{priority_suffix}"

            if full_task_spec not in task_results:
                continue

            task_data = task_results[full_task_spec]
            metrics = task_data.get("metrics", {})

            if not metrics:
                continue

            tasks_included.append(full_task_spec)

            # Collect all metrics for averaging
            for metric_name, value in metrics.items():
                if metric_name not in suite_metrics:
                    suite_metrics[metric_name] = []
                suite_metrics[metric_name].append(value)

        if not suite_metrics:
            continue

        # Compute averages
        aggregated_metrics = {
            name: sum(values) / len(values) for name, values in suite_metrics.items()
        }

        suite_aggregations[spec] = {
            "metrics": aggregated_metrics,
            "tasks": tasks_included,
            "num_tasks": len(tasks_included),
            "aggregation": suite.aggregation.value,
        }

    return suite_aggregations


@dataclass
class TaskResult:
    """Result from executing a single task."""

    spec: str
    config: dict[str, Any]
    num_instances: int
    metrics: dict[str, float]
    error: str | None = None
    duration_seconds: float = 0.0
    predictions: list[dict] | None = None


def build_predictions(scored: Sequence[Response]) -> list[dict]:
    """Build per-instance predictions from scored responses.

    Args:
        scored: Sequence of scored Response objects

    Returns:
        List of prediction dicts suitable for JSONL output
    """
    predictions = []
    for idx, resp in enumerate(scored):
        # Build doc from instance
        doc: dict[str, Any] = {"query": resp.instance.question}
        if resp.instance.choices:
            doc["choices"] = list(resp.instance.choices)
        if resp.instance.gold_answer is not None:
            # Use gold_idx from metadata if available, otherwise use gold_answer
            doc["gold"] = resp.instance.metadata.get("gold_idx", resp.instance.gold_answer)

        # Build model_output from LMOutput objects
        model_output = []
        for out in resp.outputs:
            out_data: dict[str, Any] = {"text": out.text}
            if out.logprobs:
                out_data["sum_logprob"] = sum(t.get("logprob", 0) for t in out.logprobs)
                out_data["num_tokens"] = len(out.logprobs)
            out_data["num_bytes"] = len(out.text.encode("utf-8"))
            model_output.append(out_data)

        predictions.append(
            {
                "doc_id": idx,
                "native_id": resp.instance.metadata.get("id", f"doc_{idx}"),
                "doc": doc,
                "model_output": model_output,
                "scores": dict(resp.scores),
            }
        )

    return predictions


def run_task_impl(
    spec: str,
    backend: Backend,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    temperature: float | None = None,
    sampling_overrides: dict[str, Any] | None = None,
) -> TaskResult:
    """Execute a single task and return results.

    This is the core task execution logic shared by both EvalRunner and AsyncEvalRunner.

    Args:
        spec: Task specification (e.g., "mmlu_history" or "mmlu_history:olmes")
        backend: Backend instance to use for generation
        overrides: Optional overrides for task config (num_fewshot, limit, fewshot_seed)
        progress_callback: Optional callback for progress messages
        temperature: Optional temperature for sampling (deprecated, use sampling_overrides)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)

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

        # Build sampling params from overrides
        # Priority: sampling_overrides > temperature > task default
        existing_params = task.config.sampling_params or SamplingParams()

        # Apply legacy temperature parameter (deprecated)
        if temperature is not None:
            existing_params = replace(existing_params, temperature=temperature)

        # Apply sampling_overrides (highest priority)
        if sampling_overrides:
            for key, value in sampling_overrides.items():
                if hasattr(existing_params, key):
                    existing_params = replace(existing_params, **{key: value})

        # Update task config with final sampling params
        if temperature is not None or sampling_overrides:
            task.config = replace(task.config, sampling_params=existing_params)

        # Collect instances
        instances = list(task.instances)
        if task.config.limit:
            instances = instances[: task.config.limit]

        if progress_callback:
            progress_callback(f"Evaluating {len(instances)} instances...")

        # Format requests
        requests = [task.format_request(inst) for inst in instances]

        # Generate outputs - use logprobs for LOGLIKELIHOOD requests
        from olmo_eval.core import RequestType

        if requests and requests[0].request_type == RequestType.LOGLIKELIHOOD:
            outputs = backend.logprobs(requests)
        else:
            outputs = backend.generate(requests, task.config.sampling_params)

        # Build responses
        responses = [
            Response(instance=inst, request=req, outputs=out)
            for inst, req, out in zip(instances, requests, outputs, strict=True)
        ]

        # Score and compute metrics
        scored = task.score_responses(responses)
        metrics = task.compute_metrics(scored)

        # Build predictions for per-instance inspection
        predictions = build_predictions(scored)

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
            predictions=predictions,
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

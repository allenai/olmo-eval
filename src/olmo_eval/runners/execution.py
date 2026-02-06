"""Task execution functions for running evaluations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from olmo_eval.core.logging import get_logger
from olmo_eval.core.types import Response, SamplingParams
from olmo_eval.evals.tasks import get_task
from olmo_eval.inference import InferenceProvider
from olmo_eval.runners.builders import build_predictions, build_requests
from olmo_eval.runners.common import get_metric_metadata
from olmo_eval.runners.types import TaskResult

logger = get_logger("runners.execution")


def run_task_impl(
    spec: str,
    provider: InferenceProvider,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    temperature: float | None = None,
    sampling_overrides: dict[str, Any] | None = None,
    requests_callback: Callable[[list[dict]], None] | None = None,
    response_callback: Callable[[Response], None] | None = None,
) -> TaskResult:
    """Execute a single task and return results.

    This is the core task execution logic shared by both EvalRunner and AsyncEvalRunner.

    Args:
        spec: Task specification (e.g., "mmlu_history" or "mmlu_history:olmes")
        provider: InferenceProvider instance to use for generation
        overrides: Optional overrides for task config (num_fewshot, limit, fewshot_seed)
        progress_callback: Optional callback for progress messages
        temperature: Optional temperature for sampling (deprecated, use sampling_overrides)
        sampling_overrides: Optional overrides for sampling params (temperature, max_tokens, etc.)
        requests_callback: Optional callback to receive requests early (before generation).
            Called with the list of request dicts immediately after they're built.
            Use this to write requests.jsonl before waiting for generation to complete.
        response_callback: Optional callback to receive the first scored response.
            Called after scoring with the first Response object. Useful for inspection/debugging.

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

        # Build requests in oe-eval compatible format (for debugging what model saw)
        # We do this early since we know the requests upfront - no need to wait for generation
        request_objects = build_requests(instances, requests, task.config.name, existing_params)

        # Call the requests callback early (before generation) if provided
        # This allows writing requests.jsonl without waiting for generation to complete
        if requests_callback:
            requests_callback(request_objects)

        # Generate outputs - use logprobs for LOGLIKELIHOOD requests
        from olmo_eval.core.types import RequestType

        if requests and requests[0].request_type == RequestType.LOGLIKELIHOOD:
            outputs = provider.logprobs(requests)
        else:
            outputs = provider.generate(requests, task.config.sampling_params)

        # Build responses
        responses = [
            Response(instance=inst, request=req, outputs=out)
            for inst, req, out in zip(instances, requests, outputs, strict=True)
        ]

        # Score and compute metrics
        scored = task.score_responses(responses)
        metrics = task.compute_metrics(scored)

        # Call the response callback with first scored response if provided
        if response_callback and scored:
            response_callback(scored[0])

        # Build predictions for per-instance inspection
        predictions = build_predictions(scored)

        duration = time.time() - start_time

        # Extract metric metadata (returns "metric:scorer" format)
        primary_metric = get_metric_metadata(task)

        return TaskResult(
            spec=spec,
            config=task.config.to_dict(),
            num_instances=len(instances),
            metrics=metrics,
            duration_seconds=duration,
            predictions=predictions,
            requests=request_objects,
            primary_metric=primary_metric,
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

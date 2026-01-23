"""Shared task execution utilities for sync and async runners."""

from __future__ import annotations

import json
import logging
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from olmo_eval.backends import Backend
from olmo_eval.core import Response, SamplingParams
from olmo_eval.evals.tasks import get_task

# Re-export build_predictions for parallel runner
__all__ = [
    "PREDICTIONS_SUFFIX",
    "REQUESTS_SUFFIX",
    "TaskResult",
    "build_predictions",
    "build_requests",
    "compute_suite_aggregations",
    "get_primary_metric",
    "run_task_impl",
    "sanitize_spec_for_filename",
    "serialize_sampling_params",
    "write_predictions_jsonl",
    "write_requests_jsonl",
]

logger = logging.getLogger(__name__)

# Filename suffixes for output files (consistent across all runners and storage backends)
PREDICTIONS_SUFFIX = "-predictions.jsonl"
REQUESTS_SUFFIX = "-requests.jsonl"


def sanitize_spec_for_filename(spec: str) -> str:
    """Sanitize a task spec string to be safe for use in filenames.

    Replaces characters that are not allowed or problematic in filenames:
    - ':' -> '_' (used in task variants like arc_challenge:bpb)
    - '/' -> '_' (could appear in model names or paths)
    - '\\' -> '_' (Windows path separator)
    - ' ' -> '_' (spaces)

    Args:
        spec: Task specification string (e.g., "arc_challenge:bpb::olmes")

    Returns:
        Sanitized string safe for use in filenames (e.g., "arc_challenge_bpb__olmes")

    Examples:
        >>> sanitize_spec_for_filename("arc_challenge:bpb")
        'arc_challenge_bpb'
        >>> sanitize_spec_for_filename("mmlu/history")
        'mmlu_history'
        >>> sanitize_spec_for_filename("task::key=value")
        'task__key=value'
    """
    # Replace problematic characters with underscores
    result = spec.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
    return result


def serialize_sampling_params(params: SamplingParams | None) -> dict[str, Any] | None:
    """Serialize SamplingParams to a dictionary for JSON output.

    Args:
        params: SamplingParams instance or None

    Returns:
        Dictionary representation or None if params is None
    """
    if params is None:
        return None
    return {
        "temperature": params.temperature,
        "max_tokens": params.max_tokens,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "num_samples": params.num_samples,
    }


def get_primary_metric(
    metrics: dict[str, float],
    preferred: str | None = None,
) -> tuple[str, float] | None:
    """Get the primary metric name and value from a metrics dict.

    Priority:
    1. User-specified preferred metric (if provided and present)
    2. "accuracy" if present (most common metric)
    3. First metric alphabetically (for determinism)

    Args:
        metrics: Dictionary of metric names to values
        preferred: Optional preferred metric name (from task config)

    Returns:
        Tuple of (metric_name, metric_value), or None if metrics is empty
    """
    if not metrics:
        return None

    # Use preferred metric if specified and present
    if preferred and preferred in metrics:
        return (preferred, metrics[preferred])

    # Default fallback: accuracy first
    if "accuracy" in metrics:
        return ("accuracy", metrics["accuracy"])

    # Fallback: first metric alphabetically for determinism
    name = sorted(metrics.keys())[0]
    return (name, metrics[name])


@dataclass
class _ChildAverageResult:
    """Result from computing a child average."""

    metrics: dict[str, float]
    tasks: list[str]
    # If child was a Suite, include its info for separate reporting
    nested_suite: Any | None = None  # Suite or None
    nested_suite_key: str | None = None  # Key to use in results (with suffixes)


def _compute_child_average(
    child: str | Any,  # str or Suite
    override_suffix: str,
    priority_suffix: str,
    task_results: dict[str, dict[str, Any]],
) -> _ChildAverageResult | None:
    """Compute average metrics for a single child (task string or nested Suite).

    Returns:
        _ChildAverageResult with metrics and task info, or None if no results found.
    """
    from olmo_eval.evals.suites.registry import Suite

    if isinstance(child, Suite):
        # Child is a nested Suite - average all its expanded tasks
        child_metrics: dict[str, list[float]] = {}
        tasks_included = []

        for task_spec in child.expand():
            full_task_spec = f"{task_spec}{override_suffix}{priority_suffix}"
            if full_task_spec not in task_results:
                continue

            task_data = task_results[full_task_spec]
            metrics = task_data.get("metrics", {})
            if not metrics:
                continue

            tasks_included.append(full_task_spec)
            for metric_name, value in metrics.items():
                if metric_name not in child_metrics:
                    child_metrics[metric_name] = []
                child_metrics[metric_name].append(value)

        if not child_metrics:
            return None

        averaged = {name: sum(vals) / len(vals) for name, vals in child_metrics.items()}
        # Build the key for this nested suite (with suffixes)
        nested_key = f"{child.name}{override_suffix}{priority_suffix}"
        return _ChildAverageResult(
            metrics=averaged,
            tasks=tasks_included,
            nested_suite=child,
            nested_suite_key=nested_key,
        )
    else:
        # Child is a task string - get its metrics directly
        full_task_spec = f"{child}{override_suffix}{priority_suffix}"
        if full_task_spec not in task_results:
            return None

        task_data = task_results[full_task_spec]
        metrics = task_data.get("metrics", {})
        if not metrics:
            return None

        return _ChildAverageResult(
            metrics=dict(metrics),
            tasks=[full_task_spec],
            nested_suite=None,
            nested_suite_key=None,
        )


def compute_suite_aggregations(
    task_specs: list[str],
    task_results: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute aggregated metrics for suites in the task specs.

    For each suite in task_specs, computes aggregated metrics based on the
    suite's aggregation strategy:
    - AVERAGE: Simple average of all expanded task scores
    - AVERAGE_OF_AVERAGES: Average over children, where nested suites are
      averaged first (each child gets equal weight)

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

        if suite.aggregation == AggregationStrategy.AVERAGE_OF_AVERAGES:
            # Average of averages: each child (task or nested suite) gets equal weight
            # Process each child separately, then average the child averages
            child_averages: dict[str, list[float]] = {}
            all_tasks_included: list[str] = []
            children_included = 0
            nested_suites_included: list[str] = []

            for child in suite.tasks:
                result = _compute_child_average(
                    child, override_suffix, priority_suffix, task_results
                )
                if result is None:
                    continue

                all_tasks_included.extend(result.tasks)
                children_included += 1

                for metric_name, value in result.metrics.items():
                    if metric_name not in child_averages:
                        child_averages[metric_name] = []
                    child_averages[metric_name].append(value)

                # If this child is a nested Suite, also report its aggregation separately
                if result.nested_suite is not None and result.nested_suite_key:
                    nested_suites_included.append(result.nested_suite_key)
                    suite_aggregations[result.nested_suite_key] = {
                        "metrics": result.metrics,
                        "tasks": result.tasks,
                        "num_tasks": len(result.tasks),
                        "aggregation": result.nested_suite.aggregation.value,
                        "parent_suite": spec,  # Track which parent suite this belongs to
                    }

            if not child_averages:
                continue

            # Average the child averages (each child weighted equally)
            aggregated_metrics = {
                name: sum(values) / len(values) for name, values in child_averages.items()
            }

            suite_aggregations[spec] = {
                "metrics": aggregated_metrics,
                "tasks": all_tasks_included,
                "num_tasks": len(all_tasks_included),
                "num_children": children_included,
                "nested_suites": nested_suites_included,
                "aggregation": suite.aggregation.value,
            }
        else:
            # AVERAGE or DISPLAY_ONLY: simple average of all expanded tasks
            suite_tasks = suite.expand()
            suite_metrics: dict[str, list[float]] = {}
            tasks_included: list[str] = []

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
    requests: list[dict] | None = None  # oe-eval compatible request objects
    primary_metric: str | None = None  # Preferred metric name from task config


def build_predictions(scored: Sequence[Response]) -> list[dict]:
    """Build per-instance predictions from scored responses.

    Args:
        scored: Sequence of scored Response objects

    Returns:
        List of prediction dicts suitable for JSONL output
    """
    predictions = []
    for idx, resp in enumerate(scored):
        # Build model_output from LMOutput objects
        model_output = []
        for out in resp.outputs:
            # Get values from metadata (set by backend) or compute from logprobs
            meta = out.metadata or {}
            num_bytes = len(out.text.encode("utf-8"))
            num_chars = len(out.text)

            # Use metadata values if available (from vLLM backend), else compute
            if "sum_logits" in meta:
                sum_logits = meta["sum_logits"]
                num_tokens = meta.get("num_tokens", len(out.logprobs) if out.logprobs else 0)
            elif out.logprobs:
                sum_logits = sum(t.get("logprob", 0) for t in out.logprobs)
                num_tokens = len(out.logprobs)
            else:
                sum_logits = 0.0
                num_tokens = 0

            out_data: dict[str, Any] = {
                "sum_logits": sum_logits,
                "num_tokens": num_tokens,
                "num_tokens_all": meta.get("num_tokens_all", num_tokens),
                "is_greedy": meta.get("is_greedy", False),
            }

            # Compute derived metrics (matching oe-eval format)
            if num_tokens > 0:
                out_data["logits_per_token"] = sum_logits / num_tokens
            if num_chars > 0:
                out_data["logits_per_char"] = sum_logits / num_chars
            if num_bytes > 0:
                out_data["bits_per_byte"] = -sum_logits / (num_bytes * math.log(2))

            out_data["num_chars"] = num_chars

            model_output.append(out_data)

        # Get label from metadata or gold_answer
        label = resp.instance.metadata.get("gold_idx", resp.instance.gold_answer)

        # Build prediction (doc and context available in requests.jsonl)
        prediction: dict[str, Any] = {
            "doc_id": idx,
            "native_id": resp.instance.metadata.get("id", f"doc_{idx}"),
            "model_output": model_output,
            "metrics": dict(resp.scores),
            "label": label,
        }

        predictions.append(prediction)

    return predictions


def build_requests(
    instances: Sequence[Any],
    requests: Sequence[Any],
    task_name: str,
    sampling_params: SamplingParams | None = None,
) -> list[dict]:
    """Build per-instance request objects in oe-eval compatible format.

    This produces the same format as oe-eval's *-requests.jsonl files, which
    shows exactly what the model saw during evaluation.

    Args:
        instances: Sequence of Instance objects
        requests: Sequence of LMRequest objects (parallel to instances)
        task_name: Name of the task
        sampling_params: Optional sampling parameters

    Returns:
        List of request dicts suitable for JSONL output with oe-eval compatible schema:
        {
            "request_type": "loglikelihood" | "generate_until" | ...,
            "doc": {
                "query": "...",  # The instance question
                "choices": [...],  # For BPB/MC tasks
                # Plus original metadata
            },
            "request": {
                "context": "...",  # Full prompt (few-shot + current)
                "continuation": "...",  # For loglikelihood: the text being scored
                "perplexity_context": "...",  # Usually same as context
                "stop_sequences": [...],
                "generation_kwargs": {...}
            },
            "idx": 0,
            "task_name": "...",
            "doc_id": 0,
            "native_id": "...",
            "label": ...
        }
    """
    from olmo_eval.core import RequestType

    request_list = []

    for idx, (instance, request) in enumerate(zip(instances, requests, strict=True)):
        # Build doc from instance
        doc: dict[str, Any] = {
            "query": instance.question,
            **instance.metadata,
        }

        # Add choices for BPB and MC tasks
        if instance.choices:
            doc["choices"] = list(instance.choices)
        elif request.continuations:
            # For BPB tasks without explicit choices, use continuations
            doc["choices"] = list(request.continuations)

        # Build request object based on request type
        request_dict: dict[str, Any] = {}

        if request.request_type == RequestType.LOGLIKELIHOOD:
            # oe-eval's GenerateUntilAndLoglikelihoodRequest format
            request_dict["context"] = request.prompt
            request_dict["perplexity_context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
        elif request.request_type == RequestType.COMPLETION:
            # oe-eval's GenerateUntilRequest format
            request_dict["context"] = request.prompt
            if request.continuations:
                request_dict["continuation"] = request.continuations[0]
        elif request.request_type == RequestType.CHAT:
            # Chat format - context is the messages
            request_dict["context"] = list(request.messages)

        # Add generation kwargs
        if sampling_params:
            request_dict["stop_sequences"] = (
                list(sampling_params.stop_sequences) if sampling_params.stop_sequences else []
            )
            request_dict["generation_kwargs"] = {
                "max_gen_toks": sampling_params.max_tokens,
                "do_sample": sampling_params.temperature > 0,
                "temperature": sampling_params.temperature,
            }
            if sampling_params.top_p is not None:
                request_dict["generation_kwargs"]["top_p"] = sampling_params.top_p
        else:
            request_dict["stop_sequences"] = []
            request_dict["generation_kwargs"] = {}

        # Determine request type string (oe-eval naming)
        if request.request_type == RequestType.LOGLIKELIHOOD:
            request_type_str = "loglikelihood"
        elif request.request_type == RequestType.COMPLETION:
            if request.continuations:
                request_type_str = "generate_until_and_loglikelihood"
            else:
                request_type_str = "generate_until"
        elif request.request_type == RequestType.CHAT:
            request_type_str = "generate_until"
        else:
            request_type_str = "unknown"

        # Determine label (ground truth)
        label = instance.metadata.get("gold_idx")
        if label is None and instance.gold_answer is not None:
            label = instance.gold_answer

        request_list.append(
            {
                "request_type": request_type_str,
                "doc": doc,
                "request": request_dict,
                "idx": 0,  # For multi-sample, this would vary
                "task_name": task_name,
                "doc_id": idx,
                "native_id": instance.metadata.get("id", f"doc_{idx}"),
                "label": label,
            }
        )

    return request_list


def run_task_impl(
    spec: str,
    backend: Backend,
    overrides: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    temperature: float | None = None,
    sampling_overrides: dict[str, Any] | None = None,
    requests_callback: Callable[[list[dict]], None] | None = None,
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
        requests_callback: Optional callback to receive requests early (before generation).
            Called with the list of request dicts immediately after they're built.
            Use this to write requests.jsonl before waiting for generation to complete.

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

        # Extract primary metric name from task config if specified
        primary_metric_name = None
        if task.config.primary_metric:
            primary_metric_name = task.config.primary_metric.name

        # Use existing_params (the actual params used for generation)
        return TaskResult(
            spec=spec,
            config={
                "name": task.config.name,
                "split": task.config.split.value,
                "num_fewshot": task.config.num_fewshot,
                "fewshot_seed": task.config.fewshot_seed,
                "limit": task.config.limit,
                "sampling_params": serialize_sampling_params(existing_params),
            },
            num_instances=len(instances),
            metrics=metrics,
            duration_seconds=duration,
            predictions=predictions,
            requests=request_objects,
            primary_metric=primary_metric_name,
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


def write_predictions_jsonl(
    output_dir: str,
    spec: str,
    predictions: list[dict],
    model_name: str | None = None,
) -> None:
    """Write per-instance predictions to JSONL.

    Args:
        output_dir: Base output directory
        spec: Task specification string (used for filename)
        predictions: List of prediction dicts to write
        model_name: Optional model name for multi-model runs (adds subdirectory)
    """
    if model_name:
        pred_dir = os.path.join(output_dir, "predictions", sanitize_spec_for_filename(model_name))
    else:
        pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    filename = sanitize_spec_for_filename(spec) + PREDICTIONS_SUFFIX
    filepath = os.path.join(pred_dir, filename)

    with open(filepath, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    logger.info(f"Predictions written to {filepath}")


def write_requests_jsonl(
    output_dir: str,
    spec: str,
    requests: list[dict],
    model_name: str | None = None,
) -> None:
    """Write per-instance requests to JSONL (oe-eval compatible format).

    This file shows exactly what the model saw during evaluation, useful for
    debugging and comparison with oe-eval outputs.

    Args:
        output_dir: Base output directory
        spec: Task specification string (used for filename)
        requests: List of request dicts to write
        model_name: Optional model name for multi-model runs (adds subdirectory)
    """
    if model_name:
        req_dir = os.path.join(output_dir, "requests", sanitize_spec_for_filename(model_name))
    else:
        req_dir = os.path.join(output_dir, "requests")
    os.makedirs(req_dir, exist_ok=True)

    filename = sanitize_spec_for_filename(spec) + REQUESTS_SUFFIX
    filepath = os.path.join(req_dir, filename)

    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Requests written to {filepath}")

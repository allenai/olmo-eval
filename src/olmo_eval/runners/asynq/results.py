"""Result processing and aggregation for async evaluation runner."""

from __future__ import annotations

import asyncio
import queue
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.progress import ProgressLogger
from olmo_eval.common.types import Response
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.runners.asynq.monitoring import check_workers_alive
from olmo_eval.runners.asynq.preparation import compute_task_metrics, finalize_task
from olmo_eval.runners.asynq.types import (
    WORKER_FATAL,
    ResultItem,
    TaskTracker,
)
from olmo_eval.runners.common.types import TaskResult
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.utils import compute_task_hash

if TYPE_CHECKING:
    import multiprocessing as mp

    from olmo_eval.common.execution import ScoringContext
    from olmo_eval.evals.tasks.common import Task

logger = get_logger(__name__)


def _format_scoring_error(exc: Exception, *, phase: str) -> dict[str, str]:
    """Build a JSON-serializable scoring error payload."""
    error = {
        "phase": phase,
        "type": type(exc).__qualname__,
    }
    message = str(exc).strip()
    if message:
        error["message"] = message
    return error


def _record_scoring_failure(
    response: Response,
    *,
    scorer_names: list[str],
    error: dict[str, str],
) -> Response:
    """Annotate a response when scoring aborts before per-output results exist."""
    for scorer_name in scorer_names:
        response.scores.setdefault(scorer_name, 0.0)

    for output in response.outputs:
        if output.metadata is None:
            output.metadata = {}
        existing = output.metadata.get("scoring_errors")
        if not isinstance(existing, dict):
            existing = {}
            output.metadata["scoring_errors"] = existing
        existing["__response__"] = error
        for scorer_name in scorer_names:
            output.metadata.setdefault(f"score:{scorer_name}", 0.0)

    return response


async def process_results(
    trackers: dict[str, TaskTracker],
    result_queue: mp.Queue,
    workers: list[mp.process.BaseProcess],
    scoring_context: ScoringContext,
    scoring_concurrency: int,
    total_tasks: int,
    total_instances: int,
    model_name: str,
    save_predictions: bool,
    write_predictions_fn: Any,
    save_requests: bool,
    write_requests_fn: Any,
) -> dict[str, TaskResult]:
    """Process results from workers with inline async scoring.

    As each instance result comes in, it's scored directly in this event loop
    using asyncio tasks, eliminating cross-process serialization.

    Args:
        trackers: Task trackers keyed by spec.
        result_queue: Queue receiving results from inference workers.
        workers: List of inference worker processes.
        scoring_context: Scoring context with execution environment.
        scoring_concurrency: Maximum concurrent scoring operations.
        total_tasks: Total number of tasks.
        total_instances: Total number of instances across all tasks.
        model_name: Model name for reporting.
        save_predictions: Whether to save predictions.
        write_predictions_fn: Function to write predictions.

    Returns:
        Dict mapping task spec to TaskResult.
    """
    results: dict[str, TaskResult] = {}
    tasks_complete = 0

    # Track scored responses per task: spec -> {idx: scored_response}
    scored_responses: dict[str, dict[int, Response]] = {spec: {} for spec in trackers}
    instances_scored: dict[str, int] = {spec: 0 for spec in trackers}

    # Progress tracking for scoring
    scoring_progress = ProgressLogger(
        total=total_instances, desc="Scored", logger=logger, color="cyan"
    )

    # Semaphore to limit concurrent scoring operations
    scoring_semaphore = asyncio.Semaphore(scoring_concurrency)
    in_flight_scoring: set[asyncio.Task[None]] = set()

    # Determine which tasks need the scoring worker (async scorers like sandboxed
    # code execution). All other tasks are scored inline to avoid mp.Queue overhead.
    tasks_needing_async_scoring: set[str] = set()
    for spec, tracker in trackers.items():
        if tracker.task is not None and tracker.task._has_async_scorers():
            tasks_needing_async_scoring.add(spec)
    if tasks_needing_async_scoring:
        logger.info(f"Tasks using async scoring worker: {tasks_needing_async_scoring}")

    def check_task_completion(spec: str) -> None:
        """Check if task is complete and finalize if so."""
        nonlocal tasks_complete
        tracker = trackers[spec]
        expected = tracker.total_instances - len(tracker.failed_instances)

        if instances_scored[spec] >= expected and spec not in results:
            responses_list = [
                scored_responses[spec][i] for i in sorted(scored_responses[spec].keys())
            ]
            assert tracker.task is not None
            task_result = compute_task_metrics(
                spec=spec,
                task=tracker.task,
                scored_responses=responses_list,
                failed_instances=tracker.failed_instances,
                total_instances=tracker.total_instances,
                duration_seconds=time.time() - tracker.start_time,
            )
            results[spec] = task_result
            tasks_complete += 1
            _report_task_completion(model_name, task_result)
            if save_predictions and task_result.predictions:
                task_hash = compute_task_hash(task_result.config)
                write_predictions_fn(
                    model_name, task_result.spec, task_result.predictions, task_hash
                )
            if save_requests and task_result.requests:
                task_hash = compute_task_hash(task_result.config)
                write_requests_fn(model_name, task_result.spec, task_result.requests, task_hash)

    async def score_and_store(
        spec: str,
        instance_idx: int,
        response: Response,
        task: Task,
    ) -> None:
        """Score a single response and store the result."""
        async with scoring_semaphore:
            try:
                scored_list = await task.score_responses([response], context=scoring_context)
                scored = scored_list[0] if scored_list else response
            except Exception as e:
                error = _format_scoring_error(e, phase="response")
                logger.warning(
                    "Failed to score %s[%s]: %s",
                    spec,
                    instance_idx,
                    error.get("message", error["type"]),
                )
                scored = _record_scoring_failure(
                    response,
                    scorer_names=list(task._get_scorers()),
                    error=error,
                )

        scored_responses[spec][instance_idx] = scored
        instances_scored[spec] += 1
        scoring_progress.update(1)
        check_task_completion(spec)

    def _reap_done_tasks() -> None:
        """Remove completed scoring tasks from in_flight set."""
        done = {t for t in in_flight_scoring if t.done()}
        for t in done:
            in_flight_scoring.discard(t)
            # Surface any unexpected exceptions
            if t.exception() is not None:
                logger.error(f"Scoring task failed unexpectedly: {t.exception()}")

    # Pre-add error tasks to results (these don't need scoring)
    for spec, tracker in trackers.items():
        if tracker.error:
            task_result = await finalize_task(tracker)
            results[spec] = task_result
            tasks_complete += 1
            _report_task_completion(model_name, task_result)

    pending_instances = total_instances
    last_health_check = time.time()
    health_check_interval = 5.0

    # Collect instance results and score each inline
    while pending_instances > 0:
        _reap_done_tasks()

        try:
            result_item: ResultItem = await asyncio.get_event_loop().run_in_executor(
                None, lambda: result_queue.get(timeout=1.0)
            )
        except queue.Empty:
            if time.time() - last_health_check > health_check_interval:
                check_workers_alive(workers, result_queue)
                last_health_check = time.time()
            continue

        # Check for fatal worker crash
        if result_item.task_id == WORKER_FATAL:
            logger.error(f"FATAL: Worker crashed! {result_item.error}")
            raise RuntimeError(f"Worker process crashed: {result_item.error}")

        tracker = trackers[result_item.task_id]
        pending_instances -= 1

        if tracker.error:
            continue

        # Check for hard failures (no outputs at all)
        # Soft failures (error with outputs, like MaxTurnsExceeded) should still be scored
        if result_item.error and not result_item.outputs:
            logger.warning(
                f"Instance {result_item.instance_idx} failed for {result_item.task_id}: "
                f"{result_item.error}"
            )
            tracker.add_failure(result_item.instance_idx, result_item.error)
            scoring_progress.update(1)
            check_task_completion(result_item.task_id)
            continue

        if result_item.error:
            # Soft error - log warning but continue to scoring
            logger.debug(
                f"Instance {result_item.instance_idx} completed with warning for "
                f"{result_item.task_id}: {result_item.error}"
            )

        # Build response
        trajectory = None
        if result_item.outputs:
            meta = result_item.outputs[0].metadata or {}
            traj_dict = meta.get("trajectory")
            if traj_dict:
                trajectory = AgentTrajectory.from_dict(traj_dict)

        assert result_item.instance is not None
        assert result_item.request is not None
        response = Response(
            instance=result_item.instance,
            request=result_item.request,
            outputs=result_item.outputs,
            trajectory=trajectory,
            request_trace=result_item.request_trace,
        )

        # Score inline as an async task
        assert tracker.task is not None
        scoring_task = asyncio.create_task(
            score_and_store(
                spec=result_item.task_id,
                instance_idx=result_item.instance_idx,
                response=response,
                task=tracker.task,
            )
        )
        in_flight_scoring.add(scoring_task)

    # Wait for all in-flight scoring to complete
    if in_flight_scoring:
        await asyncio.gather(*in_flight_scoring, return_exceptions=True)

    # Final check — all tasks should be complete
    if tasks_complete < total_tasks:
        # Some tasks may have had all instances fail during inference
        for spec in trackers:
            if spec not in results:
                check_task_completion(spec)

    scoring_progress.close()
    return results


def _report_task_completion(model_name: str, result: TaskResult) -> None:
    """Report when a task completes."""
    label = f"{model_name}:{result.spec}"
    if result.error:
        logger.error(f"\u2717 {label} (ERROR: {result.error})")
    else:
        logger.info(
            f"\u2713 {label} ({result.num_instances} instances, {result.duration_seconds:.1f}s)"
        )


def aggregate_results(
    results: dict[str, TaskResult],
    expanded_tasks: list[str],
    task_specs: list[str],
    provider_config: Any,
    attention_backend: str | None,
    harness_config: Any | None = None,
) -> dict[str, Any]:
    """Aggregate results and prepare final output.

    Args:
        results: Dict mapping task spec to TaskResult.
        expanded_tasks: List of all expanded task specs.
        task_specs: Original task specs (may include suite names).
        provider_config: Provider configuration.
        attention_backend: Attention backend name.
        harness_config: Optional harness configuration.

    Returns:
        Dict with full results including task metrics, errors, and summary.
    """
    model_config = provider_config.to_dict() if hasattr(provider_config, "to_dict") else {}
    if attention_backend:
        model_config["attention_backend"] = attention_backend
    results_dict: dict[str, Any] = {
        "model": provider_config.alias or provider_config.model,
        "model_path": provider_config.model,
        "provider": str(provider_config.kind),
        "model_config": model_config,
        "tasks": {},
        "summary": {},
        "errors": [],
        "timestamp": datetime.now().isoformat(),
    }
    if harness_config is not None:
        results_dict["harness_config"] = (
            harness_config.to_dict() if hasattr(harness_config, "to_dict") else harness_config
        )

    # Process each task result
    for spec, task_result in results.items():
        task_data: dict[str, Any] = {}

        if task_result.error:
            task_data["error"] = task_result.error
            results_dict["errors"].append({"task": spec, "error": task_result.error})
        else:
            task_data["metrics"] = task_result.metrics
            task_data["num_instances"] = task_result.num_instances
            task_data["duration_seconds"] = task_result.duration_seconds
            task_data["primary_metric"] = task_result.primary_metric
            if task_result.predictions:
                task_data["predictions"] = task_result.predictions

            # Get the primary metric for the summary
            if task_result.primary_metric and task_result.metrics:
                from olmo_eval.runners.processing.utils import get_primary_metric

                primary = get_primary_metric(task_result.metrics, task_result.primary_metric)
                if primary:
                    results_dict["summary"][spec] = {
                        "metric": primary[0],
                        "score": primary[1],
                    }

        task_data["config"] = task_result.config
        task_data["task_hash"] = (
            compute_task_hash(task_result.config) if task_result.config else None
        )
        results_dict["tasks"][spec] = task_data

    # Compute suite aggregations
    suite_aggs = compute_suite_aggregations(task_specs, results_dict["tasks"])
    if suite_aggs:
        results_dict["suites"] = suite_aggs

    return results_dict

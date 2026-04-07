"""Result processing and aggregation for async evaluation runner."""

from __future__ import annotations

import asyncio
import pickle
import queue
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from olmo_eval.common.logging import get_logger
from olmo_eval.common.types import Response
from olmo_eval.common.types.trajectory import AgentTrajectory
from olmo_eval.runners.asynq.monitoring import check_workers_alive
from olmo_eval.runners.asynq.preparation import compute_task_metrics, finalize_task
from olmo_eval.runners.asynq.types import (
    SCORER_FATAL,
    WORKER_FATAL,
    ResultItem,
    ScoredResponse,
    ScoringItem,
    TaskTracker,
)
from olmo_eval.runners.common.types import TaskResult
from olmo_eval.runners.processing.aggregation import compute_suite_aggregations
from olmo_eval.runners.processing.utils import compute_task_hash

if TYPE_CHECKING:
    import multiprocessing as mp

logger = get_logger(__name__)


async def process_results(
    trackers: dict[str, TaskTracker],
    result_queue: mp.Queue,
    scoring_queue: mp.Queue,
    scored_queue: mp.Queue,
    workers: list[mp.process.BaseProcess],
    scorer_proc: mp.process.BaseProcess,
    total_tasks: int,
    total_instances: int,
    model_name: str,
    save_predictions: bool,
    write_predictions_fn: Any,
) -> dict[str, TaskResult]:
    """Process results from workers with parallel instance-level scoring.

    As each instance result comes in, it's immediately sent to the scoring worker.
    Scored responses are collected and metrics are computed when all instances
    for a task are scored.

    Args:
        trackers: Task trackers keyed by spec.
        result_queue: Queue receiving results from inference workers.
        scoring_queue: Queue to send items for scoring.
        scored_queue: Queue receiving scored responses.
        workers: List of inference worker processes.
        scorer_proc: The scoring worker process.
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
    instances_sent: dict[str, int] = {spec: 0 for spec in trackers}
    instances_scored: dict[str, int] = {spec: 0 for spec in trackers}

    # Track which tasks have been sent to the scoring worker so we only
    # pickle the full Task object once per spec (the dominant serialization cost).
    tasks_sent_to_scorer: set[str] = set()

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

    def handle_scored_response(scored: ScoredResponse) -> None:
        """Process a scored response and check if task is complete."""
        assert scored.scored is not None, "scored.scored should not be None for valid responses"
        scored_responses[scored.spec][scored.instance_idx] = scored.scored
        instances_scored[scored.spec] += 1
        check_task_completion(scored.spec)

    def drain_scored_queue() -> None:
        """Non-blocking drain of scored responses."""
        while True:
            try:
                scored: ScoredResponse = scored_queue.get_nowait()
                # Check for fatal scorer error
                if scored.spec == SCORER_FATAL:
                    logger.error(f"FATAL: Scoring worker crashed! {scored.error}")
                    raise RuntimeError(f"Scoring worker crashed: {scored.error}")
                handle_scored_response(scored)
            except queue.Empty:
                break

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

    # Collect instance results and send each to scoring immediately
    while pending_instances > 0:
        # Check for scored responses (non-blocking)
        drain_scored_queue()

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
            check_task_completion(result_item.task_id)
            continue

        if result_item.error:
            # Soft error - log warning but continue to scoring
            logger.debug(
                f"Instance {result_item.instance_idx} completed with warning for "
                f"{result_item.task_id}: {result_item.error}"
            )

        # Build response and send to scoring immediately
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
        )

        # Send to scoring worker immediately.
        # Include the Task only on the first item per spec so the scoring
        # worker can cache it; subsequent items skip the expensive Task pickle.
        assert tracker.task is not None
        include_task = result_item.task_id not in tasks_sent_to_scorer
        scoring_item = ScoringItem(
            spec=result_item.task_id,
            instance_idx=result_item.instance_idx,
            response=response,
            task=tracker.task if include_task else None,
        )
        if include_task:
            # Validate pickling on the first item (which carries the Task).
            # queue.put uses a background thread that silently swallows
            # pickling errors, causing the job to stall.
            try:
                pickle.dumps(scoring_item)
            except (pickle.PicklingError, TypeError, AttributeError) as e:
                task_id = result_item.task_id
                idx = result_item.instance_idx
                logger.error(f"Failed to pickle scoring item for {task_id}[{idx}]: {e}")
                tracker.add_failure(result_item.instance_idx, f"Pickling failed: {e}")
                check_task_completion(result_item.task_id)
                continue
            tasks_sent_to_scorer.add(result_item.task_id)
        scoring_queue.put(scoring_item)
        instances_sent[result_item.task_id] += 1

    # Wait for remaining scoring to complete
    while tasks_complete < total_tasks:
        try:
            scored: ScoredResponse = await asyncio.get_event_loop().run_in_executor(
                None, lambda: scored_queue.get(timeout=1.0)
            )
            # Check for fatal scorer error
            if scored.spec == SCORER_FATAL:
                logger.error(f"FATAL: Scoring worker crashed! {scored.error}")
                raise RuntimeError(f"Scoring worker crashed: {scored.error}")
            handle_scored_response(scored)
        except queue.Empty:
            # Check if scorer died while we're still waiting for results
            if not scorer_proc.is_alive():
                total_sent = sum(instances_sent.values())
                total_scored = sum(instances_scored.values())
                missing = total_sent - total_scored
                raise RuntimeError(
                    f"Scoring worker exited while {missing} items still pending. "
                    f"Sent {total_sent}, scored {total_scored}. "
                    f"Items may have been lost due to queue errors (check logs)."
                ) from None
            continue

    logger.info(f"Scoring complete ({total_tasks} tasks, {total_instances} instances)")
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
        results: Task results keyed by spec.
        expanded_tasks: List of expanded task specs.
        task_specs: Original task specs (may include suites).
        provider_config: Provider configuration.
        attention_backend: Attention backend used.
        harness_config: Optional HarnessConfig for full config serialization.

    Returns:
        Results dictionary ready for serialization.
    """
    from olmo_eval.runners.io.formatting import get_model_display_name

    errors = [(spec, r) for spec, r in results.items() if r.error]
    if errors:
        logger.error(f"{len(errors)} tasks failed")
        for spec, error_result in errors:
            logger.error(f"  {spec}: {error_result.error}")

    provider_str = str(provider_config.kind)
    display_model_name = get_model_display_name(provider_config.model, provider_config.alias)

    results_dict: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": display_model_name,
        "model_path": provider_config.model,
        "provider": provider_str,
        "tasks": {},
        "errors": [],
    }

    for spec in expanded_tasks:
        if spec in results:
            task_result = results[spec]
            if task_result.error:
                results_dict["errors"].append({"spec": spec, "error": task_result.error})
            else:
                task_data = task_result.to_dict(include_predictions=True)
                task_hash = compute_task_hash(task_result.config)
                if task_hash:
                    task_data["task_hash"] = task_hash
                results_dict["tasks"][spec] = task_data

    model_config_dict = provider_config.to_dict()
    model_config_dict["attention_backend"] = attention_backend
    results_dict["model_config"] = model_config_dict

    if harness_config is not None:
        results_dict["harness_config"] = harness_config.to_dict()

    suite_aggs = compute_suite_aggregations(task_specs, results_dict["tasks"])
    if suite_aggs:
        results_dict["suites"] = suite_aggs

    return results_dict


__all__ = [
    "process_results",
    "aggregate_results",
]

"""Inspect AI log parsing for ASTA-bench results."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_inspect_log(log_content: dict[str, Any]) -> dict[str, Any]:
    """Parse an Inspect AI evaluation log.

    Args:
        log_content: Parsed JSON from an Inspect .eval or .json log file.

    Returns:
        Dictionary with:
            - metrics: Aggregated metrics from all scorers
            - predictions: Per-sample predictions with instance metrics
            - metadata: Additional evaluation metadata
    """
    results = log_content.get("results", {})
    samples = log_content.get("samples", [])
    eval_info = log_content.get("eval", {})

    # Extract aggregate metrics from scorers
    metrics: dict[str, float] = {}
    scores_data = results.get("scores", [])

    # Handle both list and dict formats for scores
    if isinstance(scores_data, list):
        for scorer in scores_data:
            scorer_name = scorer.get("name", "unknown")
            scorer_metrics = scorer.get("metrics", {})
            for metric_name, metric_data in scorer_metrics.items():
                value = metric_data.get("value") if isinstance(metric_data, dict) else metric_data
                if value is not None:
                    metrics[f"{scorer_name}_{metric_name}"] = float(value)
    elif isinstance(scores_data, dict):
        for scorer_name, scorer_data in scores_data.items():
            scorer_metrics = scorer_data.get("metrics", {})
            for metric_name, metric_data in scorer_metrics.items():
                value = metric_data.get("value") if isinstance(metric_data, dict) else metric_data
                if value is not None:
                    metrics[f"{scorer_name}_{metric_name}"] = float(value)

    # Extract per-sample predictions
    predictions: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample.get("id", "")
        sample_scores = sample.get("scores", {})

        # Build instance metrics from sample scores
        instance_metrics: dict[str, dict[str, float]] = {}
        if isinstance(sample_scores, dict):
            for scorer_name, score_data in sample_scores.items():
                if isinstance(score_data, dict):
                    value = score_data.get("value")
                    if value is not None:
                        instance_metrics[scorer_name] = {"external": float(value)}

        prediction: dict[str, Any] = {
            "native_id": sample_id,
            "instance_metrics": instance_metrics,
        }

        # Include input/target if available (for debugging)
        if "input" in sample:
            prediction["input"] = _extract_input_text(sample["input"])
        if "target" in sample:
            prediction["target"] = sample["target"]

        # Include error info if present
        if sample.get("error"):
            prediction["error"] = sample["error"]

        # Include model usage if available
        if "model_usage" in sample:
            prediction["model_usage"] = sample["model_usage"]

        predictions.append(prediction)

    # Build metadata
    metadata: dict[str, Any] = {
        "task": eval_info.get("task", ""),
        "model": eval_info.get("model", ""),
        "solver": eval_info.get("solver", ""),
        "created": eval_info.get("created", ""),
        "completed": results.get("completed", ""),
        "total_samples": results.get("total_samples", len(samples)),
        "completed_samples": results.get("completed_samples", len(samples)),
    }

    # Include error rate if available
    if "error" in results:
        metadata["error_info"] = results["error"]

    return {
        "metrics": metrics,
        "predictions": predictions,
        "metadata": metadata,
    }


def _extract_input_text(input_data: Any) -> str:
    """Extract text representation from Inspect sample input."""
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, list):
        # Messages format
        texts = []
        for msg in input_data:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    # Multi-part content (e.g., images + text)
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
        return "\n".join(texts)
    return str(input_data)


def load_inspect_logs(log_dir: str, executor: Any = None) -> list[dict[str, Any]]:
    """Load all Inspect AI log files from a directory.

    Can work locally or via sandbox executor.

    Args:
        log_dir: Directory containing Inspect log files.
        executor: Optional sandbox executor for remote file access.

    Returns:
        List of parsed log contents.
    """
    logs: list[dict[str, Any]] = []

    if executor is None:
        # Local file access
        log_path = Path(log_dir)
        if not log_path.exists():
            logger.warning(f"Log directory does not exist: {log_dir}")
            return logs

        # Inspect saves logs with .eval extension (gzipped JSON) or .json
        for pattern in ("*.json", "*.eval"):
            for log_file in log_path.glob(pattern):
                try:
                    content = _load_log_file(log_file)
                    if content:
                        logs.append(content)
                except Exception as e:
                    logger.warning(f"Failed to parse {log_file}: {e}")

    return logs


def _load_log_file(log_file: Path) -> dict[str, Any] | None:
    """Load a single Inspect log file."""
    if log_file.suffix == ".eval":
        # .eval files are gzipped JSON
        import gzip

        with gzip.open(log_file, "rt", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(log_file, encoding="utf-8") as f:
            return json.load(f)


def aggregate_metrics(parsed_logs: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate metrics across multiple Inspect logs.

    For metrics that appear in multiple logs, computes the mean.
    """
    from collections import defaultdict

    metric_values: dict[str, list[float]] = defaultdict(list)

    for log_data in parsed_logs:
        for metric_name, value in log_data.get("metrics", {}).items():
            metric_values[metric_name].append(value)

    # Compute means
    aggregated: dict[str, float] = {}
    for metric_name, values in metric_values.items():
        aggregated[metric_name] = sum(values) / len(values) if values else 0.0

    return aggregated


def parse_agenteval_json(content: dict[str, Any]) -> dict[str, Any]:
    """Parse scores.json produced by `astabench score`.

    The scores.json format has a `results` array where each entry has:
        - task_name: Name of the task
        - eval_spec: Evaluation configuration
        - metrics: Array of {name, value} objects
        - model_usages: Token usage per sample
        - model_costs: Cost data (often null)

    Args:
        content: Parsed JSON from scores.json.

    Returns:
        Dictionary with:
            - metrics: Aggregated scores from the scoring run
            - costs: Token usage costs per model
            - metadata: Evaluation metadata
    """
    metrics: dict[str, float] = {}
    costs: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    results = content.get("results", [])

    for task_result in results:
        task_name = task_result.get("task_name", "unknown")

        # Parse metrics array: [{"name": "metric/scorer", "value": 0.5}, ...]
        task_metrics = task_result.get("metrics", [])
        for metric_entry in task_metrics:
            metric_name = metric_entry.get("name", "")
            metric_value = metric_entry.get("value")
            if metric_name and metric_value is not None:
                # Store as task_name/metric_name for uniqueness
                full_name = f"{task_name}/{metric_name}"
                metrics[full_name] = float(metric_value)

        # Extract eval_spec metadata
        eval_spec = task_result.get("eval_spec", {})
        if eval_spec and not metadata.get("model"):
            metadata["model"] = eval_spec.get("model", "")
            metadata["solver"] = eval_spec.get("solver", "")
            metadata["revision"] = eval_spec.get("revision", {})

        # Aggregate token usage across all samples for this task
        model_usages = task_result.get("model_usages", [])
        for sample_usages in model_usages:
            for usage_entry in sample_usages:
                model_name = usage_entry.get("model", "unknown")
                usage = usage_entry.get("usage", {})
                if model_name not in costs:
                    costs[model_name] = {
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    }
                costs[model_name]["input_tokens"] += usage.get("input_tokens", 0) or 0
                costs[model_name]["output_tokens"] += usage.get("output_tokens", 0) or 0
                costs[model_name]["total_tokens"] += usage.get("total_tokens", 0) or 0

    metadata["num_tasks"] = len(results)

    return {
        "metrics": metrics,
        "costs": costs,
        "metadata": metadata,
    }


def parse_summary_stats_json(content: dict[str, Any]) -> dict[str, Any]:
    """Parse summary_stats.json produced by `astabench score`.

    The summary_stats.json format has a `stats` object with:
        - overall: {score, score_stderr, cost, cost_stderr}
        - tag/NAME: Tag-level aggregates
        - task/NAME: Task-level scores

    Args:
        content: Parsed JSON from summary_stats.json.

    Returns:
        Dictionary with:
            - metrics: All scores as flat metrics dict
            - overall_score: The primary overall score
            - tag_scores: Dict of tag name -> score
            - task_scores: Dict of task name -> score
    """
    metrics: dict[str, float] = {}
    tag_scores: dict[str, float] = {}
    task_scores: dict[str, float] = {}
    overall_score: float | None = None

    stats = content.get("stats", {})

    for key, data in stats.items():
        if not isinstance(data, dict):
            continue

        score = data.get("score")
        stderr = data.get("score_stderr")
        cost = data.get("cost")

        if key == "overall":
            if score is not None:
                overall_score = float(score)
                metrics["overall_score"] = overall_score
            if stderr is not None:
                metrics["overall_stderr"] = float(stderr)
            if cost is not None:
                metrics["overall_cost"] = float(cost)

        elif key.startswith("tag/"):
            tag_name = key[4:]  # Remove "tag/" prefix
            if score is not None:
                tag_scores[tag_name] = float(score)
                metrics[f"tag_{tag_name}_score"] = float(score)
            if stderr is not None:
                metrics[f"tag_{tag_name}_stderr"] = float(stderr)

        elif key.startswith("task/"):
            task_name = key[5:]  # Remove "task/" prefix
            if score is not None:
                task_scores[task_name] = float(score)
                metrics[f"{task_name}_score"] = float(score)
            if stderr is not None:
                metrics[f"{task_name}_stderr"] = float(stderr)

    return {
        "metrics": metrics,
        "overall_score": overall_score,
        "tag_scores": tag_scores,
        "task_scores": task_scores,
    }

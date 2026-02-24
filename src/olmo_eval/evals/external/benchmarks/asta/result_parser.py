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

    # Add count metrics
    aggregated["num_tasks"] = float(len(parsed_logs))
    total_samples = sum(log.get("metadata", {}).get("total_samples", 0) for log in parsed_logs)
    aggregated["total_samples"] = float(total_samples)

    return aggregated


def parse_agenteval_json(content: dict[str, Any]) -> dict[str, Any]:
    """Parse scores.json produced by `astabench score`.

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

    # Extract task-level scores
    tasks = content.get("tasks", {})
    for task_name, task_data in tasks.items():
        if isinstance(task_data, dict):
            # Extract primary metric score
            score = task_data.get("score")
            if score is not None:
                metrics[f"{task_name}_score"] = float(score)

            # Extract stderr if available
            stderr = task_data.get("stderr")
            if stderr is not None:
                metrics[f"{task_name}_stderr"] = float(stderr)

            # Extract cost if available
            cost = task_data.get("cost")
            if cost is not None:
                metrics[f"{task_name}_cost"] = float(cost)

    # Extract tag-level aggregates (category scores)
    tags = content.get("tags", {})
    for tag_name, tag_data in tags.items():
        if isinstance(tag_data, dict):
            score = tag_data.get("score")
            if score is not None:
                metrics[f"tag_{tag_name}_score"] = float(score)

            stderr = tag_data.get("stderr")
            if stderr is not None:
                metrics[f"tag_{tag_name}_stderr"] = float(stderr)

            cost = tag_data.get("cost")
            if cost is not None:
                metrics[f"tag_{tag_name}_cost"] = float(cost)

    # Extract overall score if available
    overall = content.get("overall", {})
    if isinstance(overall, dict):
        if "score" in overall:
            metrics["overall_score"] = float(overall["score"])
        if "cost" in overall:
            metrics["overall_cost"] = float(overall["cost"])
    elif isinstance(overall, (int, float)):
        metrics["overall_score"] = float(overall)

    # Extract cost breakdown by model
    model_costs = content.get("costs", {})
    if isinstance(model_costs, dict):
        costs = model_costs
        # Also add total cost as a metric
        total_cost = sum(
            c.get("total", 0) if isinstance(c, dict) else 0 for c in model_costs.values()
        )
        if total_cost > 0:
            metrics["total_model_cost"] = float(total_cost)

    # Extract metadata
    metadata = {
        "eval_config": content.get("eval_config", {}),
        "split": content.get("split", ""),
        "model": content.get("model", ""),
        "created": content.get("created", ""),
    }

    return {
        "metrics": metrics,
        "costs": costs,
        "metadata": metadata,
    }

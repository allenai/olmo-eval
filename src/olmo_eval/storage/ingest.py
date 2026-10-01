"""Rebuild saveable results from an evaluation output directory."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from olmo_eval.common.types import compute_model_hash
from olmo_eval.storage.artifacts import build_predictions_uri

__all__ = ["OutputDirResults", "load_output_dir"]

ReadText = Callable[[str], str | None]


@dataclass
class OutputDirResults:
    """Results read back from an output directory, ready for ``save_results``."""

    results: dict[str, Any]
    experiment_id: str
    model_hash: str
    experiment_name: str | None = None
    experiment_group: str | None = None
    experiment_duration_seconds: float | None = None
    provider_init_seconds: dict[str, float] | None = None
    missing_predictions: list[str] = field(default_factory=list)


def load_output_dir(
    base: str,
    read_text: ReadText,
    model_name: str | None = None,
    include_predictions: bool = True,
) -> OutputDirResults:
    """Read a single-model run's metrics.json and predictions from ``base``.

    The experiment ID and model hash are taken from metrics.json, so loading a
    run that was already saved identifies the same stored experiment.

    Args:
        base: Output directory path or S3 prefix the run was written to.
        read_text: Returns the text at a path under ``base``, or None if absent.
        model_name: Name to store the model under. Defaults to the name recorded
            in metrics.json, then to the model path.
        include_predictions: Whether to read per-task prediction files.

    Returns:
        The rebuilt results and the identifiers to save them under.

    Raises:
        ValueError: If metrics.json is missing or cannot identify the run.
    """
    base = base.rstrip("/")
    text = read_text(f"{base}/metrics.json")
    if text is None:
        raise ValueError(f"No metrics.json found under {base}")
    metrics = json.loads(text)

    experiment_id = metrics.get("experiment_id")
    if not experiment_id:
        raise ValueError("metrics.json has no experiment_id")
    if not metrics.get("timestamp"):
        raise ValueError("metrics.json has no timestamp")

    config = dict(metrics.get("config") or {})
    if "models" in config:
        raise ValueError("Multi-model metrics.json files are not supported")
    recorded_hash = config.pop("model_hash", None)
    # Harness runs nest the provider config; older and external runs store it flat.
    provider = config.get("provider")
    model_config = dict(provider) if isinstance(provider, dict) else config

    model_hash = recorded_hash or compute_model_hash(model_config) or "unknown"
    name = model_name or metrics.get("model_name") or model_config.get("model")
    if not name:
        raise ValueError("Cannot determine the model name; pass one explicitly")

    task_errors = {
        entry["task"]: entry.get("error")
        for entry in metrics.get("errors", [])
        if isinstance(entry, dict) and "task" in entry
    }

    tasks: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for entry in metrics.get("tasks", []):
        spec = entry["task"]
        task_hash = entry.get("task_hash")
        task_data: dict[str, Any] = {
            "metrics": entry.get("metrics", {}),
            "task_hash": task_hash,
            "config": entry.get("config"),
            "num_instances": entry.get("num_instances"),
            "primary_metric": entry.get("primary_metric"),
            "duration_seconds": entry.get("duration_seconds"),
            "instances_processed": entry.get("instances_processed"),
            "instances_failed": entry.get("instances_failed"),
            "error_summary": entry.get("error_summary"),
            "error": task_errors.get(spec),
        }
        # The runner keeps no predictions for a task that failed.
        if include_predictions and task_hash and not task_data["error"]:
            predictions_text = read_text(build_predictions_uri(base, name, spec, task_hash))
            if predictions_text is None:
                missing.append(spec)
            else:
                task_data["predictions"] = [
                    json.loads(line) for line in predictions_text.splitlines() if line.strip()
                ]
        tasks[spec] = task_data

    results = {
        "model": name,
        "model_path": model_config.get("model"),
        "provider": str(model_config.get("kind") or model_config.get("provider") or "unknown"),
        "timestamp": metrics["timestamp"],
        "model_config": model_config,
        "tasks": tasks,
    }
    return OutputDirResults(
        results=results,
        experiment_id=experiment_id,
        model_hash=model_hash,
        experiment_name=metrics.get("experiment_name"),
        experiment_group=metrics.get("experiment_group"),
        experiment_duration_seconds=metrics.get("experiment_duration_seconds"),
        provider_init_seconds=metrics.get("provider_init_seconds"),
        missing_predictions=missing,
    )

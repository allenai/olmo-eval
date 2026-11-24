"""Functions for writing predictions and requests to JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from olmo_eval.common.images import ImageReference
from olmo_eval.common.logging import get_logger
from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
from olmo_eval.runners.processing.utils import sanitize_spec_for_filename

logger = get_logger("runners.writers")


def _prediction_json_default(obj: object) -> Any:
    """Serialize typed image references while keeping other unexpected types strict."""
    if isinstance(obj, ImageReference):
        return obj.to_json()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _request_json_default(obj: object) -> Any:
    """Serialize lazy image references (and legacy callables) in request records.

    Request records spread ``instance.metadata`` into the document. Typed
    :class:`~olmo_eval.common.images.ImageReference` values emit their structured
    JSON form; legacy callable loaders fall back to a type placeholder. Any other
    value JSON cannot encode still raises.
    """
    if isinstance(obj, ImageReference):
        return obj.to_json()
    if callable(obj):
        return f"<{type(obj).__name__}>"
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def output_path(
    output_dir: str | Path,
    kind: str,
    spec: str,
    model_name: str,
    task_hash: str | None = None,
) -> Path:
    """Path of a task's ``predictions`` or ``requests`` file, as the writers below name it."""
    suffix = {"predictions": PREDICTIONS_SUFFIX, "requests": REQUESTS_SUFFIX}[kind]
    name = sanitize_spec_for_filename(spec)
    if task_hash:
        name = f"{name}_{task_hash[-6:]}"
    return Path(output_dir) / kind / sanitize_spec_for_filename(model_name) / f"{name}{suffix}"


def write_predictions_jsonl(
    output_dir: str,
    spec: str,
    predictions: list[dict],
    model_name: str,
    task_hash: str | None = None,
) -> None:
    """Write per-instance predictions to JSONL.

    Args:
        output_dir: Base output directory
        spec: Task specification string (used for filename)
        predictions: List of prediction dicts to write
        model_name: Model name or alias (used for subdirectory)
        task_hash: Optional task config hash (last 6 chars added to filename)
    """
    filepath = output_path(output_dir, "predictions", spec, model_name, task_hash)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred, default=_prediction_json_default) + "\n")

    logger.info(f"Saved {len(predictions)} predictions: {spec}")


def write_requests_jsonl(
    output_dir: str,
    spec: str,
    requests: list[dict],
    model_name: str,
    task_hash: str | None = None,
) -> None:
    """Write per-instance requests to JSONL (oe-eval compatible format).

    This file shows exactly what the model saw during evaluation, useful for
    debugging and comparison with oe-eval outputs.

    Args:
        output_dir: Base output directory
        spec: Task specification string (used for filename)
        requests: List of request dicts to write
        model_name: Model name or alias (used for subdirectory)
        task_hash: Optional task config hash (last 6 chars added to filename)
    """
    filepath = output_path(output_dir, "requests", spec, model_name, task_hash)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req, default=_request_json_default) + "\n")

    logger.info(f"Saved {len(requests)} requests: {spec}")

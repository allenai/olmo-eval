"""Functions for writing predictions and requests to JSONL files."""

from __future__ import annotations

import json
import os

from olmo_eval.common.logging import get_logger
from olmo_eval.runners.common.types import PREDICTIONS_SUFFIX, REQUESTS_SUFFIX
from olmo_eval.runners.processing.utils import sanitize_spec_for_filename

logger = get_logger("runners.writers")


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
    pred_dir = os.path.join(output_dir, "predictions", sanitize_spec_for_filename(model_name))
    os.makedirs(pred_dir, exist_ok=True)

    # Build filename with optional hash suffix
    base_name = sanitize_spec_for_filename(spec)
    if task_hash:
        base_name = f"{base_name}_{task_hash[-6:]}"
    filename = base_name + PREDICTIONS_SUFFIX
    filepath = os.path.join(pred_dir, filename)

    with open(filepath, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

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
    req_dir = os.path.join(output_dir, "requests", sanitize_spec_for_filename(model_name))
    os.makedirs(req_dir, exist_ok=True)

    # Build filename with optional hash suffix
    base_name = sanitize_spec_for_filename(spec)
    if task_hash:
        base_name = f"{base_name}_{task_hash[-6:]}"
    filename = base_name + REQUESTS_SUFFIX
    filepath = os.path.join(req_dir, filename)

    with open(filepath, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Saved {len(requests)} requests: {spec}")

"""Rescore saved text correctness predictions without loading data or inference providers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import Task
from olmo_eval.evals.tasks.common.final_answer import (
    score_with_answer_region,
    validate_final_answer_only,
)
from olmo_eval.evals.tasks.gpqa import GPQATask
from olmo_eval.runners.io.builders import build_predictions
from olmo_eval.runners.processing.utils import compute_task_hash

_METADATA_FIELDS = (
    "finish_reason",
    "completion_tokens",
    "prompt_tokens",
    "sum_logits",
    "num_tokens",
    "num_tokens_all",
    "is_greedy",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read local saved prediction or request rows."""
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_identity(path: Path) -> dict[str, str]:
    """Record immutable content identity alongside the source path."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest}


async def rescore_predictions(
    task: Task,
    predictions: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    text_is_original: bool = False,
) -> dict[str, Any]:
    """Rescore original generations against saved labels, without reconstructing the dataset.

    ``text_is_original`` explicitly asserts that any output lacking ``original_text``
    has unmodified ``text``. Never infer that assertion from the absence of a delimiter.
    Only the synchronous text correctness combinations audited for the gate are supported.
    """
    validate_final_answer_only(task, force=True)
    by_id = {}
    for request in requests:
        key = json.dumps(request["native_id"], sort_keys=True)
        if key in by_id:
            raise ValueError(f"Duplicate request native_id: {key}")
        by_id[key] = request
    seen = set()
    responses = []
    for prediction in predictions:
        key = json.dumps(prediction["native_id"], sort_keys=True)
        if key in seen:
            raise ValueError(f"Duplicate prediction native_id: {key}")
        seen.add(key)
        if key not in by_id:
            raise ValueError(f"No saved request for native_id: {key}")
        saved = by_id[key]
        if saved.get("task_name") != task.config.name:
            raise ValueError(
                f"Saved task {saved.get('task_name')!r} does not match {task.config.name!r}"
            )
        if (
            saved.get("request_type") != "generate_until"
            or prediction.get("trajectory")
            or saved["request"].get("tools")
            or any("turns" in o or o.get("tool_calls") for o in prediction.get("model_output", []))
        ):
            raise ValueError("Offline rescoring supports only non-agent text generation requests")
        label = prediction.get("label")
        if label != saved.get("label") or label is None:
            raise ValueError(f"Missing or inconsistent saved labels for {key}")
        doc = copy.deepcopy(saved["doc"])
        choices = doc.get("choices")
        if isinstance(prediction.get("gold_answer"), str):
            gold = prediction["gold_answer"]
        elif (
            isinstance(task, GPQATask)
            and isinstance(label, int)
            and not isinstance(label, bool)
            and choices
            and 0 <= label < len(choices)
        ):
            gold = chr(ord("A") + label)
        elif isinstance(label, str):
            gold = label
        else:
            raise ValueError(
                f"Ambiguous saved label for {key}: {label!r}; explicit gold_answer required"
            )
        doc["id"] = prediction["native_id"]
        instance = Instance(
            question=doc.get("query", ""),
            gold_answer=gold,
            choices=tuple(choices) if choices else None,
            metadata=doc,
        )
        context = saved["request"].get("context", "")
        request = (
            LMRequest(request_type=RequestType.CHAT, messages=tuple(context))
            if isinstance(context, list)
            else LMRequest(request_type=RequestType.COMPLETION, prompt=context)
        )
        outputs = []
        if not prediction.get("model_output"):
            raise ValueError(
                f"No saved outputs for {key}; cannot treat a missing record as an answer"
            )
        for index, output in enumerate(prediction["model_output"]):
            raw = output.get("original_text")
            if raw is None and text_is_original:
                raw = output.get("text")
            if not isinstance(raw, str):
                raise ValueError(
                    f"Original text unavailable for {key} output {index}; supply original_text, "
                    "or use --text-is-original only if saved text is known to be unmodified"
                )
            metadata = {k: output[k] for k in _METADATA_FIELDS if k in output}
            metadata["original_text"] = raw
            outputs.append(LMOutput(text=raw, metadata=metadata))
        responses.append(Response(instance=instance, request=request, outputs=outputs))
    if not responses:
        raise ValueError("No saved predictions to rescore")
    scored = await score_with_answer_region(task, responses)
    rows = build_predictions(scored, metrics=task.config.metrics)
    for row, source in zip(rows, predictions, strict=True):
        row["doc_id"] = source["doc_id"]
    config = task.config.to_dict()
    return {
        "config": config,
        "task_hash": compute_task_hash(config),
        "num_instances": len(scored),
        "metrics": task.compute_metrics(scored),
        "predictions": rows,
        "text_is_original": text_is_original,
    }

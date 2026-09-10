#!/usr/bin/env python3
"""Build a deterministic Qwen3.5 router-profile prompt manifest.

The source is one validated default-K=8 olmo-eval result.  We retain 20
prompts from each of MATH-500, GPQA Diamond, and IFEval OOD, balanced between
that run's correct and incorrect examples when both strata have enough rows.
The saved model output is metadata only: the profiler generates a fresh full
thinking trajectory so reasoning tokens are not lost to answer parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TASKS = {
    "math500": ("*math500*-predictions.jsonl", "*math500*-requests.jsonl"),
    "gpqa_diamond": ("*gpqa_diamond*-predictions.jsonl", "*gpqa_diamond*-requests.jsonl"),
    "ifeval_ood": ("*ifeval_ood*-predictions.jsonl", "*ifeval_ood*-requests.jsonl"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def unique_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.rglob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern!r} below {root}, found {matches}")
    return matches[0]


def identity(row: dict[str, Any]) -> str:
    return str(row.get("native_id", row["doc_id"]))


def accuracy(row: dict[str, Any]) -> int:
    metrics = row.get("instance_metrics", {})
    for name in ("accuracy", "prompt_level_strict_acc", "minerva_math_flex", "multiple_choice", "ifeval"):
        values = metrics.get(name)
        if isinstance(values, dict) and values:
            return int(round(float(next(iter(values.values())))))
    raise ValueError(f"no supported accuracy metric for {identity(row)}: {metrics}")


def order_key(seed: int, task: str, row: dict[str, Any]) -> str:
    value = f"{seed}:{task}:{identity(row)}".encode()
    return hashlib.sha256(value).hexdigest()


def example_seed(seed: int, task: str, row: dict[str, Any]) -> int:
    digest = hashlib.sha256(f"generation:{seed}:{task}:{identity(row)}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def select(rows: list[dict[str, Any]], *, task: str, seed: int, per_task: int) -> list[dict[str, Any]]:
    target_correct = per_task // 2
    target_incorrect = per_task - target_correct
    strata = {
        score: sorted(
            (row for row in rows if row["baseline_score"] == score),
            key=lambda row: order_key(seed, task, row),
        )
        for score in (0, 1)
    }
    chosen = strata[1][:target_correct] + strata[0][:target_incorrect]
    if len(chosen) < per_task:
        used = {identity(row) for row in chosen}
        remainder = sorted(
            (row for row in rows if identity(row) not in used),
            key=lambda row: order_key(seed, task, row),
        )
        chosen.extend(remainder[: per_task - len(chosen)])
    if len(chosen) != per_task:
        raise ValueError(f"{task} has only {len(chosen)} selectable rows")
    return sorted(chosen, key=lambda row: order_key(seed, task, row))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    selected: list[dict[str, Any]] = []
    report: dict[str, dict[str, int]] = {}
    for task, (prediction_pattern, request_pattern) in TASKS.items():
        predictions = read_jsonl(unique_match(args.results, prediction_pattern))
        requests = {
            identity(row): row for row in read_jsonl(unique_match(args.results, request_pattern))
        }
        candidates = []
        for prediction in predictions:
            key = identity(prediction)
            request = requests[key]
            outputs = prediction.get("model_output") or []
            output = outputs[0] if outputs else {}
            candidates.append(
                {
                    "task": task,
                    "doc_id": prediction["doc_id"],
                    "native_id": prediction.get("native_id"),
                    "messages": request["request"]["context"],
                    "doc": request.get("doc"),
                    "label": prediction.get("label"),
                    "baseline_score": accuracy(prediction),
                    "baseline_final_output": output.get("text", ""),
                    "baseline_num_tokens": output.get("num_tokens"),
                }
            )
        task_rows = select(candidates, task=task, seed=args.seed, per_task=args.per_task)
        for row in task_rows:
            row["seed"] = example_seed(args.seed, task, row)
        selected.extend(task_rows)
        report[task] = {
            "selected": len(task_rows),
            "correct": sum(row["baseline_score"] for row in task_rows),
            "incorrect": sum(1 - row["baseline_score"] for row in task_rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "total": len(selected), "tasks": report}, indent=2))


if __name__ == "__main__":
    main()

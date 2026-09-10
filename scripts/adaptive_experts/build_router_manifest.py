#!/usr/bin/env python3
"""Build a deterministic, correctness-stratified router-profile manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

TASKS = {
    "gpqa_diamond": (
        "gpqa_diamond_qwen3_thinking_144c3d",
        ("multiple_choice", "multiple_choice"),
    ),
    "ifeval_ood": ("ifeval_ood_qwen3_thinking_b458d6", ("ifeval", "ifeval")),
    "math500": ("math500_chat_beca1b", ("minerva_math_flex", "minerva_math_flex")),
}
MODEL_DIR = "Qwen_Qwen3-30B-A3B"


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def example_seed(seed: int, task: str, doc_id: int | str) -> int:
    payload = f"{seed}:{task}:{doc_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def build_manifest(source: Path, seed: int, per_stratum: int) -> list[dict]:
    rows: list[dict] = []
    for task, (stem, metric_path) in TASKS.items():
        requests = read_jsonl(source / "requests" / MODEL_DIR / f"{stem}-requests.jsonl")
        predictions = read_jsonl(source / "predictions" / MODEL_DIR / f"{stem}-predictions.jsonl")
        request_by_id = {row["doc_id"]: row for row in requests}
        candidates = {0: [], 1: []}
        for prediction in predictions:
            score = prediction["instance_metrics"][metric_path[0]][metric_path[1]]
            if score not in (0, 1):
                continue
            request = request_by_id[prediction["doc_id"]]
            candidates[int(score)].append((request, prediction))

        for score in (0, 1):
            rng = random.Random(f"{seed}:{task}:{score}")
            rng.shuffle(candidates[score])
            if len(candidates[score]) < per_stratum:
                raise ValueError(
                    f"{task} only has {len(candidates[score])} examples with score={score}"
                )
            for request, prediction in candidates[score][:per_stratum]:
                doc_id = request["doc_id"]
                rows.append(
                    {
                        "task": task,
                        "doc_id": doc_id,
                        "native_id": request.get("native_id"),
                        "messages": request["request"]["context"],
                        "stop_sequences": request["request"].get("stop_sequences", []),
                        "doc": request.get("doc"),
                        "label": request.get("label"),
                        "baseline_score": score,
                        "baseline_final_output": prediction.get("final_output"),
                        "baseline_num_tokens": prediction.get("model_output", [{}])[0].get(
                            "num_tokens"
                        ),
                        "seed": example_seed(seed, task, doc_id),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--per-stratum", type=int, default=10)
    args = parser.parse_args()

    rows = build_manifest(args.source, args.seed, args.per_stratum)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()

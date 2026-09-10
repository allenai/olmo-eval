#!/usr/bin/env python3
"""Summarize collected Qwen reference-preserving adaptive-threshold evaluations."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "adaptive_experts"
LEDGER = ROOT / "notes" / "beaker_jobs.jsonl"
GROUP = "adaptive-experts-qwen-adaptive-reference-eager-thresholds-20260722"
NORMALIZED_GROUP = "adaptive-experts-qwen-adaptive-mass-expanded-20260717"
THRESHOLD_RE = re.compile(r"t0p(?P<digits>\d+)-r(?P<replicate>\d+)")

TASK_FILES = {
    "math500": "math500_chat_*-predictions.jsonl",
    "gpqa": "gpqa_diamond_qwen3_thinking_*-predictions.jsonl",
    "ifbench_base": "ifeval_ood_[0-9a-f]*-predictions.jsonl",
    "ifbench_mt": "ifeval_mt_wildchat_unused_withRewrite_*-predictions.jsonl",
    "ifbench_mt_ood": "ifeval_mt_ood_wildchat_unused_withRewrite_*-predictions.jsonl",
    "humaneval": "humaneval_chat_pass_at_1_qwen3_thinking_*-predictions.jsonl",
}


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def result_dir(experiment_id: str) -> Path | None:
    path = RESULTS / experiment_id
    if (path / "metrics.json").is_file():
        return path
    candidates = sorted(path.rglob("metrics.json"))
    return candidates[0].parent if len(candidates) == 1 else None


def ledger_rows(group: str) -> list[dict[str, Any]]:
    with LEDGER.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [row for row in rows if row.get("group") == group]


def threshold_and_replicate(row: dict[str, Any]) -> tuple[float, int]:
    match = THRESHOLD_RE.search(row["run_tag"])
    if match is None:
        raise ValueError(f"Cannot parse threshold/replicate from {row['run_tag']}")
    return int(match.group("digits")) / 100, int(match.group("replicate"))


def task_map(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {task["task"]: task for task in metrics["tasks"]}


def scores(metrics: dict[str, Any]) -> dict[str, float]:
    tasks = task_map(metrics)
    ifbench_names = (
        "ifeval_ood",
        "ifeval_mt_wildchat_unused_withRewrite",
        "ifeval_mt_ood_wildchat_unused_withRewrite",
    )
    values = {
        "math500": 100
        * tasks["math500:chat"]["metrics"]["accuracy"]["minerva_math_flex"],
        "gpqa": 100
        * tasks["gpqa_diamond:qwen3_thinking"]["metrics"]["accuracy"]["multiple_choice"],
        "ifbench": 100
        * statistics.mean(
            tasks[name]["metrics"]["prompt_level_loose_acc"]["ifeval"]
            for name in ifbench_names
        ),
        "humaneval": 100
        * tasks["humaneval:chat:pass_at_1:qwen3_thinking"]["metrics"]["pass_at_1"][
            "code_exec"
        ],
    }
    values["macro"] = statistics.mean(values.values())
    return values


def combine_telemetry(path: Path) -> dict[str, Any]:
    files = sorted((path / "realized_k").glob("qwen_realized_k_*.json"))
    if len(files) != 4:
        raise RuntimeError(f"Expected four telemetry files in {path}, got {len(files)}")
    hist = [0] * 9
    layer_hists = [[0] * 9 for _ in range(48)]
    policies = set()
    for file in files:
        payload = read_json(file)
        policies.add(payload["policy"])
        if len(payload["layers"]) != 48:
            raise RuntimeError(f"Expected 48 layers in {file}")
        for layer in payload["layers"]:
            index = int(layer["layer_index"])
            for k, count in enumerate(layer["histogram"]):
                hist[k] += int(count)
                layer_hists[index][k] += int(count)
    if len(policies) != 1:
        raise RuntimeError(f"Mismatched policies in {path}: {sorted(policies)}")

    total = sum(hist)
    mean_k = sum(k * count for k, count in enumerate(hist)) / total
    layer_means = [
        sum(k * count for k, count in enumerate(layer_hist)) / sum(layer_hist)
        for layer_hist in layer_hists
    ]
    return {
        "policy": policies.pop(),
        "telemetry_files": len(files),
        "token_layer_observations": total,
        "mean_k": mean_k,
        "histogram": hist,
        "histogram_pct": [100 * count / total for count in hist],
        "layer_mean_k_min": min(layer_means),
        "layer_mean_k_max": max(layer_means),
        "layer_mean_k": layer_means,
    }


def find_tokenizer() -> Tokenizer:
    roots = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/weka/oe-eval-default/oyvindt/hf-cache/hub"),
        Path("/weka/oe-eval-default/oyvindt/hf-cache"),
    ]
    for root in roots:
        candidates = sorted(
            root.glob("models--Qwen--Qwen3-30B-A3B/snapshots/*/tokenizer.json")
        )
        if candidates:
            return Tokenizer.from_file(str(candidates[-1]))
    raise FileNotFoundError("Could not find a cached Qwen3-30B-A3B tokenizer.json")


def prediction_file(path: Path, pattern: str) -> Path:
    candidates = sorted((path / "predictions" / "Qwen_Qwen3-30B-A3B").glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {pattern} under {path}, got {candidates}")
    return candidates[0]


def repeated_span_flag(text: str) -> bool:
    words = re.findall(r"\S+", text.lower())
    if len(words) < 100:
        return False
    width = 8
    spans = Counter(tuple(words[i : i + width]) for i in range(len(words) - width + 1))
    return bool(spans and spans.most_common(1)[0][1] >= 5)


def response_diagnostics(path: Path, tokenizer: Tokenizer) -> dict[str, Any]:
    by_task: dict[str, dict[str, Any]] = {}
    all_rows = 0
    all_empty = 0
    all_cap_hits = 0
    all_repetition = 0
    for task, pattern in TASK_FILES.items():
        rows = read_jsonl(prediction_file(path, pattern))
        total_tokens: list[int] = []
        visible_tokens: list[int] = []
        chars: list[int] = []
        empty = cap_hits = repetition = marker_missing = 0
        for row in rows:
            outputs = row.get("model_output") or []
            output = outputs[0] if outputs else {}
            text = output.get("text") or ""
            generated = int(output.get("num_tokens_all", output.get("num_tokens", 0)))
            visible = len(tokenizer.encode(text, add_special_tokens=False).ids)
            total_tokens.append(generated)
            visible_tokens.append(visible)
            chars.append(len(text))
            empty += not text.strip()
            cap_hits += generated >= 32760
            repetition += repeated_span_flag(text)
            if task in {"math500", "gpqa"}:
                marker_missing += not output.get("extracted_answer")

        n = len(rows)
        by_task[task] = {
            "n": n,
            "mean_total_completion_tokens": statistics.mean(total_tokens),
            "median_total_completion_tokens": statistics.median(total_tokens),
            "p95_total_completion_tokens": sorted(total_tokens)[math.ceil(0.95 * n) - 1],
            "max_total_completion_tokens": max(total_tokens),
            "mean_returned_text_tokens": statistics.mean(visible_tokens),
            "mean_inferred_hidden_reasoning_tokens": statistics.mean(
                total - visible for total, visible in zip(total_tokens, visible_tokens, strict=True)
            ),
            "mean_returned_chars": statistics.mean(chars),
            "empty_outputs": empty,
            "cap_hits": cap_hits,
            "repeated_span_flags": repetition,
        }
        if task in {"math500", "gpqa"}:
            by_task[task]["missing_final_answer_marker"] = marker_missing
        all_rows += n
        all_empty += empty
        all_cap_hits += cap_hits
        all_repetition += repetition
    return {
        "all_tasks": {
            "rows": all_rows,
            "empty_outputs": all_empty,
            "cap_hits": all_cap_hits,
            "repeated_span_flags": all_repetition,
        },
        "tasks": by_task,
    }


def stat(values: list[float]) -> dict[str, Any]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "n": len(runs),
        "experiment_ids": [run["experiment_id"] for run in runs],
    }
    for metric in ("math500", "gpqa", "ifbench", "humaneval", "macro"):
        output[metric] = stat([run["scores"][metric] for run in runs])
    output["realized_mean_k"] = stat([run["telemetry"]["mean_k"] for run in runs])
    for task in TASK_FILES:
        task_rows = [run["responses"]["tasks"][task] for run in runs]
        output.setdefault("responses", {})[task] = {
            key: stat([float(row[key]) for row in task_rows])
            for key in (
                "mean_total_completion_tokens",
                "mean_returned_text_tokens",
                "mean_inferred_hidden_reasoning_tokens",
                "cap_hits",
                "repeated_span_flags",
            )
        }
    return output


def normalized_baseline(tokenizer: Tokenizer) -> dict[str, Any]:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows(NORMALIZED_GROUP):
        path = result_dir(row["beaker_experiment_id"])
        if path is None:
            continue
        threshold, _ = threshold_and_replicate(row)
        metrics = read_json(path / "metrics.json")
        if metrics.get("errors"):
            continue
        grouped[threshold].append(
            {"scores": scores(metrics), "responses": response_diagnostics(path, tokenizer)}
        )
    output = {}
    for threshold, runs in sorted(grouped.items()):
        summary = {
            metric: stat([run["scores"][metric] for run in runs])
            for metric in ("math500", "gpqa", "ifbench", "humaneval", "macro")
        }
        summary["responses"] = {}
        for task in TASK_FILES:
            task_rows = [run["responses"]["tasks"][task] for run in runs]
            summary["responses"][task] = {
                key: stat([float(row[key]) for row in task_rows])
                for key in (
                    "mean_total_completion_tokens",
                    "mean_returned_text_tokens",
                    "mean_inferred_hidden_reasoning_tokens",
                    "empty_outputs",
                    "cap_hits",
                    "repeated_span_flags",
                )
            }
        output[f"{threshold:.2f}"] = summary
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = find_tokenizer()

    runs = []
    for row in ledger_rows(GROUP):
        experiment_id = row["beaker_experiment_id"]
        path = result_dir(experiment_id)
        if path is None:
            continue
        threshold, replicate = threshold_and_replicate(row)
        metrics = read_json(path / "metrics.json")
        if metrics.get("errors"):
            raise RuntimeError(f"{experiment_id} has metric errors: {metrics['errors']}")
        runs.append(
            {
                "experiment_id": experiment_id,
                "threshold": threshold,
                "replicate": replicate,
                "scores": scores(metrics),
                "telemetry": combine_telemetry(path),
                "responses": response_diagnostics(path, tokenizer),
            }
        )
    runs.sort(key=lambda run: (run["threshold"], run["replicate"]))
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["threshold"]].append(run)

    payload = {
        "schema": {
            "group": GROUP,
            "partial": len(runs) < 12,
            "scores_pct": True,
            "ifbench": "equal mean of the three 32k IFBench prompt-level-loose accuracies",
            "macro": "equal mean of MATH-500, GPQA Diamond, IFBench, and HumanEval",
            "realized_k": (
                "weighted across all recorded token-layer observations and four servers; "
                "includes warmup"
            ),
            "inferred_hidden_reasoning_tokens": (
                "full completion tokens minus Qwen-tokenized returned text"
            ),
        },
        "runs_collected": len(runs),
        "runs": runs,
        "by_threshold": {
            f"{threshold:.2f}": summarize_runs(threshold_runs)
            for threshold, threshold_runs in sorted(grouped.items())
        },
        "normalized_adaptive_baseline": normalized_baseline(tokenizer),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

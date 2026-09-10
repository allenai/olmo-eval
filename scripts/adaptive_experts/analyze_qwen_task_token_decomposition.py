#!/usr/bin/env python3
"""Analyze Qwen completion-token components on matched GPQA or IFBench successes."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from analyze_qwen_math_coherence import RUNS, percentile
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "adaptive_experts"
LEDGER = ROOT / "notes" / "beaker_jobs.jsonl"
REFERENCE_RUNS = ROOT / "notes" / "qwen_math_reference_scaled_runs.json"
ANALYSIS_KS = tuple(range(4, 17))

NORMALIZED_IFBENCH_GROUP = "adaptive-experts-qwen-normalized-capability-backfill-20260717"
REFERENCE_GROUP = "adaptive-experts-qwen-reference-scaled-expanded-20260717"
IFBENCH32K_GROUP = "adaptive-experts-qwen-capability-ifbench32k-20260717"


@dataclass(frozen=True)
class TaskSpec:
    key: str
    display: str
    file_prefix: str
    expected_prompts: int


TASK_SETS = {
    "gpqa": (
        TaskSpec("gpqa", "GPQA Diamond", "gpqa_diamond_qwen3_thinking", 198),
    ),
    "ifbench": (
        TaskSpec("ifeval_ood", "IFEval OOD", "ifeval_ood_32c350", 300),
        TaskSpec(
            "ifeval_mt_wildchat",
            "IFBench MT WildChat",
            "ifeval_mt_wildchat_unused_withRewrite_380773",
            1774,
        ),
        TaskSpec(
            "ifeval_mt_ood_wildchat",
            "IFBench MT OOD WildChat",
            "ifeval_mt_ood_wildchat_unused_withRewrite_75954c",
            1387,
        ),
    ),
}

DISPLAY_NAMES = {
    "gpqa": "GPQA Diamond",
    "ifbench": "IFBench-32k macro",
}


def find_cached_tokenizer() -> Path:
    root = (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-30B-A3B"
        / "snapshots"
    )
    candidates = sorted(root.glob("*/tokenizer.json"))
    if not candidates:
        raise FileNotFoundError("No cached Qwen/Qwen3-30B-A3B tokenizer.json")
    return candidates[-1]


def read_ledger() -> list[dict]:
    with LEDGER.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def matching_ids(group: str, pattern: str) -> list[str]:
    regex = re.compile(pattern)
    matches = []
    for row in read_ledger():
        if row.get("group") != group:
            continue
        match = regex.fullmatch(row.get("run_tag", ""))
        if match is not None:
            matches.append((int(match.group(1)), row["beaker_experiment_id"]))
    return [experiment_id for _, experiment_id in sorted(matches)]


def grouped_ids(group: str, pattern: str) -> dict[int, list[str]]:
    regex = re.compile(pattern)
    matches: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for row in read_ledger():
        if row.get("group") != group:
            continue
        match = regex.fullmatch(row.get("run_tag", ""))
        if match is not None:
            matches[int(match.group(1))].append(
                (int(match.group(2)), row["beaker_experiment_id"])
            )
    return {
        k: [experiment_id for _, experiment_id in sorted(values)]
        for k, values in matches.items()
    }


def resolve_runs(task: str, routing: str) -> dict[int, list[str]]:
    if task == "gpqa" and routing == "normalized":
        return {k: list(RUNS[k]) for k in ANALYSIS_KS}
    if task == "gpqa":
        with REFERENCE_RUNS.open() as f:
            payload = json.load(f)
        return {int(k): ids for k, ids in payload["runs"].items()}

    if routing == "normalized":
        runs = grouped_ids(
            NORMALIZED_IFBENCH_GROUP,
            r"capability32k-qwen-normalized-k(\d+)-r(\d+)-20260717",
        )
        k4_label = "normalized"
    else:
        runs = grouped_ids(
            REFERENCE_GROUP,
            r"expanded-qwen-reference-k(\d+)-r(\d+)-20260717",
        )
        k4_label = "reference"
    runs[4] = matching_ids(
        IFBENCH32K_GROUP,
        rf"capability32k-{k4_label}-k4-r(\d+)-20260717",
    )
    runs[8] = matching_ids(
        IFBENCH32K_GROUP,
        r"capability32k-native-k8-r(\d+)-20260717",
    )
    return {k: runs[k] for k in ANALYSIS_KS}


def prediction_path(experiment_id: str, task: TaskSpec) -> Path:
    candidates = list(
        (RESULTS / experiment_id / "predictions").rglob(
            f"{task.file_prefix}*-predictions.jsonl"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {task.key} prediction file for {experiment_id}, got {candidates}"
        )
    return candidates[0]


def is_correct(row: dict, task: str) -> bool:
    if task == "gpqa":
        return bool(row["instance_metrics"]["accuracy"]["multiple_choice"])
    return bool(row["instance_metrics"]["prompt_level_loose_acc"]["ifeval"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASK_SETS), required=True)
    parser.add_argument(
        "--routing",
        choices=("normalized", "reference_scaled"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()

    tasks = TASK_SETS[args.task]
    runs = resolve_runs(args.task, args.routing)
    tokenizer_path = args.tokenizer or find_cached_tokenizer()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    missing_ks = sorted(set(ANALYSIS_KS) - runs.keys())
    if missing_ks:
        raise RuntimeError(f"Run set is missing K values: {missing_ks}")

    records: dict[tuple[int, int, str, int], dict] = {}
    expected_doc_ids: dict[str, set[int]] = {}
    negative_residuals = []
    for k in ANALYSIS_KS:
        for replicate, experiment_id in enumerate(runs[k], start=1):
            for task in tasks:
                with prediction_path(experiment_id, task).open() as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                if len(rows) != task.expected_prompts:
                    raise RuntimeError(
                        f"Incomplete {task.key} run {experiment_id}: {len(rows)} rows"
                    )
                doc_ids = {int(row["doc_id"]) for row in rows}
                expected = expected_doc_ids.setdefault(task.key, doc_ids)
                if doc_ids != expected:
                    raise RuntimeError(
                        f"Prompt IDs changed for {task.key} in experiment {experiment_id}"
                    )
                for row in rows:
                    samples = row.get("model_output") or []
                    sample = samples[0] if samples else {}
                    text = row.get("final_output") or sample.get("text") or ""
                    total_tokens = int(
                        sample.get("num_tokens_all", sample.get("num_tokens", 0))
                    )
                    returned_tokens = len(
                        tokenizer.encode(text, add_special_tokens=False).ids
                    )
                    inferred_tokens = total_tokens - returned_tokens
                    if inferred_tokens < 0:
                        negative_residuals.append(
                            (k, replicate, task.key, row["doc_id"], inferred_tokens)
                        )
                    records[(k, replicate, task.key, int(row["doc_id"]))] = {
                        "correct": is_correct(row, args.task),
                        "total_tokens": total_tokens,
                        "returned_solution_tokens": returned_tokens,
                        "inferred_reasoning_and_boundary_tokens": max(inferred_tokens, 0),
                    }
    if negative_residuals:
        raise RuntimeError(f"Returned text exceeded total tokens: {negative_residuals[:5]}")

    common_ids: dict[str, list[int]] = {}
    for task in tasks:
        common_ids[task.key] = sorted(
            doc_id
            for doc_id in expected_doc_ids[task.key]
            if all(
                records[(k, replicate, task.key, doc_id)]["correct"]
                for k in ANALYSIS_KS
                for replicate in range(1, len(runs[k]) + 1)
            )
        )
        if not common_ids[task.key]:
            raise RuntimeError(f"No common-success prompts remain for {task.key}")

    performance = {}
    for k in ANALYSIS_KS:
        replicate_scores = []
        for replicate in range(1, len(runs[k]) + 1):
            task_scores = []
            for task in tasks:
                task_scores.append(
                    100
                    * statistics.mean(
                        records[(k, replicate, task.key, doc_id)]["correct"]
                        for doc_id in expected_doc_ids[task.key]
                    )
                )
            replicate_scores.append(statistics.mean(task_scores))
        performance[str(k)] = {
            "replicate_scores_pct": replicate_scores,
            "mean_pct": statistics.mean(replicate_scores),
            "min_pct": min(replicate_scores),
            "max_pct": max(replicate_scores),
            "sample_sd_pct": (
                statistics.stdev(replicate_scores) if len(replicate_scores) > 1 else 0.0
            ),
        }

    components = (
        "total_tokens",
        "returned_solution_tokens",
        "inferred_reasoning_and_boundary_tokens",
    )
    prompt_medians: dict[tuple[int, str, int, str], float] = {}
    for k in ANALYSIS_KS:
        for task in tasks:
            for doc_id in common_ids[task.key]:
                for component in components:
                    prompt_medians[(k, task.key, doc_id, component)] = percentile(
                        [
                            records[(k, replicate, task.key, doc_id)][component]
                            for replicate in range(1, len(runs[k]) + 1)
                        ],
                        0.5,
                    )

    by_k = {}
    for k in ANALYSIS_KS:
        task_means: dict[str, dict[str, float]] = {}
        task_ratios: dict[str, dict[str, float]] = {}
        for task in tasks:
            task_means[task.key] = {}
            task_ratios[task.key] = {}
            for component in components:
                prompt_means = []
                prompt_ratios = []
                for doc_id in common_ids[task.key]:
                    prompt_means.append(
                        statistics.mean(
                            records[(k, replicate, task.key, doc_id)][component]
                            for replicate in range(1, len(runs[k]) + 1)
                        )
                    )
                    prompt_ratios.append(
                        prompt_medians[(k, task.key, doc_id, component)]
                        / max(prompt_medians[(8, task.key, doc_id, component)], 1)
                    )
                task_means[task.key][component] = statistics.mean(prompt_means)
                task_ratios[task.key][component] = percentile(prompt_ratios, 0.5)

        means = {
            component: statistics.mean(
                task_means[task.key][component] for task in tasks
            )
            for component in components
        }
        ratios = {
            component: statistics.mean(
                task_ratios[task.key][component] for task in tasks
            )
            for component in components
        }
        by_k[str(k)] = {
            "replicates": len(runs[k]),
            "mean_of_prompt_means_task_balanced": means,
            "median_prompt_ratio_vs_k8_task_balanced": ratios,
            "per_task_mean_of_prompt_means": task_means,
            "per_task_median_prompt_ratio_vs_k8": task_ratios,
            "mean_returned_solution_share_pct": (
                100 * means["returned_solution_tokens"] / means["total_tokens"]
            ),
        }

    payload = {
        "schema": {
            "task": args.task,
            "task_display": DISPLAY_NAMES[args.task],
            "routing": args.routing,
            "performance": (
                "Mean accuracy over all prompts; IFBench first averages within each of its "
                "three tasks and then averages tasks equally."
            ),
            "cohort": (
                "Prompt IDs correct in every available repetition at every K from 4 through 16; "
                "IFBench token summaries weight its three tasks equally."
            ),
            "returned_solution_tokens": (
                "Qwen tokenizer count of parser-returned assistant content, without added "
                "special tokens"
            ),
            "inferred_reasoning_and_boundary_tokens": (
                "server full completion tokens minus returned_solution_tokens; includes hidden "
                "reasoning and a small number of parser delimiter/EOS/boundary tokens"
            ),
            "tokenizer": str(tokenizer_path),
        },
        "n_all_prompts": sum(task.expected_prompts for task in tasks),
        "n_all_prompts_by_task": {
            task.key: task.expected_prompts for task in tasks
        },
        "n_common_success_prompts": sum(len(ids) for ids in common_ids.values()),
        "n_common_success_prompts_by_task": {
            task.key: len(common_ids[task.key]) for task in tasks
        },
        "common_success_doc_ids_by_task": common_ids,
        "run_ids": {str(k): runs[k] for k in ANALYSIS_KS},
        "performance_all_prompts": performance,
        "by_k": by_k,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

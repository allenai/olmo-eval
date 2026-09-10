#!/usr/bin/env python3
"""Analyze the three-seed GLM-5.2 AIME 2026 routing comparison."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


POLICIES = ("native", "reference")
OUTPUT_CAP = 163_840


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def fourgram_repetition(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[index : index + 4]) for index in range(max(0, len(tokens) - 3))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def compression_ratio(text: str) -> float:
    if not text:
        return 0.0
    encoded = text.encode()
    return len(zlib.compress(encoded)) / len(encoded)


def prediction_path(run_dir: Path) -> Path:
    paths = sorted(run_dir.glob("predictions/**/*.jsonl"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one prediction file under {run_dir}, found {len(paths)}")
    return paths[0]


def load_run(run_dir: Path, policy: str, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with prediction_path(run_dir).open() as file:
        for line in file:
            problem = json.loads(line)
            for sample_index, output in enumerate(problem["model_output"]):
                text = output.get("text") or ""
                records.append(
                    {
                        "policy": policy,
                        "seed": seed,
                        "problem": int(problem["native_id"]),
                        "sample": sample_index,
                        "label": str(problem["label"]),
                        "text": text,
                        "answer": output.get("extracted_answer") or "",
                        "correct": int(
                            output["sample_metrics"]["minerva_math_flex"]["minerva_math_flex"]
                        ),
                        "tokens": int(output["num_tokens"]),
                        "chars": len(text),
                        "fourgram_repetition": fourgram_repetition(text),
                        "compression_ratio": compression_ratio(text),
                    }
                )
    if len(records) != 960:
        raise RuntimeError(f"Expected 960 generations in {run_dir}, found {len(records)}")
    return records


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def summarize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_problem: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        per_problem[record["problem"]].append(record)
    return {
        "correct": sum(record["correct"] for record in records),
        "pass_at_1": statistics.mean(record["correct"] for record in records),
        "pass_at_32": statistics.mean(
            any(record["correct"] for record in problem_records)
            for problem_records in per_problem.values()
        ),
        "zero_correct_problems": sorted(
            problem
            for problem, problem_records in per_problem.items()
            if not any(record["correct"] for record in problem_records)
        ),
        "empty_text": sum(not record["text"] for record in records),
        "empty_answer": sum(not record["answer"] for record in records),
        "at_output_cap": sum(record["tokens"] >= OUTPUT_CAP for record in records),
        # This deliberately conservative heuristic catches only gross loops.
        "severe_repetition": sum(
            record["fourgram_repetition"] >= 0.90
            and record["compression_ratio"] <= 0.10
            for record in records
        ),
        "tokens": distribution([record["tokens"] for record in records]),
        "visible_chars": distribution([record["chars"] for record in records]),
    }


def problem_accuracy(records: list[dict[str, Any]]) -> dict[int, float]:
    values: dict[int, list[int]] = defaultdict(list)
    for record in records:
        values[record["problem"]].append(record["correct"])
    return {problem: statistics.mean(scores) for problem, scores in values.items()}


def bootstrap_problem_difference(
    native: list[dict[str, Any]], reference: list[dict[str, Any]], *, samples: int = 20_000
) -> tuple[float, float]:
    native_by_problem = problem_accuracy(native)
    reference_by_problem = problem_accuracy(reference)
    problems = sorted(native_by_problem)
    if problems != sorted(reference_by_problem):
        raise RuntimeError("Policies do not contain the same AIME problems")
    differences = [reference_by_problem[problem] - native_by_problem[problem] for problem in problems]
    rng = random.Random(20260902)
    draws = [
        statistics.mean(rng.choice(differences) for _ in problems)
        for _ in range(samples)
    ]
    return percentile(draws, 0.025), percentile(draws, 0.975)


def paired_subset(
    native: list[dict[str, Any]], reference: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    native_by_key = {
        (record["seed"], record["problem"], record["sample"]): record for record in native
    }
    reference_by_key = {
        (record["seed"], record["problem"], record["sample"]): record for record in reference
    }
    keys = sorted(set(native_by_key) & set(reference_by_key))
    keys = [key for key in keys if native_by_key[key][field] and reference_by_key[key][field]]
    return {
        "n": len(keys),
        "native_accuracy": statistics.mean(native_by_key[key]["correct"] for key in keys),
        "reference_accuracy": statistics.mean(reference_by_key[key]["correct"] for key in keys),
        "native_tokens": distribution([native_by_key[key]["tokens"] for key in keys]),
        "reference_tokens": distribution([reference_by_key[key]["tokens"] for key in keys]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/glm52_aime_replicates")
    )
    args = parser.parse_args()

    runs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for policy in POLICIES:
        for seed in range(3):
            run_dir = args.input_dir / f"{policy}-seed{seed}"
            runs[(policy, seed)] = load_run(run_dir, policy, seed)

    run_summaries = {
        f"{policy}-seed{seed}": summarize_run(records)
        for (policy, seed), records in runs.items()
    }
    combined = {
        policy: [record for seed in range(3) for record in runs[(policy, seed)]]
        for policy in POLICIES
    }

    policy_summaries: dict[str, Any] = {}
    for policy, records in combined.items():
        seed_pass_1 = [run_summaries[f"{policy}-seed{seed}"]["pass_at_1"] for seed in range(3)]
        seed_pass_32 = [
            run_summaries[f"{policy}-seed{seed}"]["pass_at_32"] for seed in range(3)
        ]
        per_problem = problem_accuracy(records)
        policy_summaries[policy] = {
            "correct": sum(record["correct"] for record in records),
            "generations": len(records),
            "pooled_pass_at_1": statistics.mean(record["correct"] for record in records),
            "seed_pass_at_1": seed_pass_1,
            "seed_pass_at_1_mean": statistics.mean(seed_pass_1),
            "seed_pass_at_1_sample_sd": statistics.stdev(seed_pass_1),
            "seed_pass_at_32": seed_pass_32,
            "seed_pass_at_32_mean": statistics.mean(seed_pass_32),
            "pooled_pass_at_96": statistics.mean(score > 0 for score in per_problem.values()),
            "empty_text": sum(not record["text"] for record in records),
            "empty_answer": sum(not record["answer"] for record in records),
            "at_output_cap": sum(record["tokens"] >= OUTPUT_CAP for record in records),
            "severe_repetition": sum(
                record["fourgram_repetition"] >= 0.90
                and record["compression_ratio"] <= 0.10
                for record in records
            ),
            "tokens": distribution([record["tokens"] for record in records]),
            "visible_chars": distribution([record["chars"] for record in records]),
        }

    native_problem = problem_accuracy(combined["native"])
    reference_problem = problem_accuracy(combined["reference"])
    problem_differences = {
        problem: reference_problem[problem] - native_problem[problem]
        for problem in sorted(native_problem)
    }
    paired_seed_differences = [
        run_summaries[f"reference-seed{seed}"]["pass_at_1"]
        - run_summaries[f"native-seed{seed}"]["pass_at_1"]
        for seed in range(3)
    ]
    interval = bootstrap_problem_difference(combined["native"], combined["reference"])

    output = {
        "run_summaries": run_summaries,
        "policy_summaries": policy_summaries,
        "comparison": {
            "reference_minus_native_pass_at_1": statistics.mean(paired_seed_differences),
            "paired_seed_differences": paired_seed_differences,
            "paired_seed_difference_sample_sd": statistics.stdev(paired_seed_differences),
            "problem_cluster_bootstrap_95_interval": interval,
            "problems_reference_better_equal_worse": {
                "better": sum(value > 0 for value in problem_differences.values()),
                "equal": sum(value == 0 for value in problem_differences.values()),
                "worse": sum(value < 0 for value in problem_differences.values()),
            },
            "largest_reference_losses": sorted(
                problem_differences.items(), key=lambda item: item[1]
            )[:10],
            "largest_reference_gains": sorted(
                problem_differences.items(), key=lambda item: item[1], reverse=True
            )[:10],
            "common_nonempty_text": paired_subset(
                combined["native"], combined["reference"], "text"
            ),
            "common_extracted_answer": paired_subset(
                combined["native"], combined["reference"], "answer"
            ),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

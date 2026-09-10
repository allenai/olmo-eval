#!/usr/bin/env python3
"""Split Qwen MATH-500 completion tokens into returned text and inferred reasoning.

olmo-eval retains the parser-returned assistant content and the full completion-token count, but
not vLLM's separate ``reasoning_content`` field. Tokenizing the returned content with Qwen's own
tokenizer lets us infer the omitted component by subtraction. The residual also contains a handful
of generated reasoning delimiters, finalization markers, and EOS/boundary tokens, so it should be
read as an extremely close estimate rather than a byte-exact reasoning field.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_qwen_math_coherence import (
    RUNS,
    all_runs_correct,
    load_run,
    one_file,
    percentile,
    read_jsonl,
)
from tokenizers import Tokenizer

ANALYSIS_KS = tuple(range(4, 17))


def find_cached_tokenizer() -> Path:
    root = (
        Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3-30B-A3B" / "snapshots"
    )
    candidates = sorted(root.glob("*/tokenizer.json"))
    if not candidates:
        raise FileNotFoundError(
            "No cached Qwen/Qwen3-30B-A3B tokenizer.json; pass --tokenizer explicitly"
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--runs-manifest",
        type=Path,
        help="Optional JSON manifest with a routing label and K-to-experiment-ID mapping.",
    )
    args = parser.parse_args()

    if args.runs_manifest is None:
        runs = {k: tuple(ids) for k, ids in RUNS.items()}
        routing_label = "normalized"
        expected_common_correct = 416
    else:
        with args.runs_manifest.open() as f:
            manifest = json.load(f)
        runs = {int(k): tuple(ids) for k, ids in manifest["runs"].items()}
        routing_label = manifest["routing"]
        expected_common_correct = manifest.get("expected_common_correct")

    missing_ks = sorted(set(ANALYSIS_KS) - runs.keys())
    if missing_ks:
        raise RuntimeError(f"Run manifest is missing K values: {missing_ks}")

    tokenizer_path = args.tokenizer or find_cached_tokenizer()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    canonical_requests = {
        row["doc_id"]: row for row in read_jsonl(one_file(runs[8][0], "requests"))
    }
    responses = []
    for k in ANALYSIS_KS:
        experiment_ids = runs[k]
        for replicate, experiment_id in enumerate(experiment_ids, start=1):
            responses.extend(load_run(k, replicate, experiment_id, canonical_requests))
    groups = defaultdict(list)
    for item in responses:
        groups[(item.k, item.doc_id)].append(item)
    for k in ANALYSIS_KS:
        for doc_id in range(500):
            items = groups[(k, doc_id)]
            if len(items) != len(runs[k]):
                raise RuntimeError(
                    f"Expected {len(runs[k])} responses for K={k}, doc={doc_id}; got {len(items)}"
                )
            items.sort(key=lambda item: item.replicate)

    common_doc_ids = {
        doc_id
        for doc_id in range(500)
        if all(all_runs_correct(groups[(k, doc_id)]) for k in ANALYSIS_KS)
    }
    if expected_common_correct is not None and len(common_doc_ids) != expected_common_correct:
        raise RuntimeError(
            f"Expected {expected_common_correct} common-correct prompts, got {len(common_doc_ids)}"
        )

    components: dict[tuple[int, int, int], dict[str, int]] = {}
    negative_residuals = []
    for item in responses:
        returned_tokens = len(tokenizer.encode(item.text, add_special_tokens=False).ids)
        inferred_reasoning_tokens = item.generated_tokens - returned_tokens
        if inferred_reasoning_tokens < 0:
            negative_residuals.append(
                (item.k, item.replicate, item.doc_id, inferred_reasoning_tokens)
            )
        components[(item.k, item.replicate, item.doc_id)] = {
            "total_tokens": item.generated_tokens,
            "returned_solution_tokens": returned_tokens,
            "inferred_reasoning_and_boundary_tokens": max(inferred_reasoning_tokens, 0),
        }
    if negative_residuals:
        raise RuntimeError(f"Returned text exceeded total tokens: {negative_residuals[:5]}")

    component_names = (
        "total_tokens",
        "returned_solution_tokens",
        "inferred_reasoning_and_boundary_tokens",
    )
    by_k = {}
    prompt_medians: dict[tuple[int, int, str], float] = {}
    for k in ANALYSIS_KS:
        for doc_id in sorted(common_doc_ids):
            rows = groups[(k, doc_id)]
            for name in component_names:
                values = [components[(k, item.replicate, doc_id)][name] for item in rows]
                prompt_medians[(k, doc_id, name)] = percentile(values, 0.5)

    for k in ANALYSIS_KS:
        prompt_means = {name: [] for name in component_names}
        prompt_median_values = {name: [] for name in component_names}
        for doc_id in sorted(common_doc_ids):
            rows = groups[(k, doc_id)]
            for name in component_names:
                values = [components[(k, item.replicate, doc_id)][name] for item in rows]
                prompt_means[name].append(statistics.mean(values))
                value = prompt_medians[(k, doc_id, name)]
                prompt_median_values[name].append(value)

        means = {name: statistics.mean(values) for name, values in prompt_means.items()}
        medians = {name: percentile(values, 0.5) for name, values in prompt_median_values.items()}
        ratios = {}
        for name in component_names:
            prompt_ratios = [
                prompt_medians[(k, doc_id, name)] / max(prompt_medians[(8, doc_id, name)], 1)
                for doc_id in sorted(common_doc_ids)
            ]
            ratios[name] = {
                "mean": statistics.mean(prompt_ratios),
                "median": percentile(prompt_ratios, 0.5),
            }
        by_k[str(k)] = {
            "replicates": len(runs[k]),
            "mean_of_prompt_means": means,
            "median_of_prompt_medians": medians,
            "median_prompt_ratio_vs_k8": {
                name: values["median"] for name, values in ratios.items()
            },
            "mean_prompt_ratio_vs_k8": {name: values["mean"] for name, values in ratios.items()},
            "mean_returned_solution_share_pct": 100
            * means["returned_solution_tokens"]
            / means["total_tokens"],
        }

    def comparison(start: int, end: int) -> dict:
        start_means = by_k[str(start)]["mean_of_prompt_means"]
        end_means = by_k[str(end)]["mean_of_prompt_means"]
        drops = {name: start_means[name] - end_means[name] for name in component_names}
        total_drop = drops["total_tokens"]
        return {
            "start_k": start,
            "end_k": end,
            "mean_token_change": drops,
            "share_of_total_drop_pct": {
                "returned_solution_tokens": 100 * drops["returned_solution_tokens"] / total_drop,
                "inferred_reasoning_and_boundary_tokens": 100
                * drops["inferred_reasoning_and_boundary_tokens"]
                / total_drop,
            },
        }

    adjacent = [comparison(k, k + 1) for k in range(4, 16)]
    math500_accuracy = {}
    for k in ANALYSIS_KS:
        replicate_scores = []
        for replicate in range(1, len(runs[k]) + 1):
            items = [
                item
                for doc_id in range(500)
                for item in groups[(k, doc_id)]
                if item.replicate == replicate
            ]
            replicate_scores.append(100 * statistics.mean(item.correct for item in items))
        math500_accuracy[str(k)] = {
            "replicate_scores_pct": replicate_scores,
            "mean_pct": statistics.mean(replicate_scores),
            "min_pct": min(replicate_scores),
            "max_pct": max(replicate_scores),
            "sample_sd_pct": statistics.stdev(replicate_scores),
        }

    payload = {
        "schema": {
            "cohort": (
                f"{len(common_doc_ids)} prompt IDs correct in every available repetition at "
                "every K from 4 through 16"
            ),
            "routing": routing_label,
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
        "n_prompts": len(common_doc_ids),
        "doc_ids": sorted(common_doc_ids),
        "run_ids": {str(k): list(runs[k]) for k in ANALYSIS_KS},
        "math500_accuracy_all_prompts": math500_accuracy,
        "by_k": by_k,
        "comparisons": {
            "k4_to_k8": comparison(4, 8),
            "k8_to_k16": comparison(8, 16),
            "k4_to_k16": comparison(4, 16),
            "adjacent_k4_to_k16": adjacent,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()

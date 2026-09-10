#!/usr/bin/env python3
"""Plot the complete valid-v4 Qwen3.5 expert-sweep results."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "notes" / "beaker_jobs.jsonl"
RESULTS = ROOT / "results" / "adaptive_experts"
OUTPUT = ROOT / "notes" / "plots"
CSV_PATH = ROOT / "notes" / "qwen35_35b_a3b_results.csv"
TAG = re.compile(r"^qwen35-k(?P<k>[4-8])-(?P<routing>normalized|reference)-r\d+-v4-20260725$")

METRICS = (
    "MATH-500 ↑",
    "GPQA Diamond ↑",
    "IFBench macro ↑",
    "HumanEval pass@1 ↑",
)
STYLES = {
    "MATH-500 ↑": ("#0072B2", "o"),
    "GPQA Diamond ↑": ("#E69F00", "s"),
    "IFBench macro ↑": ("#009E73", "^"),
    "HumanEval pass@1 ↑": ("#D55E00", "D"),
}


def metric_path(experiment_id: str) -> Path | None:
    candidates = list((RESULTS / experiment_id).rglob("metrics.json"))
    return candidates[0] if candidates else None


def scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    if payload["errors"]:
        raise RuntimeError(f"Metric errors in {path}: {payload['errors']}")
    tasks = {task["task"]: task for task in payload["tasks"]}
    ifbench_names = (
        "ifeval_ood",
        "ifeval_mt_wildchat_unused_withRewrite",
        "ifeval_mt_ood_wildchat_unused_withRewrite",
    )
    return {
        "MATH-500 ↑": 100 * tasks["math500:chat"]["metrics"]["accuracy"]["minerva_math_flex"],
        "GPQA Diamond ↑": 100
        * tasks["gpqa_diamond:qwen3_thinking"]["metrics"]["accuracy"]["multiple_choice"],
        "IFBench macro ↑": 100
        * float(
            np.mean(
                [tasks[name]["metrics"]["prompt_level_loose_acc"]["ifeval"] for name in ifbench_names]
            )
        ),
        "HumanEval pass@1 ↑": 100
        * tasks["humaneval:chat:pass_at_1:qwen3_thinking"]["metrics"]["pass_at_1"]["code_exec"],
    }


def completed() -> dict[str, dict[int, list[tuple[str, dict[str, float]]]]]:
    grouped: dict[str, dict[int, list[tuple[str, dict[str, float]]]]] = defaultdict(lambda: defaultdict(list))
    for line in LEDGER.read_text().splitlines():
        row = json.loads(line)
        match = TAG.fullmatch(row.get("run_tag", ""))
        if not match:
            continue
        experiment_id = row["beaker_experiment_id"]
        path = metric_path(experiment_id)
        if path is None:
            continue
        grouped[match.group("routing")][int(match.group("k"))].append((experiment_id, scores(path)))
    return grouped


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_csv(grouped: dict[str, dict[int, list[tuple[str, dict[str, float]]]]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=("routing", "k", "metric", "n", "mean", "sd", "experiment_ids")
        )
        writer.writeheader()
        for routing, points in sorted(grouped.items()):
            for k, runs in sorted(points.items()):
                for metric in METRICS:
                    mean, sd = summarize([run_scores[metric] for _, run_scores in runs])
                    writer.writerow(
                        {
                            "routing": routing,
                            "k": k,
                            "metric": metric,
                            "n": len(runs),
                            "mean": f"{mean:.6f}",
                            "sd": f"{sd:.6f}",
                            "experiment_ids": ";".join(experiment_id for experiment_id, _ in runs),
                        }
                    )


def plot(grouped: dict[str, dict[int, list[tuple[str, dict[str, float]]]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharey=True)
    fig.suptitle("Qwen3.5-35B-A3B: Current-Suite Performance vs Active Experts", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Complete valid-v4 sweep; lines are replicate means and shading is ±1 sample SD. "
        "Seventeen over-context IFBench prompts per run are retained as failures.",
        ha="center",
        color="#555A63",
        fontsize=10.2,
    )
    for ax, routing, title in zip(
        axes,
        ("normalized", "reference"),
        ("Normalized top-K", "K=8-reference scaled (not renormalized)"),
        strict=True,
    ):
        points = grouped.get(routing, {})
        ks = np.asarray(sorted(points))
        if len(ks):
            for metric in METRICS:
                means, sds = zip(
                    *(summarize([run_scores[metric] for _, run_scores in points[k]]) for k in ks),
                    strict=True,
                )
                means_array = np.asarray(means)
                sds_array = np.asarray(sds)
                color, marker = STYLES[metric]
                ax.fill_between(
                    ks,
                    np.maximum(0, means_array - sds_array),
                    np.minimum(100, means_array + sds_array),
                    color=color,
                    alpha=0.13,
                )
                ax.plot(ks, means_array, color=color, marker=marker, linewidth=2.2, markersize=6, label=metric)
            for k in ks:
                ax.text(k, 2.5, f"n={len(points[int(k)])}", ha="center", fontsize=8.5, color="#666A73")
        ax.set_xlim(3.7, 8.3)
        ax.set_xticks(range(4, 9))
        ax.set_ylim(0, 102)
        ax.set_title(title, fontweight="semibold", pad=10)
        ax.set_xlabel("Active experts per token", fontweight="semibold")
        ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.8)
        ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.6)
        ax.axvline(8, color="#73777F", linestyle="--", linewidth=1.1, alpha=0.7, zorder=0)
    axes[0].set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.88), ncol=4, frameon=False)
    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.79, wspace=0.08)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT / f"qwen35_35b_a3b_expert_sweep.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    grouped = completed()
    write_csv(grouped)
    plot(grouped)
    print({routing: {k: len(runs) for k, runs in sorted(points.items())} for routing, points in grouped.items()})


if __name__ == "__main__":
    main()

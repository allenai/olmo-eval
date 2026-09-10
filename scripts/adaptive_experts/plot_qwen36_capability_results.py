#!/usr/bin/env python3
"""Plot the available Qwen3.6 capability-suite expert sweep."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "notes" / "beaker_jobs.jsonl"
RESULTS = ROOT / "results" / "adaptive_experts"
OUTPUT = ROOT / "notes" / "plots"
CSV_PATH = ROOT / "notes" / "qwen36_capability_results.csv"
TAG = re.compile(
    r"^qwen36-k(?P<k>4|6|8|10|12|16|32)-(?P<routing>reference|native)-"
    r"full-suite-r(?P<replicate>[1-3])-seed(?P<seed>42|43|44)-"
    r"(?P<version>v1-20260812|nestedfix1-20260815|bridge1-20260817)$"
)
# The 20260812 v1 commands for K>8 set a nonexistent outer config field and
# therefore ran at native K=8. Keep only the scientifically valid historical
# conditions; corrected K>8 runs must use a new tag/version.
VALID_HISTORICAL_KS = frozenset({4, 6, 8})

METRICS = (
    "MATH-500 ↑",
    "GPQA Diamond ↑",
    "IFBench macro ↑",
    "AIME 2025 pass@1 ↑",
)
STYLES = {
    "MATH-500 ↑": ("#0072B2", "o"),
    "GPQA Diamond ↑": ("#E69F00", "s"),
    "IFBench macro ↑": ("#009E73", "^"),
    "AIME 2025 pass@1 ↑": ("#CC79A7", "D"),
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
        "AIME 2025 pass@1 ↑": 100
        * tasks["aime_2025:pass_at_32"]["metrics"]["pass_at_1"]["minerva_math_flex"],
    }


def completed() -> dict[int, list[tuple[int, int, str, str, dict[str, float]]]]:
    grouped: dict[int, list[tuple[int, int, str, str, dict[str, float]]]] = defaultdict(list)
    seen: set[str] = set()
    for line in LEDGER.read_text().splitlines():
        row = json.loads(line)
        match = TAG.fullmatch(row.get("run_tag", ""))
        if not match:
            continue
        k = int(match.group("k"))
        version = match.group("version")
        if version == "v1-20260812" and k not in VALID_HISTORICAL_KS:
            continue
        if version == "nestedfix1-20260815" and k not in {10, 12}:
            continue
        if version == "bridge1-20260817" and k not in {16, 32}:
            continue
        experiment_id = row["beaker_experiment_id"]
        if experiment_id in seen:
            continue
        path = metric_path(experiment_id)
        if path is None:
            continue
        seen.add(experiment_id)
        grouped[k].append(
            (
                int(match.group("replicate")),
                int(match.group("seed")),
                match.group("routing"),
                experiment_id,
                scores(path),
            )
        )
    for runs in grouped.values():
        runs.sort(key=lambda row: row[0])
    return grouped


def summarize(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def write_csv(grouped: dict[int, list[tuple[int, int, str, str, dict[str, float]]]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=("k", "routing", "metric", "n", "mean", "sd", "experiment_ids"),
        )
        writer.writeheader()
        for k, runs in sorted(grouped.items()):
            routing = runs[0][2]
            for metric in METRICS:
                mean, sd = summarize([run_scores[metric] for *_, run_scores in runs])
                writer.writerow(
                    {
                        "k": k,
                        "routing": routing,
                        "metric": metric,
                        "n": len(runs),
                        "mean": f"{mean:.6f}",
                        "sd": f"{sd:.6f}",
                        "experiment_ids": ";".join(run[3] for run in runs),
                    }
                )


def plot(grouped: dict[int, list[tuple[int, int, str, str, dict[str, float]]]]) -> None:
    ks = np.asarray(sorted(grouped))
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fig.suptitle(
        "Qwen3.6-35B-A3B: Capability Performance vs Active Experts",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.915,
        "Validated routing conditions only; lines are replicate means and shading is ±1 sample SD",
        ha="center",
        color="#555A63",
        fontsize=9.8,
    )
    for metric in METRICS:
        means, sds = zip(
            *(summarize([run_scores[metric] for *_, run_scores in grouped[int(k)]]) for k in ks),
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
            alpha=0.14,
        )
        ax.plot(
            ks,
            means_array,
            color=color,
            marker=marker,
            linewidth=2.3,
            markersize=7,
            label=metric,
        )
    for k in ks:
        ax.text(k, 2.5, f"n={len(grouped[int(k)])}", ha="center", fontsize=8.7, color="#666A73")
    if 8 in grouped:
        ax.axvline(8, color="#73777F", linestyle="--", linewidth=1.1, alpha=0.7, zorder=0)
    ax.set_xlim(float(ks.min()) - 0.5, float(ks.max()) + 0.5)
    ax.set_xticks(ks)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Active experts per token", fontweight="semibold")
    ax.set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.6)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(top=0.84, bottom=0.20)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT / f"qwen36_capability_expert_sweep.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    grouped = completed()
    if not grouped:
        raise RuntimeError("No completed Qwen3.6 capability runs found")
    write_csv(grouped)
    plot(grouped)
    print({k: len(runs) for k, runs in sorted(grouped.items())})


if __name__ == "__main__":
    main()

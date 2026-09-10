#!/usr/bin/env python3
"""Plot matched GPQA or IFBench completion-token decompositions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def padded_limits(
    values: np.ndarray, minimum_span: float, lower_bound: float = 0
) -> tuple[float, float]:
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, minimum_span)
    pad = 0.22 * span
    center = (low + high) / 2
    return max(lower_bound, center - span / 2 - pad), center + span / 2 + pad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open() as f:
        report = json.load(f)

    schema = report["schema"]
    task = schema["task_display"]
    routing = schema["routing"].replace("reference_scaled", "K=8-reference-scaled")
    ks = np.asarray(sorted(int(k) for k in report["by_k"]), dtype=int)
    by_k = report["by_k"]

    def mean_values(field: str) -> np.ndarray:
        return np.asarray(
            [
                by_k[str(k)]["mean_of_prompt_means_task_balanced"][field]
                for k in ks
            ],
            dtype=float,
        )

    def ratio_values(field: str) -> np.ndarray:
        return np.asarray(
            [
                by_k[str(k)]["median_prompt_ratio_vs_k8_task_balanced"][field]
                for k in ks
            ],
            dtype=float,
        )

    reasoning = mean_values("inferred_reasoning_and_boundary_tokens")
    returned = mean_values("returned_solution_tokens")
    performance = report["performance_all_prompts"]
    perf_mean = np.asarray([performance[str(k)]["mean_pct"] for k in ks])
    perf_min = np.asarray([performance[str(k)]["min_pct"] for k in ks])
    perf_max = np.asarray([performance[str(k)]["max_pct"] for k in ks])

    sns.set_theme(style="whitegrid", context="talk")
    colors = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    fig.suptitle(
        f"Qwen3-30B-A3B {task} performance and token decomposition ({routing})",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    if schema["task"] == "ifbench":
        subtitle = (
            f"Performance uses all {report['n_all_prompts']:,} prompts; token panels use "
            f"{report['n_common_success_prompts']:,} common-success prompts; both weight the "
            "three IFBench tasks equally"
        )
    else:
        subtitle = (
            f"Performance uses all {report['n_all_prompts']:,} prompts; token panels use "
            f"{report['n_common_success_prompts']:,} prompts correct in every available K=4–16 run"
        )
    fig.text(0.5, 0.925, subtitle, ha="center", color="#555A63", fontsize=11)

    ax = axes[0]
    ax.fill_between(
        ks,
        perf_min,
        perf_max,
        color=colors[0],
        alpha=0.2,
        linewidth=0,
        label="Replicate min–max",
    )
    ax.plot(
        ks,
        perf_mean,
        marker="o",
        color=colors[0],
        linestyle=(0, (2, 2)),
        linewidth=2.4,
        label="Replicate mean",
    )
    ax.set_title(f"{task} performance (all prompts)")
    ax.set_ylabel("Accuracy / macro score (%) ↑")
    ax.set_ylim(*padded_limits(np.concatenate((perf_min, perf_max)), 4.0))
    ax.legend(frameon=False, fontsize=10)
    ax.text(
        0.01,
        0.03,
        "Zoomed y-axis",
        transform=ax.transAxes,
        color="#666B73",
        fontsize=9,
    )

    ax = axes[1]
    ax.bar(
        ks,
        reasoning,
        color=colors[0],
        label="Inferred hidden reasoning + boundaries",
    )
    ax.bar(
        ks,
        returned,
        bottom=reasoning,
        color=colors[2],
        label="Parser-returned response",
    )
    ax.set_title("Mean full-generation tokens")
    ax.set_ylabel("Tokens per response ↓")
    ax.set_ylim(0, math.ceil(float((reasoning + returned).max()) / 500) * 500 * 1.08)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[2]
    ratio_fields = (
        ("total_tokens", "Full generation", colors[1], "o"),
        ("inferred_reasoning_and_boundary_tokens", "Inferred hidden reasoning", colors[0], "s"),
        ("returned_solution_tokens", "Returned response", colors[2], "^"),
    )
    all_ratios = []
    for field, label, color, marker in ratio_fields:
        values = ratio_values(field)
        all_ratios.append(values)
        ax.plot(ks, values, color=color, marker=marker, linewidth=2.2, label=label)
    ax.axhline(1.0, color="#73777F", linestyle="--", linewidth=1.1, alpha=0.8)
    ax.set_title("Prompt-matched median multiplier vs K=8")
    ax.set_ylabel("Token ratio vs K=8 →")
    ax.set_ylim(*padded_limits(np.concatenate(all_ratios), 0.15, 0.0))
    ax.legend(frameon=False, fontsize=10)

    for ax in axes:
        ax.axvline(8, color="#8A8E96", linestyle="--", linewidth=1.1, alpha=0.7)
        ax.set_xlabel("Active experts per token (K)")
        ax.set_xticks(ks)
    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.79, wspace=0.30)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stem = args.output.with_suffix("")
    for suffix in ("png", "svg"):
        fig.savefig(
            stem.with_suffix(f".{suffix}"),
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

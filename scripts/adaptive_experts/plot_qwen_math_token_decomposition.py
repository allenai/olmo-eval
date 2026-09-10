#!/usr/bin/env python3
"""Plot the Qwen MATH-500 completion-token decomposition on common successes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open() as f:
        report = json.load(f)

    ks = np.asarray(sorted(int(k) for k in report["by_k"]), dtype=int)
    by_k = report["by_k"]
    routing = report.get("schema", {}).get("routing", "normalized")
    n_prompts = report["n_prompts"]

    def mean_values(field: str) -> np.ndarray:
        return np.asarray(
            [by_k[str(k)]["mean_of_prompt_means"][field] for k in ks], dtype=float
        )

    def ratio_values(field: str) -> np.ndarray:
        return np.asarray(
            [by_k[str(k)]["median_prompt_ratio_vs_k8"][field] for k in ks],
            dtype=float,
        )

    reasoning = mean_values("inferred_reasoning_and_boundary_tokens")
    returned = mean_values("returned_solution_tokens")

    sns.set_theme(style="whitegrid", context="talk")
    colors = sns.color_palette("colorblind")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    fig.suptitle(
        f"Qwen3-30B-A3B MATH-500 performance and token decomposition ({routing})",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.925,
        f"Performance uses all 500 prompts; token panels use {n_prompts} prompts correct in "
        "every available K=4–16 run",
        ha="center",
        color="#555A63",
        fontsize=11,
    )

    accuracy = report["math500_accuracy_all_prompts"]
    accuracy_mean = np.asarray([accuracy[str(k)]["mean_pct"] for k in ks])
    accuracy_min = np.asarray([accuracy[str(k)]["min_pct"] for k in ks])
    accuracy_max = np.asarray([accuracy[str(k)]["max_pct"] for k in ks])

    ax = axes[0]
    ax.fill_between(
        ks,
        accuracy_min,
        accuracy_max,
        color=colors[0],
        alpha=0.2,
        linewidth=0,
        label="Replicate min–max",
    )
    ax.plot(
        ks,
        accuracy_mean,
        marker="o",
        color=colors[0],
        linestyle=(0, (2, 2)),
        linewidth=2.4,
        label="Replicate mean",
    )
    ax.set_title("MATH-500 performance (all 500 prompts)")
    ax.set_ylabel("Accuracy (%) ↑")
    ax.set_ylim(92.5, 96.5)
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
        label="Parser-returned worked solution",
    )
    ax.set_title("Mean full-generation tokens")
    ax.set_ylabel("Tokens per response ↓")
    ax.set_ylim(0, 5700)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[2]
    ax.plot(
        ks,
        ratio_values("total_tokens"),
        marker="o",
        color=colors[5],
        linewidth=2.4,
        label="Full generation",
    )
    ax.plot(
        ks,
        ratio_values("inferred_reasoning_and_boundary_tokens"),
        marker="s",
        color=colors[0],
        linewidth=2.2,
        label="Inferred hidden reasoning",
    )
    ax.plot(
        ks,
        ratio_values("returned_solution_tokens"),
        marker="^",
        color=colors[2],
        linewidth=2.2,
        label="Returned worked solution",
    )
    ax.axhline(1, color="#8B8F97", linestyle=":", linewidth=1.5)
    ax.set_title("Prompt-matched median multiplier vs K=8")
    ax.set_ylabel("Token ratio vs K=8 ↓")
    ax.set_ylim(0.8, 1.42)
    ax.legend(frameon=False, fontsize=10)

    for ax in axes:
        ax.axvline(8, color="#8B8F97", linestyle="--", linewidth=1.3, alpha=0.8)
        ax.set_xlabel("Active experts per token (K)")
        ax.set_xticks(ks)
        ax.tick_params(axis="x", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()

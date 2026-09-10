#!/usr/bin/env python3
"""Plot Qwen MATH-500 completion/coherence diagnostics across active K."""

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

    ks = np.asarray(sorted(int(k) for k in report["summary"]), dtype=int)
    summary = report["summary"]
    cohorts = report["fixed_prompt_cohorts"]

    def values(field: str) -> np.ndarray:
        return np.asarray([summary[str(k)][field] for k in ks], dtype=float)

    sns.set_theme(style="whitegrid", context="talk")
    colors = sns.color_palette("colorblind")
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Qwen3-30B-A3B MATH-500 response behavior vs active experts",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "Three 500-example runs per K except K=13–15 (two runs); normalized routing; "
        "32,768-token cap",
        ha="center",
        color="#555A63",
        fontsize=11,
    )

    ax = axes[0, 0]
    ax.plot(ks, values("accuracy_pct"), marker="o", color=colors[0], linewidth=2.4)
    ax.set_title("MATH-500 accuracy")
    ax.set_ylabel("Accuracy (%) ↑")
    ax.set_ylim(-2, 102)

    ax = axes[0, 1]
    ax.plot(
        ks,
        values("generated_tokens_median"),
        marker="o",
        color=colors[2],
        linewidth=2.4,
        label="Median",
    )
    ax.plot(
        ks,
        values("generated_tokens_mean"),
        marker="s",
        linestyle="--",
        color=colors[1],
        linewidth=2,
        label="Mean",
    )
    ax.axhline(32768, color="#8B8F97", linestyle=":", linewidth=1.5, label="Generation cap")
    ax.set_title("Full generated length")
    ax.set_ylabel("Tokens ↓")
    ax.set_ylim(0, 34500)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1, 0]
    ax.plot(
        ks,
        values("empty_pct"),
        marker="o",
        color=colors[3],
        linewidth=2.4,
        label="Empty visible final",
    )
    ax.plot(
        ks,
        values("cap_hit_pct"),
        marker="s",
        color=colors[4],
        linewidth=2.4,
        label="32k cap hit",
    )
    ax.set_title("Generation/finalization failures")
    ax.set_ylabel("Responses (%) ↓")
    ax.set_ylim(-2, 102)
    ax.legend(frameon=False, fontsize=10)

    ax = axes[1, 1]
    ratio_ks = np.asarray([k for k in ks if k >= 3], dtype=int)
    cohort_styles = (
        ("all_prompts", "All 500 prompts", colors[5], "-", ratio_ks),
        ("k8_majority_correct", "Fixed K=8-solvable cohort", colors[0], "--", ratio_ks),
        (
            "common_all_correct_k4_k16",
            "Correct on every K=4–16 run",
            colors[2],
            ":",
            np.asarray([k for k in ratio_ks if 4 <= k <= 16], dtype=int),
        ),
    )
    all_ratios = []
    for cohort_name, label, color, linestyle, cohort_ks in cohort_styles:
        ratios = np.asarray(
            [
                cohorts[cohort_name]["by_k"][str(k)][
                    "generated_token_ratio_vs_k8_median"
                ]
                for k in cohort_ks
            ],
            dtype=float,
        )
        all_ratios.extend(ratios)
        ax.plot(
            cohort_ks,
            ratios,
            marker="o",
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=label,
        )
    ax.axhline(1, color="#8B8F97", linestyle=":", linewidth=1.5)
    ax.set_title("Prompt-matched token multiplier (fixed cohorts)")
    ax.set_ylabel("Median generated-token ratio vs K=8 ↓")
    ax.set_ylim(0, max(2.2, float(max(all_ratios)) + 0.1))
    ax.legend(frameon=False, fontsize=9)

    for ax in axes.flat:
        ax.axvline(8, color="#8B8F97", linestyle="--", linewidth=1.3, alpha=0.8)
        ax.set_xlabel("Active experts per token (K)")
        ax.set_xticks(ks)
        ax.tick_params(axis="x", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2, w_pad=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot inference-K MATH-500 performance for neutral mixed-K and fixed-K8 RL."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
PLOTS = NOTES / "plots"


def main() -> None:
    plateau = pd.read_csv(NOTES / "slime_qwen3_base_dapo_step350_plateau_summary.csv")
    mixed = pd.read_csv(NOTES / "slime_qwen3_base_dapo_mixed_k_checkpoint_evals.csv")
    fixed = pd.read_csv(NOTES / "slime_qwen3_base_dapo_checkpoint_evals.csv")

    fixed_350 = plateau[
        plateau["run"].eq("k8-normalized")
        & plateau["mode"].eq("reference")
        & plateau["k"].between(4, 8)
        & plateau["complete"]
    ].sort_values("k")
    neutral_350 = mixed[
        mixed["objective"].eq("neutral") & mixed["step"].eq(350)
    ].copy()
    neutral_100 = mixed[
        mixed["objective"].eq("neutral") & mixed["step"].eq(100)
    ].copy()
    routing_k = {"reference-k4": 4, "reference-k6": 6, "native-k8": 8}
    for frame in (neutral_350, neutral_100):
        frame["k"] = frame["routing"].map(routing_k)
        frame.sort_values("k", inplace=True)

    fixed_100_k8 = fixed[
        fixed["run"].eq("k8-normalized") & fixed["step"].eq(100)
    ].iloc[0]

    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    fixed_color = "#0072B2"
    neutral_color = "#D55E00"

    ax.plot(
        fixed_350["k"],
        100 * fixed_350["mean"],
        color=fixed_color,
        marker="o",
        linewidth=2.8,
        markersize=8,
        label="Fixed K=8 training · step 350",
    )
    ax.fill_between(
        fixed_350["k"],
        100 * fixed_350["min"],
        100 * fixed_350["max"],
        color=fixed_color,
        alpha=0.14,
        linewidth=0,
    )

    ax.plot(
        neutral_350["k"],
        100 * neutral_350["math500_accuracy"],
        color=neutral_color,
        marker="s",
        linewidth=2.8,
        markersize=8,
        label="Neutral mixed-K training · step 350",
    )
    ax.fill_between(
        neutral_350["k"],
        100 * neutral_350["math500_min"],
        100 * neutral_350["math500_max"],
        color=neutral_color,
        alpha=0.18,
        linewidth=0,
    )

    ax.plot(
        neutral_100["k"],
        100 * neutral_100["math500_accuracy"],
        color=neutral_color,
        marker="s",
        markerfacecolor="white",
        markeredgewidth=1.8,
        linestyle="--",
        linewidth=2.2,
        markersize=7.5,
        label="Neutral mixed-K training · step 100",
    )
    ax.fill_between(
        neutral_100["k"],
        100 * neutral_100["math500_min"],
        100 * neutral_100["math500_max"],
        color=neutral_color,
        alpha=0.08,
        linewidth=0,
    )

    ax.scatter(
        [8],
        [100 * fixed_100_k8["math500_accuracy"]],
        color=fixed_color,
        facecolors="white",
        marker="o",
        linewidths=2.0,
        s=78,
        zorder=5,
        label="Fixed K=8 training · step 100 (K=8 only)",
    )

    ax.set_title(
        "Qwen3-30B-A3B Base DAPO RL: inference-K MATH-500 plateau",
        fontsize=17,
        weight="bold",
        pad=30,
    )
    ax.text(
        0.5,
        1.025,
        "Reduced-K points are K=8-reference-scaled · lines show three-run means · shading shows min–max",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#555555",
    )
    ax.set_xlabel("Experts used at inference (K)", fontsize=12)
    ax.set_ylabel("MATH-500 accuracy (%)", fontsize=12)
    ax.set_xticks(range(4, 9))
    ax.set_xlim(3.8, 8.2)
    ax.set_ylim(73.5, 85.5)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = PLOTS / "slime_qwen3_base_dapo_neutral_vs_k8_step100_350_inference_plateau"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

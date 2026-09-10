#!/usr/bin/env python3
"""Compare neutral mixed-K DAPO RL with matched fixed-K training arms."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "notes"
PLOTS = NOTES / "plots"

FIXED_CSV = NOTES / "slime_qwen3_base_dapo_checkpoint_evals.csv"
MIXED_CSV = NOTES / "slime_qwen3_base_dapo_mixed_k_checkpoint_evals.csv"

CONDITIONS = (
    ("reference-k4", "k4-reference-k8", "K=4 reference-scaled", "#009E73"),
    ("reference-k6", "k6-reference-k8", "K=6 reference-scaled", "#E69F00"),
    ("native-k8", "k8-normalized", "K=8 native", "#0072B2"),
)


def main() -> None:
    fixed = pd.read_csv(FIXED_CSV)
    mixed = pd.read_csv(MIXED_CSV)
    mixed = mixed[mixed["objective"].eq("neutral")]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.0), sharex="col", sharey="row")
    for col, (routing, fixed_run, title, color) in enumerate(CONDITIONS):
        mixed_rows = mixed[mixed["routing"].eq(routing)].sort_values("step")
        fixed_rows = fixed[fixed["run"].eq(fixed_run)].sort_values("step")

        math_ax = axes[0, col]
        math_ax.plot(
            fixed_rows["step"],
            100 * fixed_rows["math500_accuracy"],
            color=color,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            linewidth=2.3,
            label="Fixed-K training",
        )
        math_ax.fill_between(
            fixed_rows["step"],
            100 * (fixed_rows["math500_accuracy"] - fixed_rows["math500_accuracy_stddev"]),
            100 * (fixed_rows["math500_accuracy"] + fixed_rows["math500_accuracy_stddev"]),
            color=color,
            alpha=0.10,
            linewidth=0,
        )
        math_ax.plot(
            mixed_rows["step"],
            100 * mixed_rows["math500_accuracy"],
            color=color,
            linestyle="-",
            marker="s",
            linewidth=2.6,
            label="Neutral mixed-K training",
        )
        math_ax.fill_between(
            mixed_rows["step"],
            100 * (mixed_rows["math500_accuracy"] - mixed_rows["math500_stddev"]),
            100 * (mixed_rows["math500_accuracy"] + mixed_rows["math500_stddev"]),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        math_ax.set_title(title, fontsize=14, weight="bold")
        math_ax.set_ylabel("MATH-500 accuracy (%)" if col == 0 else "")

        aime_ax = axes[1, col]
        aime_ax.plot(
            fixed_rows["step"],
            100 * fixed_rows["aime_pass_at_1"],
            color=color,
            linestyle="--",
            marker="o",
            markerfacecolor="white",
            linewidth=2.3,
            label="Fixed-K training",
        )
        aime_ax.plot(
            mixed_rows["step"],
            100 * mixed_rows["aime_pass_at_1"],
            color=color,
            linestyle="-",
            marker="s",
            linewidth=2.6,
            label="Neutral mixed-K training",
        )
        aime_ax.set_ylabel("AIME 2025 Pass@1 (%)" if col == 0 else "")
        aime_ax.set_xlabel("RL training step")

        for ax in (math_ax, aime_ax):
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle(
        "Qwen3-30B-A3B Base DAPO RL: neutral mixed-K vs fixed-K training",
        fontsize=20,
        weight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "Each column uses the same hard-coded inference K and routing scale; MATH lines show three-run means",
        ha="center",
        va="top",
        fontsize=11.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.99, 0.90), h_pad=2.0, w_pad=1.5)

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = PLOTS / "slime_qwen3_base_dapo_neutral_mixed_vs_fixed_k"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

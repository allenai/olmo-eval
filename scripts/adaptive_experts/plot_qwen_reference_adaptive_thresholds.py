#!/usr/bin/env python3
"""Plot Qwen reference-preserving adaptive-threshold performance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "notes" / "qwen_reference_adaptive_threshold_results.json"
DEFAULT_OUTPUT = ROOT / "notes" / "plots" / "qwen3_reference_adaptive_tau_performance"

SERIES = (
    ("math500", "MATH-500", "#0072B2", "o", "-"),
    ("gpqa", "GPQA Diamond", "#E69F00", "s", "-"),
    ("ifbench", "IFBench macro (32k)", "#009E73", "^", "-"),
    ("humaneval", "HumanEval pass@1", "#D55E00", "D", "-"),
    ("macro", "Four-eval macro", "#30343B", "P", "--"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.input.open() as f:
        payload = json.load(f)
    points = sorted((float(tau), values) for tau, values in payload["by_threshold"].items())
    taus = np.asarray([tau for tau, _ in points])
    mean_ks = np.asarray([values["realized_mean_k"]["mean"] for _, values in points])
    sample_sizes = [values["n"] for _, values in points]

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13.6, 7.6))
    for key, label, color, marker, linestyle in SERIES:
        means = np.asarray([values[key]["mean"] for _, values in points])
        sds = np.asarray([values[key]["sample_sd"] or 0 for _, values in points])
        linewidth = 2.8 if key == "macro" else 2.2
        ax.fill_between(
            taus,
            np.maximum(0, means - sds),
            np.minimum(100, means + sds),
            color=color,
            alpha=0.11 if key == "macro" else 0.14,
            linewidth=0,
        )
        ax.plot(
            taus,
            means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=8,
            label=label,
        )

    fig.suptitle(
        "Qwen3-30B-A3B: reference-preserving adaptive expert thresholds",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Retained top-eight weights are not renormalized; shading is ±1 sample SD; "
        "top labels show realized mean K; n=2 at τ=0.50 and n=3 otherwise",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#555A63",
    )
    ax.set_xlabel("Cumulative selected-mass threshold (τ)", fontweight="semibold", labelpad=10)
    ax.set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    ax.set_xticks(taus)
    ax.set_xticklabels([f"{tau:.2f}" for tau in taus])
    ax.set_xlim(taus.min() - 0.018, taus.max() + 0.018)
    ax.set_ylim(45, 100)
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.7)

    top = ax.secondary_xaxis("top", functions=(lambda x: x, lambda x: x))
    top.set_xticks(taus)
    top.set_xticklabels([f"K̄={value:.2f}" for value in mean_ks], fontsize=10)

    for tau, values, n in zip(taus, (value for _, value in points), sample_sizes, strict=True):
        macro = values["macro"]["mean"]
        ax.annotate(
            f"{macro:.1f}",
            (tau, macro),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=9.5,
            color="#30343B",
        )
        if n < 3:
            ax.annotate(
                "n=2",
                (tau, 46.2),
                ha="center",
                fontsize=9,
                color="#777B82",
            )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        frameon=True,
        framealpha=0.94,
        fontsize=10.5,
        borderaxespad=0,
    )
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.09, right=0.77, bottom=0.13, top=0.86)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        target = args.output.with_suffix(f".{suffix}")
        fig.savefig(
            target,
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

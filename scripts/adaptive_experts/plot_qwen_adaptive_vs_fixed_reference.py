#!/usr/bin/env python3
"""Compare fixed-K and adaptive-mass Qwen reference-preserving inference."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ADAPTIVE = ROOT / "notes" / "qwen_reference_adaptive_threshold_results.json"
DEFAULT_FIXED = ROOT / "notes" / "current_suite_expert_sweeps.csv"
DEFAULT_OUTPUT = ROOT / "notes" / "plots" / "qwen3_reference_adaptive_vs_fixed_k"

MODEL = "Qwen3-30B-A3B hybrid thinking"
ROUTING = "reference_scaled"
METRICS = (
    ("math500", "MATH-500 ↑", "MATH-500", (90, 97)),
    ("gpqa", "GPQA Diamond ↑", "GPQA Diamond", (47, 68)),
    ("ifbench", "IFBench macro (32k) ↑", "IFBench macro (32k)", (46, 60)),
    ("humaneval", "HumanEval pass@1 ↑", "HumanEval pass@1", (70, 97)),
)


def load_fixed(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    values: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["model"] != MODEL or row["routing"] != ROUTING:
                continue
            k = int(row["k"])
            if 3 <= k <= 8:
                values[row["metric"]][k] = (float(row["mean"]), float(row["sd"]))
    for _, fixed_metric, _, _ in METRICS:
        if sorted(values[fixed_metric]) != list(range(3, 9)):
            raise RuntimeError(f"Missing fixed-K values for {fixed_metric}: {values[fixed_metric]}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", type=Path, default=DEFAULT_ADAPTIVE)
    parser.add_argument("--fixed", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.adaptive.open() as f:
        adaptive = json.load(f)["by_threshold"]
    fixed = load_fixed(args.fixed)

    fixed_ks = np.arange(3, 9, dtype=float)
    thresholds = sorted((float(tau), data) for tau, data in adaptive.items())
    adaptive_ks = np.asarray([data["realized_mean_k"]["mean"] for _, data in thresholds])

    fixed_color = "#0072B2"
    adaptive_color = "#D55E00"
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.7), sharex=True)

    for ax, (adaptive_metric, fixed_metric, title, ylim) in zip(
        axes.flat, METRICS, strict=True
    ):
        fixed_means = np.asarray([fixed[fixed_metric][int(k)][0] for k in fixed_ks])
        fixed_sds = np.asarray([fixed[fixed_metric][int(k)][1] for k in fixed_ks])
        adaptive_means = np.asarray([data[adaptive_metric]["mean"] for _, data in thresholds])
        adaptive_sds = np.asarray(
            [data[adaptive_metric]["sample_sd"] or 0 for _, data in thresholds]
        )

        ax.fill_between(
            fixed_ks,
            fixed_means - fixed_sds,
            fixed_means + fixed_sds,
            color=fixed_color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            fixed_ks,
            fixed_means,
            color=fixed_color,
            marker="o",
            linewidth=2.4,
            markersize=6.5,
        )
        ax.fill_between(
            adaptive_ks,
            adaptive_means - adaptive_sds,
            adaptive_means + adaptive_sds,
            color=adaptive_color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            adaptive_ks,
            adaptive_means,
            color=adaptive_color,
            marker="D",
            linestyle="--",
            linewidth=2.4,
            markersize=7,
        )
        for index, ((tau, _), x, y) in enumerate(
            zip(thresholds, adaptive_ks, adaptive_means, strict=True)
        ):
            offset = (0, 10) if index % 2 == 0 else (0, -16)
            ax.annotate(
                f"τ={tau:.1f}",
                (x, y),
                xytext=offset,
                textcoords="offset points",
                ha="center",
                fontsize=8.8,
                color=adaptive_color,
            )

        ax.axvline(8, color="#777B82", linestyle=":", linewidth=1.1, alpha=0.75)
        ax.set_title(title, fontsize=14, fontweight="semibold", pad=8)
        ax.set_xlim(2.8, 8.18)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
        ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.7)

    fig.suptitle(
        "Qwen3-30B-A3B: adaptive mass versus strict expert cutoffs",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "Both preserve native top-eight weights without renormalizing; adaptive x-values are "
        "realized mean K; shading is ±1 sample SD",
        ha="center",
        fontsize=10.5,
        color="#555A63",
    )
    fig.supxlabel(
        "Active experts per token-layer (fixed K or adaptive realized mean K)",
        fontsize=13,
        fontweight="semibold",
        y=0.035,
    )
    fig.supylabel("Accuracy / score (%)  ↑", fontsize=13, fontweight="semibold", x=0.035)
    handles = (
        Line2D(
            [0],
            [0],
            color=fixed_color,
            marker="o",
            linewidth=2.4,
            label="Strict fixed-K cutoff",
        ),
        Line2D(
            [0],
            [0],
            color=adaptive_color,
            marker="D",
            linestyle="--",
            linewidth=2.4,
            label="Adaptive mass threshold",
        ),
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
        frameon=False,
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.095, top=0.855, hspace=0.25, wspace=0.18)

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

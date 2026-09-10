#!/usr/bin/env python3
"""Plot completed OpenThoughts-Agent SFT TBLite inference-K results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "notes" / "openthoughts_all_sft_tblite_results.csv"
OUT = REPO / "notes" / "plots"

ORDER = ("fixed_k4", "fixed_k8", "fixed_k12", "mixed_k4_k8_k12", "phased_k12_k8_k4")
COLORS = {
    "fixed_k4": "#0072B2",
    "fixed_k8": "#D55E00",
    "fixed_k12": "#009E73",
    "mixed_k4_k8_k12": "#7A3E9D",
    "phased_k12_k8_k4": "#CC79A7",
}
MARKERS = {
    "fixed_k4": "o",
    "fixed_k8": "s",
    "fixed_k12": "^",
    "mixed_k4_k8_k12": "D",
    "phased_k12_k8_k4": "P",
}


def read_rows(min_k: int | None = None, max_k: int | None = None) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with DATA.open(newline="") as f:
        for raw in csv.DictReader(f):
            eval_k = int(raw["eval_k"])
            if min_k is not None and eval_k < min_k:
                continue
            if max_k is not None and eval_k > max_k:
                continue
            grouped[raw["model_id"]].append(
                {
                    "label": raw["label"],
                    "eval_k": eval_k,
                    "mean_reward": 100 * float(raw["mean_reward"]),
                    "sd_reward": 100 * float(raw["sd_reward"]),
                }
            )
    for rows in grouped.values():
        rows.sort(key=lambda row: row["eval_k"])
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-k", type=int)
    parser.add_argument("--max-k", type=int)
    parser.add_argument(
        "--output-stem",
        default="qwen35_openthoughts_sft_tblite_expert_sweep",
    )
    args = parser.parse_args()

    grouped = read_rows(args.min_k, args.max_k)
    fig, ax = plt.subplots(figsize=(11.2, 7.0), constrained_layout=True)

    for model_id in ORDER:
        rows = grouped[model_id]
        x = [row["eval_k"] for row in rows]
        y = [row["mean_reward"] for row in rows]
        sd = [row["sd_reward"] for row in rows]
        lower = [mean - spread for mean, spread in zip(y, sd)]
        upper = [mean + spread for mean, spread in zip(y, sd)]
        is_adaptive = model_id in {"mixed_k4_k8_k12", "phased_k12_k8_k4"}

        ax.fill_between(x, lower, upper, color=COLORS[model_id], alpha=0.13, linewidth=0)
        ax.plot(
            x,
            y,
            color=COLORS[model_id],
            marker=MARKERS[model_id],
            markersize=8 if is_adaptive else 7,
            linewidth=3.2 if is_adaptive else 2.4,
            linestyle="-" if is_adaptive else "--",
            label=str(rows[0]["label"]),
            zorder=3 if is_adaptive else 2,
        )

    all_x = sorted({int(row["eval_k"]) for rows in grouped.values() for row in rows})
    ax.set_xticks(all_x)
    ax.set_xlim(min(all_x) - 0.6, max(all_x) + 0.6)
    ax.set_ylim(24, 50)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Active experts at evaluation (K)", fontsize=13)
    ax.set_ylabel("TBLite mean reward", fontsize=13)
    ax.set_title(
        "Qwen3.5-35B-A3B OpenThoughts-Agent SFT · TBLite",
        fontsize=17,
        fontweight="bold",
        pad=14,
    )
    ax.text(
        0.5,
        1.01,
        "K=8 reference scaling without renormalization · mean ± sample SD over 3 runs",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4C4C4C",
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.9, alpha=0.8)
    ax.grid(axis="x", color="#EEEEEE", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)
    ax.legend(
        title="SFT routing during training",
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        fontsize=10.5,
        title_fontsize=10.5,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUT / f"{args.output_stem}.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()

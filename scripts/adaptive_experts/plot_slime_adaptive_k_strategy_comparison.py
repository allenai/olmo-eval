#!/usr/bin/env python3
"""Compare adaptive-K DAPO strategies at matched inference K."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
OLD_CSV = NOTES / "slime_qwen3_base_dapo_mixed_k_checkpoint_evals.csv"
NEW_CSV = NOTES / "slime_qwen3_base_dapo_adaptive_k_checkpoint_evals.csv"
FIXED_CSV = NOTES / "slime_qwen3_base_dapo_checkpoint_evals.csv"

ROUTINGS = ("reference-k4", "native-k8")
ROUTING_LABELS = {
    "reference-k4": "Inference K=4 reference-scaled",
    "native-k8": "Inference K=8 native",
}
STRATEGIES = ("fixed-k4", "fixed-k8", "even-k4-k6-k8", "preallocated", "even-k4-k8-k12")
STRATEGY_STYLES = {
    "fixed-k4": {
        "label": "Fixed train K=4 (baseline)",
        "color": "#6B7280",
        "marker": "x",
        "linestyle": "--",
    },
    "fixed-k8": {
        "label": "Fixed train K=8 (baseline)",
        "color": "#6B7280",
        "marker": "x",
        "linestyle": "--",
    },
    "even-k4-k6-k8": {
        "label": "Previous even train K=4/6/8",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "preallocated": {
        "label": "Preallocated K per prompt",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "-",
    },
    "even-k4-k8-k12": {
        "label": "Even train K=4/8/12",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-",
    },
}


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with OLD_CSV.open() as handle:
        for row in csv.DictReader(handle):
            if row["objective"] != "neutral" or row["routing"] not in ROUTINGS:
                continue
            rows.append(parse_row(row, "even-k4-k6-k8"))

    with NEW_CSV.open() as handle:
        for row in csv.DictReader(handle):
            if row["routing"] not in ROUTINGS:
                continue
            strategy = {
                "preallocated": "preallocated",
                "mixed-k4-k8-k12": "even-k4-k8-k12",
            }[row["run"]]
            rows.append(parse_row(row, strategy))

    # Each original fixed-K arm is shown only under its matched inference routing.
    # We do not have checkpoint-by-checkpoint cross-routing evaluations for these
    # two baselines.
    with FIXED_CSV.open() as handle:
        for row in csv.DictReader(handle):
            fixed = {
                "k4-reference-k8": ("fixed-k4", "reference-k4"),
                "k8-normalized": ("fixed-k8", "native-k8"),
            }.get(row["run"])
            if fixed is None:
                continue
            strategy, routing = fixed
            rows.append(
                {
                    "strategy": strategy,
                    "routing": routing,
                    "step": int(row["step"]),
                    "math": 100 * float(row["math500_accuracy"]),
                    "math_min": 100 * float(row["math500_accuracy_min"]),
                    "math_max": 100 * float(row["math500_accuracy_max"]),
                    "aime": 100 * float(row["aime_pass_at_1"]),
                }
            )
    return rows


def optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def parse_row(row: dict[str, str], strategy: str) -> dict[str, object]:
    return {
        "strategy": strategy,
        "routing": row["routing"],
        "step": int(row["step"]),
        "math": 100 * float(row["math500_accuracy"]),
        "math_min": 100 * float(row["math500_min"]),
        "math_max": 100 * float(row["math500_max"]),
        "aime": None
        if optional_float(row["aime_pass_at_1"]) is None
        else 100 * float(row["aime_pass_at_1"]),
    }


def style_axis(ax: plt.Axes, *, ylabel: str, show_xlabel: bool) -> None:
    if show_xlabel:
        ax.set_xlabel("RL training step", fontweight="semibold")
    ax.set_ylabel(ylabel, fontweight="semibold")
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    rows = load_rows()
    all_steps = sorted({int(row["step"]) for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.4), sharex=True, sharey="row")

    for column, routing in enumerate(ROUTINGS):
        math_ax = axes[0, column]
        aime_ax = axes[1, column]
        for strategy in STRATEGIES:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["strategy"] == strategy and row["routing"] == routing
                ],
                key=lambda row: int(row["step"]),
            )
            if not selected:
                continue
            style = STRATEGY_STYLES[strategy]
            x = [int(row["step"]) for row in selected]
            math = [float(row["math"]) for row in selected]
            math_ax.fill_between(
                x,
                [float(row["math_min"]) for row in selected],
                [float(row["math_max"]) for row in selected],
                color=style["color"],
                alpha=0.11,
                linewidth=0,
            )
            math_ax.plot(
                x,
                math,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.35,
                markersize=6.5,
            )

            aime_rows = [row for row in selected if row["aime"] is not None]
            aime_ax.plot(
                [int(row["step"]) for row in aime_rows],
                [float(row["aime"]) for row in aime_rows],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.35,
                markersize=6.5,
            )

        math_ax.set_title(ROUTING_LABELS[routing], fontsize=13, fontweight="semibold")
        style_axis(math_ax, ylabel="MATH-500 accuracy (%)  ↑", show_xlabel=False)
        style_axis(aime_ax, ylabel="AIME 2025 Pass@1 (%)  ↑", show_xlabel=True)
        aime_ax.set_xticks(all_steps)

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Qwen3-30B-A3B Base · Adaptive-K DAPO Training Strategy Comparison",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.945,
        "Matched inference routing · MATH line = 3-run mean, shading = observed min–max · AIME = one pass@32 evaluation",
        ha="center",
        color="#555A63",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.0, w_pad=2.2)

    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = PLOTS / "slime_qwen3_base_dapo_adaptive_k_strategy_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {stem}.png and {stem}.svg from {len(rows)} matched rows")


if __name__ == "__main__":
    main()

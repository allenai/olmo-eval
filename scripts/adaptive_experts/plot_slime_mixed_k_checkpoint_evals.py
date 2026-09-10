#!/usr/bin/env python3
"""Plot the two mixed-K DAPO runs across checkpoints and inference K."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = OLMO_EVAL_ROOT.parent
RESULTS = PROJECT_ROOT / "results" / "rl_checkpoint_evals"
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
CSV_PATH = NOTES / "slime_qwen3_base_dapo_mixed_k_checkpoint_evals.csv"

ORIGINAL_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-mixed-(?P<objective>cost|neutral)-"
    r"(?P<routing>reference-k4|reference-k6|native-k8)-step(?P<step>\d+)-"
    r"math-(?:eval|aime)-v\d+-\d{8}$"
)
REPLICATE_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-mixed-(?P<objective>cost|neutral)-"
    r"(?P<routing>reference-k4|reference-k6|native-k8)-step(?P<step>\d+)-"
    r"math500-rep(?P<replicate>[23])-seed(?P<seed>\d+)-v\d+-\d{8}$"
)

OBJECTIVES = ("cost", "neutral")
OBJECTIVE_LABELS = {
    "cost": "Cost-shaped mixed-K RL",
    "neutral": "Neutral mixed-K RL",
}
ROUTINGS = ("reference-k4", "reference-k6", "native-k8")
ROUTING_STYLES = {
    "reference-k4": {"label": "Eval K=4 reference-scaled", "color": "#009E73", "marker": "^"},
    "reference-k6": {"label": "Eval K=6 reference-scaled", "color": "#E69F00", "marker": "s"},
    "native-k8": {"label": "Eval K=8 native", "color": "#0072B2", "marker": "o"},
}


def task(payload: dict, name: str) -> dict | None:
    matches = [item for item in payload.get("tasks", []) if item.get("task") == name]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {name!r} tasks in one metrics file")
    return matches[0] if matches else None


def load_rows() -> list[dict[str, object]]:
    math_values: dict[tuple[str, str, int], dict[int, tuple[float, str]]] = defaultdict(dict)
    aime_values: dict[tuple[str, str, int], tuple[float, float, str]] = {}

    for metrics_path in sorted(RESULTS.glob("*/**/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        if payload.get("errors"):
            continue
        name = payload.get("config", {}).get("metrics", {}).get("experiment_name", "")
        original = ORIGINAL_PATTERN.fullmatch(name)
        replicate = REPLICATE_PATTERN.fullmatch(name)
        if original is None and replicate is None:
            continue

        match = original or replicate
        assert match is not None
        key = (match.group("objective"), match.group("routing"), int(match.group("step")))
        experiment_id = metrics_path.relative_to(RESULTS).parts[0]
        math = task(payload, "math500:chat")
        if math is None or math.get("num_instances") != 500:
            raise RuntimeError(f"Missing or incomplete MATH-500 task in {metrics_path}")
        math_accuracy = float(math["metrics"]["accuracy"]["minerva_math_flex"])

        replicate_number = 1 if original is not None else int(match.group("replicate"))
        if replicate_number in math_values[key]:
            previous_id = math_values[key][replicate_number][1]
            raise RuntimeError(
                f"Duplicate MATH replicate {replicate_number} for {key}: "
                f"{previous_id}, {experiment_id}"
            )
        math_values[key][replicate_number] = (math_accuracy, experiment_id)

        if original is not None:
            aime = task(payload, "aime_2025:pass_at_32")
            if aime is None or aime.get("num_instances") != 30:
                raise RuntimeError(f"Missing or incomplete AIME task in {metrics_path}")
            metrics = aime["metrics"]
            aime_values[key] = (
                float(metrics["pass_at_1"]["minerva_math_flex"]),
                float(metrics["pass_at_8"]["minerva_math_flex"]),
                experiment_id,
            )

    rows: list[dict[str, object]] = []
    for key in sorted(math_values):
        objective, routing, step = key
        replicates = math_values[key]
        values = [value for value, _ in replicates.values()]
        aime = aime_values.get(key)
        rows.append(
            {
                "objective": objective,
                "routing": routing,
                "step": step,
                "math500_accuracy": statistics.fmean(values),
                "math500_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "math500_min": min(values),
                "math500_max": max(values),
                "math500_replicate_count": len(values),
                "math500_replicate_1": replicates.get(1, (None, None))[0],
                "math500_replicate_2": replicates.get(2, (None, None))[0],
                "math500_replicate_3": replicates.get(3, (None, None))[0],
                "aime_pass_at_1": None if aime is None else aime[0],
                "aime_pass_at_8": None if aime is None else aime[1],
                "aime_experiment_id": None if aime is None else aime[2],
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes, steps: list[int], ylabel: str) -> None:
    ax.set_xlabel("RL training step", fontweight="semibold")
    ax.set_ylabel(ylabel, fontweight="semibold")
    ax.set_xticks(steps)
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_math(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharey=True)
    all_steps = sorted({int(row["step"]) for row in rows})
    for ax, objective in zip(axes, OBJECTIVES, strict=True):
        objective_rows = [row for row in rows if row["objective"] == objective]
        for routing in ROUTINGS:
            routing_rows = sorted(
                [row for row in objective_rows if row["routing"] == routing],
                key=lambda row: int(row["step"]),
            )
            if not routing_rows:
                continue
            style = ROUTING_STYLES[routing]
            x = [int(row["step"]) for row in routing_rows]
            y = [100 * float(row["math500_accuracy"]) for row in routing_rows]
            low = [100 * float(row["math500_min"]) for row in routing_rows]
            high = [100 * float(row["math500_max"]) for row in routing_rows]
            ax.fill_between(x, low, high, color=style["color"], alpha=0.13, linewidth=0)
            ax.plot(
                x,
                y,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.4,
                markersize=7,
            )
        ax.set_title(OBJECTIVE_LABELS[objective], fontsize=13, fontweight="semibold")
        style_axes(ax, all_steps, "MATH-500 accuracy (%)  ↑")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Qwen3-30B-A3B Base · Mixed-K DAPO RL · MATH-500", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Line: mean of available seeds · shading: observed min–max (normally 3 evaluations per point)",
        ha="center",
        color="#555A63",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "slime_qwen3_base_dapo_mixed_k_math500_by_step")


def plot_aime(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharey=True)
    plotted_rows = [row for row in rows if row["aime_pass_at_1"] is not None]
    all_steps = sorted({int(row["step"]) for row in plotted_rows})
    for ax, objective in zip(axes, OBJECTIVES, strict=True):
        objective_rows = [row for row in plotted_rows if row["objective"] == objective]
        for routing in ROUTINGS:
            routing_rows = sorted(
                [row for row in objective_rows if row["routing"] == routing],
                key=lambda row: int(row["step"]),
            )
            if not routing_rows:
                continue
            style = ROUTING_STYLES[routing]
            ax.plot(
                [int(row["step"]) for row in routing_rows],
                [100 * float(row["aime_pass_at_1"]) for row in routing_rows],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.4,
                markersize=7,
            )
        ax.set_title(OBJECTIVE_LABELS[objective], fontsize=13, fontweight="semibold")
        style_axes(ax, all_steps, "AIME 2025 Pass@1 (%)  ↑")
    axes[1].legend(loc="best", frameon=False)
    fig.suptitle("Qwen3-30B-A3B Base · Mixed-K DAPO RL · AIME 2025", fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.925,
        "Pass@1 estimate from 32 sampled responses per problem · one evaluation per checkpoint",
        ha="center",
        color="#555A63",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "slime_qwen3_base_dapo_mixed_k_aime_by_step")


def save(fig: plt.Figure, stem: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            PLOTS / f"{stem}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if not rows:
        raise RuntimeError("No mixed-K checkpoint evaluations found")
    write_csv(rows)
    plot_math(rows)
    plot_aime(rows)
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()

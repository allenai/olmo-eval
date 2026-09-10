#!/usr/bin/env python3
"""Collect and plot the two K=10 DAPO pilot checkpoint evaluations."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = OLMO_EVAL_ROOT.parent
RESULTS = PROJECT_ROOT / "results" / "rl_checkpoint_evals"
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
CSV_PATH = NOTES / "slime_qwen3_base_dapo_k10_checkpoint_evals.csv"

NAME_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-k10-(?P<mode>normalized|reference-k8)-"
    r"step(?P<step>\d+)-k10-math-aime-(?:v1|retry1)-20260810$"
)
MODES = ("normalized", "reference-k8")
STYLES = {
    "normalized": {"label": "K=10 normalized", "color": "#0072B2", "marker": "o"},
    "reference-k8": {
        "label": "K=10 reference-scaled to K=8",
        "color": "#D55E00",
        "marker": "s",
    },
}


def task(payload: dict, name: str) -> dict:
    matches = [item for item in payload.get("tasks", []) if item.get("task") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name!r} task, found {len(matches)}")
    return matches[0]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for metrics_path in sorted(RESULTS.glob("*/**/metrics.json")):
        payload = json.loads(metrics_path.read_text())
        name = payload.get("config", {}).get("metrics", {}).get("experiment_name", "")
        match = NAME_PATTERN.fullmatch(name)
        if match is None:
            continue
        if payload.get("errors"):
            raise RuntimeError(f"Metric errors in {metrics_path}: {payload['errors']}")

        mode = match.group("mode")
        step = int(match.group("step"))
        key = (mode, step)
        if key in seen:
            raise RuntimeError(f"Duplicate completed evaluation for {key}")
        seen.add(key)

        math = task(payload, "math500:chat")
        aime = task(payload, "aime_2025:pass_at_32")
        if math.get("num_instances") != 500 or aime.get("num_instances") != 30:
            raise RuntimeError(f"Incomplete evaluation in {metrics_path}")
        aime_metrics = aime["metrics"]
        rows.append(
            {
                "mode": mode,
                "step": step,
                "math500_accuracy": math["metrics"]["accuracy"]["minerva_math_flex"],
                "aime_pass_at_1": aime_metrics["pass_at_1"]["minerva_math_flex"],
                "aime_pass_at_8": aime_metrics["pass_at_8"]["minerva_math_flex"],
                "experiment_id": metrics_path.relative_to(RESULTS).parts[0],
                "experiment_name": name,
            }
        )

    expected = {(mode, step) for mode in MODES for step in (20, 40, 60, 80, 100)}
    if seen != expected:
        raise RuntimeError(f"K=10 result mismatch: missing={sorted(expected - seen)}, extra={sorted(seen - expected)}")
    return sorted(rows, key=lambda row: (MODES.index(str(row["mode"])), int(row["step"])))


def write_csv(rows: list[dict[str, object]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("RL training step", fontweight="semibold")
    ax.set_ylabel(ylabel, fontweight="semibold")
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot_metric(
    rows: list[dict[str, object]], metric: str, ylabel: str, title: str, subtitle: str, stem: str
) -> None:
    fig, ax = plt.subplots(figsize=(9.3, 6.2))
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        style = STYLES[mode]
        ax.plot(
            [int(row["step"]) for row in selected],
            [100 * float(row[metric]) for row in selected],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.5,
            markersize=7,
        )
    style_axes(ax, ylabel)
    ax.legend(frameon=False, loc="best")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.92,
        subtitle,
        ha="center",
        color="#555A63",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(PLOTS / f"{stem}.{suffix}", dpi=220 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    plot_metric(
        rows,
        "math500_accuracy",
        "MATH-500 accuracy (%)  ↑",
        "Qwen3-30B-A3B Base · K=10 DAPO RL · MATH-500",
        "One seeded evaluation per checkpoint · 500 problems",
        "slime_qwen3_base_dapo_k10_math500_by_step",
    )
    plot_metric(
        rows,
        "aime_pass_at_1",
        "AIME 2025 Pass@1 (%)  ↑",
        "Qwen3-30B-A3B Base · K=10 DAPO RL · AIME 2025",
        "One seeded evaluation per checkpoint · 32 sampled responses per problem",
        "slime_qwen3_base_dapo_k10_aime_pass1_by_step",
    )
    plot_metric(
        rows,
        "aime_pass_at_8",
        "AIME 2025 Pass@8 (%)  ↑",
        "Qwen3-30B-A3B Base · K=10 DAPO RL · AIME 2025 Pass@8",
        "One seeded evaluation per checkpoint · 32 sampled responses per problem",
        "slime_qwen3_base_dapo_k10_aime_pass8_by_step",
    )
    print(f"Wrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()

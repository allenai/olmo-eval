#!/usr/bin/env python3
"""Plot the five-run Qwen3.6 Terminal-Bench 2.0 expert-count sweep."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
RESULTS = Path("/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b")
CORRECTED_RESULTS = Path(
    "/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-35b-a3b-kgt8-nestedfix1-20260815"
)
CSV_PATH = NOTES / "qwen36_terminalbench_full_suite_summary.csv"

NAME_PATTERN = re.compile(
    r"^qwen36-tb2-k(?P<k>4|6|8|10|12)-(?P<routing>reference|native)-full89-"
    r"rep(?P<replicate>[1-5])-qwen(?:-seed\d+)?-tp2-dp4-262k-c4-"
    r"shard-(?P<shard>[ab])-\d{8}$"
)

# Historical K=10/12 commands set a nonexistent outer config field and
# silently remained at native K=8. They are excluded until corrected runs are
# available.
EXPERT_COUNTS = (4, 6, 8, 10, 12)

LABELS = {
    4: "K=4\nreference-scaled",
    6: "K=6\nreference-scaled",
    8: "K=8\nnative",
    10: "K=10\nreference-scaled",
    12: "K=12\nreference-scaled",
}


def load_replicates() -> dict[int, list[float]]:
    shards: dict[tuple[int, int], dict[str, dict]] = defaultdict(dict)
    directories = [path for path in RESULTS.iterdir() if path.is_dir()]
    if CORRECTED_RESULTS.exists():
        directories.extend(path for path in CORRECTED_RESULTS.iterdir() if path.is_dir())
    for directory in sorted(directories):
        match = NAME_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        metrics_path = directory / "metrics.json"
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text())
        k = int(match.group("k"))
        if k > 8 and directory.parent != CORRECTED_RESULTS:
            continue
        if k <= 8 and directory.parent != RESULTS:
            continue
        replicate = int(match.group("replicate"))
        shard = match.group("shard")
        if shard in shards[k, replicate]:
            raise RuntimeError(f"Duplicate shard {shard} for K={k}, replicate {replicate}")
        shards[k, replicate][shard] = payload

    values: dict[int, list[float]] = defaultdict(list)
    for k in EXPERT_COUNTS:
        complete = all(
            set(shards.get((k, replicate), {})) == {"a", "b"}
            for replicate in range(1, 6)
        )
        if not complete:
            if k <= 8:
                raise RuntimeError(f"Missing a historical full-suite replicate for K={k}")
            # Never put an in-progress corrected condition on the maintained
            # plot. It becomes eligible only after all five A/B pairs finish.
            continue
        for replicate in range(1, 6):
            pair = shards.get((k, replicate), {})
            if set(pair) != {"a", "b"}:
                raise RuntimeError(
                    f"Expected shards a/b for K={k}, replicate {replicate}; found {sorted(pair)}"
                )
            task_names = [set(pair[shard]["per_task"]) for shard in ("a", "b")]
            if task_names[0] & task_names[1]:
                raise RuntimeError(f"Overlapping tasks for K={k}, replicate {replicate}")
            n_tasks = sum(int(pair[shard]["n_tasks"]) for shard in ("a", "b"))
            if n_tasks != 89 or len(task_names[0] | task_names[1]) != 89:
                raise RuntimeError(f"Incomplete suite for K={k}, replicate {replicate}: {n_tasks}")
            correct = sum(
                int(task["n_correct"])
                for shard in ("a", "b")
                for task in pair[shard]["per_task"].values()
            )
            values[k].append(correct / n_tasks)
    return dict(values)


def write_csv(values: dict[int, list[float]]) -> None:
    fieldnames = [
        "expert_count",
        "routing",
        "replicate_1",
        "replicate_2",
        "replicate_3",
        "replicate_4",
        "replicate_5",
        "mean",
        "sample_stddev",
        "min",
        "max",
        "correct_total",
        "attempt_total",
    ]
    with CSV_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for k in sorted(values):
            scores = values[k]
            writer.writerow(
                {
                    "expert_count": k,
                    "routing": "native" if k == 8 else "reference-scaled",
                    **{f"replicate_{index}": score for index, score in enumerate(scores, 1)},
                    "mean": statistics.fmean(scores),
                    "sample_stddev": statistics.stdev(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "correct_total": round(sum(scores) * 89),
                    "attempt_total": 5 * 89,
                }
            )


def plot(values: dict[int, list[float]]) -> None:
    expert_counts = sorted(values)
    means = [100 * statistics.fmean(values[k]) for k in expert_counts]
    lows = [100 * min(values[k]) for k in expert_counts]
    highs = [100 * max(values[k]) for k in expert_counts]
    colors = ["#009E73", "#E69F00", "#0072B2", "#CC79A7", "#D55E00"]
    offsets = [-0.18, -0.09, 0.0, 0.09, 0.18]

    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    ax.fill_between(expert_counts, lows, highs, color="#7A8794", alpha=0.12, linewidth=0)
    ax.plot(
        expert_counts,
        means,
        color="#293241",
        linewidth=2.5,
        marker="o",
        markersize=8,
        label="Five-run mean",
        zorder=4,
    )
    for position, k, color in zip(
        expert_counts, expert_counts, colors[: len(expert_counts)], strict=True
    ):
        ax.vlines(position, lows[expert_counts.index(k)], highs[expert_counts.index(k)], color=color, alpha=0.55)
        ax.scatter(
            [position + offset for offset in offsets],
            [100 * score for score in values[k]],
            color=color,
            s=54,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.7,
            zorder=5,
            label="Individual full-suite runs" if k == 4 else None,
        )
        ax.annotate(
            f"{100 * statistics.fmean(values[k]):.2f}%",
            (position, 100 * statistics.fmean(values[k])),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            fontweight="semibold",
        )

    ax.set_xticks(expert_counts, [LABELS[k] for k in expert_counts])
    ax.set_xlabel("Active routed experts", fontweight="semibold")
    ax.set_ylabel("Terminal-Bench 2.0 pass@1 (%)  ↑", fontweight="semibold")
    ax.set_ylim(20, 36)
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(
        "Qwen3.6-35B-A3B · Terminal-Bench 2.0 Expert Sweep",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    ax.set_title(
        "Validated routing conditions only · five 89-task runs per condition · vertical span is observed range",
        fontsize=10.5,
        color="#555A63",
        pad=14,
    )
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    stem = PLOTS / "qwen36_terminalbench_expert_sweep"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    values = load_replicates()
    write_csv(values)
    plot(values)
    print(
        f"Wrote {CSV_PATH} and Terminal-Bench sweep plots from "
        f"{sum(len(scores) for scores in values.values())} valid full-suite runs"
    )


if __name__ == "__main__":
    main()

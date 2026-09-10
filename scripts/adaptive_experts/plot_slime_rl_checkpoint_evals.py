#!/usr/bin/env python3
"""Plot Qwen3 base DAPO-RL checkpoint evaluations across training steps."""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = OLMO_EVAL_ROOT.parent
RESULTS = PROJECT_ROOT / "results" / "rl_checkpoint_evals"
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
CSV_PATH = NOTES / "slime_qwen3_base_dapo_checkpoint_evals.csv"

ORIGINAL_NAME_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-(?P<run>"
    r"k8-normalized|k6-normalized|k6-reference-k8|k4-reference-k8|"
    r"k8-trained-k6-eval|k6-trained-k8-eval)"
    r"-step(?P<step>\d+)-math-(?:eval|aime)-v1-\d{8}$"
)
REPLICATE_NAME_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-(?P<run>"
    r"k8-normalized|k6-normalized|k6-reference-k8|k4-reference-k8|"
    r"k8-trained-k6-eval|k6-trained-k8-eval)"
    r"-step(?P<step>\d+)-math500-(?:replicate-)?rep(?P<replicate>[23])"
    r"(?:-seed\d+)?-v1-\d{8}$"
)
K10_NAME_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-(?P<run>k10-normalized|k10-reference-k8)-"
    r"step(?P<step>\d+)-k10-math-aime-(?:v1|retry1)-20260810$"
)

RUN_ORDER = (
    "k10-normalized",
    "k10-reference-k8",
    "k8-normalized",
    "k8-trained-k6-eval",
    "k6-normalized",
    "k6-trained-k8-eval",
    "k6-reference-k8",
    "k4-reference-k8",
)
RUN_STYLES = {
    "k10-normalized": {
        "label": "Train/eval K=10 normalized",
        "color": "#D55E00",
        "marker": "P",
        "linestyle": "-",
    },
    "k10-reference-k8": {
        "label": "Train/eval K=10 reference-scaled (K=8)",
        "color": "#D55E00",
        "marker": "P",
        "linestyle": "--",
    },
    "k8-normalized": {
        "label": "Train K=8 · eval K=8",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
    },
    "k8-trained-k6-eval": {
        "label": "Train K=8 · eval K=6",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "--",
    },
    "k6-normalized": {
        "label": "Train K=6 · eval K=6",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "-",
    },
    "k6-trained-k8-eval": {
        "label": "Train K=6 · eval K=8",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
    },
    "k6-reference-k8": {
        "label": "Train/eval K=6 reference-scaled (K=8)",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "-",
    },
    "k4-reference-k8": {
        "label": "Train/eval K=4 reference-scaled (K=8)",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
    },
}


def task_by_name(payload: dict, name: str) -> dict:
    matches = [task for task in payload["tasks"] if task["task"] == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name!r} task, found {len(matches)}")
    return matches[0]


def load_rows() -> list[dict[str, object]]:
    rows_by_key: dict[tuple[str, int], dict[str, object]] = {}
    replicates_by_key: dict[tuple[str, int], dict[int, dict[str, object]]] = {}
    for experiment_dir in sorted(path for path in RESULTS.iterdir() if path.is_dir()):
        candidates = list(experiment_dir.rglob("metrics.json"))
        if not candidates:
            continue
        # Older downloads include a main/ wrapper; newer Beaker fetches place
        # the same result files directly under the experiment directory.
        path = min(candidates, key=lambda candidate: len(candidate.parts))
        payload = json.loads(path.read_text())
        if payload.get("errors"):
            raise RuntimeError(f"Metric errors in {path}: {payload['errors']}")

        experiment_name = payload["config"]["metrics"]["experiment_name"]
        original_match = ORIGINAL_NAME_PATTERN.fullmatch(experiment_name)
        replicate_match = REPLICATE_NAME_PATTERN.fullmatch(experiment_name)
        k10_match = K10_NAME_PATTERN.fullmatch(experiment_name)
        if original_match is None and replicate_match is None and k10_match is None:
            continue

        math500 = task_by_name(payload, "math500:chat")
        if math500["num_instances"] != 500:
            raise RuntimeError(
                f"Incomplete MATH-500 evaluation in {path}: {math500['num_instances']} instances"
            )
        math_accuracy = math500["metrics"]["accuracy"]["minerva_math_flex"]

        if replicate_match is not None:
            key = (replicate_match.group("run"), int(replicate_match.group("step")))
            replicate = int(replicate_match.group("replicate"))
            per_key = replicates_by_key.setdefault(key, {})
            if replicate in per_key:
                raise RuntimeError(f"Duplicate replicate {replicate} for {key}")
            per_key[replicate] = {
                "accuracy": math_accuracy,
                "experiment_id": experiment_dir.name,
                "experiment_name": experiment_name,
            }
            continue

        primary_match = original_match or k10_match
        assert primary_match is not None
        aime = task_by_name(payload, "aime_2025:pass_at_32")
        if aime["num_instances"] != 30:
            raise RuntimeError(f"Incomplete AIME evaluation in {path}: {aime['num_instances']} instances")
        aime_metrics = aime["metrics"]
        key = (primary_match.group("run"), int(primary_match.group("step")))
        if key in rows_by_key:
            raise RuntimeError(f"Duplicate original checkpoint evaluation for {key}")
        rows_by_key[key] = {
            "run": key[0],
            "step": key[1],
            "math500_accuracy_replicate_1": math_accuracy,
            "aime_pass_at_1": aime_metrics["pass_at_1"]["minerva_math_flex"],
            "aime_pass_at_4": aime_metrics["pass_at_4"]["minerva_math_flex"],
            "aime_pass_at_8": aime_metrics["pass_at_8"]["minerva_math_flex"],
            "aime_pass_at_16": aime_metrics["pass_at_16"]["minerva_math_flex"],
            "aime_pass_at_32": aime_metrics["pass_at_32"]["minerva_math_flex"],
            "experiment_id": experiment_dir.name,
            "experiment_name": experiment_name,
        }

    replicate_keys = set(replicates_by_key)
    replicated_original_keys = {
        key for key in rows_by_key if not key[0].startswith("k10-")
    }
    if replicate_keys != replicated_original_keys:
        raise RuntimeError(
            "MATH replicate/original key mismatch: "
            f"missing_replicates={sorted(replicated_original_keys - replicate_keys)}, "
            f"unexpected_replicates={sorted(replicate_keys - replicated_original_keys)}"
        )

    rows: list[dict[str, object]] = []
    for key, row in rows_by_key.items():
        replicates = replicates_by_key.get(key, {})
        if not key[0].startswith("k10-") and set(replicates) != {2, 3}:
            raise RuntimeError(f"Expected MATH replicates 2 and 3 for {key}, found {sorted(replicates)}")
        values = [float(row["math500_accuracy_replicate_1"])]
        values.extend(float(replicates[index]["accuracy"]) for index in sorted(replicates))
        row.update(
            {
                "math500_accuracy": statistics.fmean(values),
                "math500_accuracy_stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "math500_accuracy_min": min(values),
                "math500_accuracy_max": max(values),
                "math500_replicate_count": len(values),
                "math500_accuracy_replicate_2": None if 2 not in replicates else replicates[2]["accuracy"],
                "math500_accuracy_replicate_3": None if 3 not in replicates else replicates[3]["accuracy"],
                "math500_replicate_2_experiment_id": None
                if 2 not in replicates
                else replicates[2]["experiment_id"],
                "math500_replicate_3_experiment_id": None
                if 3 not in replicates
                else replicates[3]["experiment_id"],
            }
        )
        rows.append(row)

    # Step zero is the same untouched model for every training arm. Reuse the
    # matching routing-policy baselines instead of launching duplicate jobs.
    by_key = {(str(row["run"]), int(row["step"])): row for row in rows}
    synthetic_step_zero = (
        ("k8-trained-k6-eval", "k6-normalized"),
        ("k6-trained-k8-eval", "k8-normalized"),
    )
    for target_run, source_run in synthetic_step_zero:
        source = by_key.get((source_run, 0))
        if source is not None and (target_run, 0) not in by_key:
            copied = dict(source)
            copied["run"] = target_run
            copied["experiment_name"] = f"{source['experiment_name']} (reused step-0 baseline)"
            rows.append(copied)

    expected_runs = (
        "k10-normalized",
        "k10-reference-k8",
        "k8-normalized",
        "k6-normalized",
        "k4-reference-k8",
    )
    expected = {(run, step) for run in expected_runs for step in (20, 40, 60, 80, 100)}
    found = {(str(row["run"]), int(row["step"])) for row in rows}
    if not expected.issubset(found):
        raise RuntimeError(f"Initial checkpoint matrix incomplete: missing={expected - found}")
    if len(found) != len(rows):
        raise RuntimeError("Duplicate run/step checkpoint evaluations found")
    return sorted(rows, key=lambda row: (RUN_ORDER.index(str(row["run"])), int(row["step"])))


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "run",
        "step",
        "math500_accuracy",
        "math500_accuracy_stddev",
        "math500_accuracy_min",
        "math500_accuracy_max",
        "math500_replicate_count",
        "math500_accuracy_replicate_1",
        "math500_accuracy_replicate_2",
        "math500_accuracy_replicate_3",
        "aime_pass_at_1",
        "aime_pass_at_4",
        "aime_pass_at_8",
        "aime_pass_at_16",
        "aime_pass_at_32",
        "experiment_id",
        "experiment_name",
        "math500_replicate_2_experiment_id",
        "math500_replicate_3_experiment_id",
    )
    with CSV_PATH.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(
    rows: list[dict[str, object]],
    *,
    metric: str,
    title: str,
    subtitle: str,
    ylabel: str,
    filename: str,
    uncertainty_metric: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    steps = sorted({int(row["step"]) for row in rows})

    for run in RUN_ORDER:
        run_rows = [row for row in rows if row["run"] == run]
        if not run_rows:
            continue
        style = RUN_STYLES[run]
        line = ax.plot(
            [int(row["step"]) for row in run_rows],
            [100 * float(row[metric]) for row in run_rows],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.6,
            markersize=7.5,
        )[0]
        if style["linestyle"] == "--":
            line.set_markerfacecolor("white")
            line.set_markeredgewidth(1.8)
        if uncertainty_metric is not None:
            ax.errorbar(
                [int(row["step"]) for row in run_rows],
                [100 * float(row[metric]) for row in run_rows],
                yerr=[100 * float(row[uncertainty_metric]) for row in run_rows],
                color=style["color"],
                linestyle="none",
                linewidth=1.0,
                capsize=2.5,
                alpha=0.45,
                zorder=1,
            )

    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=10.5, color="#555A63", pad=12)
    ax.set_xlabel("RL training step", fontweight="semibold")
    ax.set_ylabel(ylabel, fontweight="semibold")
    ax.set_xticks(steps)
    ax.set_xlim(min(steps) - 8, max(steps) + 8)
    values = [100 * float(row[metric]) for row in rows]
    span = max(values) - min(values)
    padding = max(1.5, 0.1 * span)
    ax.set_ylim(max(0, min(values) - padding), min(100, max(values) + padding))
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.65)
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=9.2, columnspacing=1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            PLOTS / f"{filename}.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    plot_metric(
        rows,
        metric="aime_pass_at_1",
        title="Qwen3-30B-A3B Base: DAPO RL · AIME 2025",
        subtitle="Pass@1 estimate from 32 sampled responses per problem · one evaluation per checkpoint",
        ylabel="AIME 2025 Pass@1 (%)  ↑",
        filename="slime_qwen3_base_dapo_rl_aime_by_step",
    )
    plot_metric(
        rows,
        metric="aime_pass_at_8",
        title="Qwen3-30B-A3B Base: DAPO RL · AIME 2025 Pass@8",
        subtitle="Pass@8 estimate from 32 sampled responses per problem · one evaluation per checkpoint",
        ylabel="AIME 2025 Pass@8 (%)  ↑",
        filename="slime_qwen3_base_dapo_rl_aime_pass_at_8_by_step",
    )
    plot_metric(
        rows,
        metric="math500_accuracy",
        title="Qwen3-30B-A3B Base: DAPO RL · MATH-500",
        subtitle=(
            "K=4/6/8: mean of 3 evaluations with ±1 SD · K=10: one evaluation per checkpoint"
        ),
        ylabel="MATH-500 accuracy (%)  ↑",
        filename="slime_qwen3_base_dapo_rl_math500_by_step",
        uncertainty_metric="math500_accuracy_stddev",
    )
    print(f"Wrote {len(rows)} checkpoint rows to {CSV_PATH}")


if __name__ == "__main__":
    main()

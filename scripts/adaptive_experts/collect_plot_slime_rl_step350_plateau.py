#!/usr/bin/env python3
"""Collect and plot the step-350 RL inference-K MATH-500 plateau sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PHASE = "slime-rl-step350-plateau-math500"
TAG_RE = re.compile(
    r"^qwen3-base-dapo-"
    r"(?P<run>k8-normalized|k6-reference-k8|k4-reference-k8)"
    r"-step350-plateau-(?P<mode>normalized|reference)"
    r"-k(?P<k>\d+)-r(?P<replicate>\d+)-v3-math500-only-20260729$"
)
RUN_LABELS = {
    "k8-normalized": "Trained K=8 normalized",
    "k6-reference-k8": "Trained K=6 reference-scaled",
    "k4-reference-k8": "Trained K=4 reference-scaled",
}
MODE_LABELS = {
    "normalized": "Normalized inference",
    "reference": "Reference-scaled inference",
}
PRIOR_CONDITIONS = {
    ("k8-normalized", "normalized", 8): "k8-normalized",
    ("k8-normalized", "normalized", 6): "k8-trained-k6-eval",
    ("k6-reference-k8", "reference", 6): "k6-reference-k8",
    ("k4-reference-k8", "reference", 4): "k4-reference-k8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--ledger", type=Path, default=root / "notes/beaker_jobs.jsonl")
    parser.add_argument(
        "--results-root", type=Path, default=root / "results/adaptive_experts"
    )
    parser.add_argument(
        "--checkpoint-csv",
        type=Path,
        default=root / "notes/slime_qwen3_base_dapo_checkpoint_evals.csv",
    )
    parser.add_argument(
        "--replicate-csv",
        type=Path,
        default=root / "notes/slime_qwen3_base_dapo_step350_plateau_replicates.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=root / "notes/slime_qwen3_base_dapo_step350_plateau_summary.csv",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=root / "notes/plots/slime_qwen3_base_dapo_step350_math500_plateau.png",
    )
    return parser.parse_args()


def read_ledger(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if record.get("phase") == PHASE:
                    records.append(record)
    return records


def prediction_stats(result_dir: Path) -> tuple[int, int]:
    files = list(result_dir.glob("predictions/**/*math500*-predictions.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected one MATH-500 prediction file in {result_dir}, got {files}")
    count = 0
    empty = 0
    with files[0].open() as f:
        for line in f:
            item = json.loads(line)
            outputs = item.get("model_output") or []
            text = outputs[0].get("text", "") if outputs else ""
            count += 1
            empty += not bool(text.strip())
    return count, empty


def collect_new(records: list[dict], results_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for record in records:
        experiment_id = record["beaker_experiment_id"]
        if experiment_id in seen:
            continue
        seen.add(experiment_id)
        match = TAG_RE.match(record["run_tag"])
        if not match:
            raise ValueError(f"Unexpected run tag: {record['run_tag']}")
        result_dir = results_root / experiment_id
        metrics_path = result_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("errors"):
            raise ValueError(f"{experiment_id} has metric errors: {metrics['errors']}")
        task_rows = [task for task in metrics["tasks"] if task["task"] == "math500:chat"]
        if len(task_rows) != 1:
            raise ValueError(f"{experiment_id} has unexpected task records: {task_rows}")
        task = task_rows[0]
        score = task["metrics"]["accuracy"]["minerva_math_flex"]
        prediction_count, empty_outputs = prediction_stats(result_dir)
        if task["num_instances"] != 500 or prediction_count != 500:
            raise ValueError(
                f"{experiment_id} is incomplete: metrics={task['num_instances']}, "
                f"predictions={prediction_count}"
            )
        rows.append(
            {
                "run": match["run"],
                "mode": match["mode"],
                "k": int(match["k"]),
                "replicate": int(match["replicate"]),
                "score": float(score),
                "experiment_id": experiment_id,
                "source": "new-sweep",
                "num_instances": prediction_count,
                "empty_outputs": empty_outputs,
            }
        )
    return rows


def collect_prior(path: Path) -> list[dict]:
    with path.open() as f:
        checkpoint_rows = list(csv.DictReader(f))
    rows = []
    for (run, mode, k), checkpoint_run in PRIOR_CONDITIONS.items():
        matches = [
            row
            for row in checkpoint_rows
            if row["run"] == checkpoint_run and int(row["step"]) == 350
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one prior row for {checkpoint_run} step 350, got {len(matches)}"
            )
        source = matches[0]
        scores = [float(source[f"math500_accuracy_replicate_{rep}"]) for rep in (1, 2, 3)]
        experiment_ids = [
            source["experiment_id"],
            source["math500_replicate_2_experiment_id"],
            source["math500_replicate_3_experiment_id"],
        ]
        for replicate, (score, experiment_id) in enumerate(
            zip(scores, experiment_ids, strict=True), start=1
        ):
            rows.append(
                {
                    "run": run,
                    "mode": mode,
                    "k": k,
                    "replicate": replicate,
                    "score": score,
                    "experiment_id": experiment_id,
                    "source": "reused-checkpoint-eval",
                    "num_instances": 500,
                    "empty_outputs": 0,
                }
            )
    return rows


def add_reference_k8_aliases(rows: list[dict]) -> list[dict]:
    output = list(rows)
    for run in RUN_LABELS:
        normalized = [
            row for row in rows if row["run"] == run and row["mode"] == "normalized" and row["k"] == 8
        ]
        for row in normalized:
            alias = dict(row)
            alias["mode"] = "reference"
            alias["source"] = f"{row['source']};normalized-k8-alias"
            output.append(alias)
    return output


def deduplicate(rows: list[dict]) -> list[dict]:
    by_key = {}
    source_priority = {"reused-checkpoint-eval": 2, "new-sweep": 1}
    for row in rows:
        key = (row["run"], row["mode"], row["k"], row["replicate"])
        base_source = row["source"].split(";")[0]
        priority = source_priority.get(base_source, 0)
        if key not in by_key or priority > by_key[key][0]:
            by_key[key] = (priority, row)
    return [value[1] for value in by_key.values()]


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run"], row["mode"], row["k"])].append(row)
    summaries = []
    for (run, mode, k), group in sorted(grouped.items()):
        scores = [row["score"] for row in group]
        summaries.append(
            {
                "run": run,
                "mode": mode,
                "k": k,
                "replicate_count": len(scores),
                "complete": len(scores) == 3,
                "mean": statistics.mean(scores),
                "sample_stddev": statistics.stdev(scores) if len(scores) > 1 else math.nan,
                "min": min(scores),
                "max": max(scores),
                "empty_outputs": sum(row["empty_outputs"] for row in group),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, summaries: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), sharey=True)
    colors = {"normalized": "#0072B2", "reference": "#D55E00"}
    for axis, run in zip(axes, RUN_LABELS, strict=True):
        for mode in MODE_LABELS:
            points = [
                row
                for row in summaries
                if row["run"] == run and row["mode"] == mode and row["complete"]
            ]
            points.sort(key=lambda row: row["k"])
            if not points:
                continue
            x = [row["k"] for row in points]
            y = [100 * row["mean"] for row in points]
            lower = [100 * row["min"] for row in points]
            upper = [100 * row["max"] for row in points]
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                color=colors[mode],
                label=MODE_LABELS[mode],
            )
            axis.fill_between(x, lower, upper, color=colors[mode], alpha=0.14)
        axis.set_title(RUN_LABELS[run])
        axis.set_xlabel("Experts used at inference")
        axis.set_xticks(range(2, 13))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("MATH-500 accuracy (%)")
    axes[-1].legend(loc="best")
    fig.suptitle("Qwen3-30B-A3B DAPO step 350: inference-time expert plateau")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    new_rows = collect_new(read_ledger(args.ledger), args.results_root)
    rows = deduplicate(collect_prior(args.checkpoint_csv) + new_rows)
    rows = add_reference_k8_aliases(rows)
    rows.sort(key=lambda row: (row["run"], row["mode"], row["k"], row["replicate"]))
    summaries = summarize(rows)
    write_csv(
        args.replicate_csv,
        rows,
        [
            "run",
            "mode",
            "k",
            "replicate",
            "score",
            "experiment_id",
            "source",
            "num_instances",
            "empty_outputs",
        ],
    )
    write_csv(
        args.summary_csv,
        summaries,
        [
            "run",
            "mode",
            "k",
            "replicate_count",
            "complete",
            "mean",
            "sample_stddev",
            "min",
            "max",
            "empty_outputs",
        ],
    )
    plot(args.plot, summaries)
    complete = sum(row["complete"] for row in summaries)
    partial = len(summaries) - complete
    print(
        f"Collected {len(new_rows)} new evaluations; "
        f"{complete} complete logical points and {partial} partial points."
    )
    print(f"Empty generated outputs across new evaluations: {sum(r['empty_outputs'] for r in new_rows)}")
    print(args.summary_csv)
    print(args.plot)


if __name__ == "__main__":
    main()

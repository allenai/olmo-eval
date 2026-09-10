#!/usr/bin/env python3
"""Plot observed compute costs for the Qwen3-base DAPO RL experiments.

No theoretical FLOP estimates are used. Training cost comes from finalized
checkpoint timestamps on the continuously allocated 8xB200 jobs. Evaluation
cost comes from olmo-eval's recorded experiment duration and vLLM completion
token telemetry on the 4xH100 checkpoint evaluations.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = OLMO_EVAL_ROOT.parent
CHECKPOINTS = (
    PROJECT_ROOT / "slime-runs/qwen3-30b-a3b-base-dapo/checkpoints"
)
RESULTS = PROJECT_ROOT / "results/rl_checkpoint_evals"
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
CSV_PATH = NOTES / "slime_qwen3_base_dapo_compute_costs.csv"

TRAINING_RUNS = {
    "k8-normalized": {
        "directory": "slime-qwen3-base-dapo-k8-pilot-100step-8gpu-v5-20260725",
        "label": "Train K=8 normalized",
        "color": "#0072B2",
        "marker": "o",
        "restart_after": 100,
        "actual_k": 8,
    },
    "k6-normalized": {
        "directory": "slime-qwen3-base-dapo-k6-pilot-100step-8gpu-v1-20260725",
        "label": "Train K=6 normalized",
        "color": "#E69F00",
        "marker": "s",
        "restart_after": 100,
        "actual_k": 6,
    },
    "k6-reference-k8": {
        "directory": "slime-qwen3-base-dapo-k6-reference-k8-500step-8gpu-v1-20260726",
        "label": "Train K=6 reference-scaled",
        "color": "#CC79A7",
        "marker": "D",
        "restart_after": None,
        "actual_k": 6,
    },
    "k4-reference-k8": {
        "directory": "slime-qwen3-base-dapo-k4-reference-k8-pilot-100step-8gpu-v1-20260725",
        "label": "Train K=4 reference-scaled",
        "color": "#009E73",
        "marker": "^",
        "restart_after": 100,
        "actual_k": 4,
    },
}

EVAL_NAME_PATTERN = re.compile(
    r"^adaptive-qwen3-base-dapo-(?P<run>"
    r"k8-normalized|k6-normalized|k6-reference-k8|k4-reference-k8|"
    r"k8-trained-k6-eval|k6-trained-k8-eval)"
    r"-step(?P<step>\d+)-math-eval-v1-\d{8}$"
)
EVAL_STYLES = {
    "k8-normalized": {
        "label": "Train K=8 · eval kernel K=8",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "actual_k": 8,
    },
    "k8-trained-k6-eval": {
        "label": "Train K=8 · eval kernel K=6",
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "--",
        "actual_k": 6,
    },
    "k6-normalized": {
        "label": "Train K=6 · eval kernel K=6",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "-",
        "actual_k": 6,
    },
    "k6-trained-k8-eval": {
        "label": "Train K=6 · eval kernel K=8",
        "color": "#E69F00",
        "marker": "s",
        "linestyle": "--",
        "actual_k": 8,
    },
    "k6-reference-k8": {
        "label": "Train K=6 ref · eval K=8 zero-mask",
        "color": "#CC79A7",
        "marker": "D",
        "linestyle": "-",
        "actual_k": 8,
    },
    "k4-reference-k8": {
        "label": "Train K=4 ref · eval K=8 zero-mask",
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-",
        "actual_k": 8,
    },
}


def finalized_checkpoint_time(checkpoint_dir: Path) -> float:
    metadata = checkpoint_dir / "metadata.json"
    if not metadata.is_file():
        raise FileNotFoundError(f"missing finalized checkpoint metadata: {metadata}")
    return metadata.stat().st_mtime


def load_training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"iter_(\d+)$")

    for arm, style in TRAINING_RUNS.items():
        root = CHECKPOINTS / str(style["directory"])
        checkpoints: list[tuple[int, float]] = []
        for path in root.glob("iter_*"):
            match = pattern.fullmatch(path.name)
            if match is None:
                continue
            step = int(match.group(1)) + 1
            checkpoints.append((step, finalized_checkpoint_time(path)))
        checkpoints.sort()

        for (start_step, start_time), (end_step, end_time) in zip(
            checkpoints, checkpoints[1:], strict=False
        ):
            # The original 100-step jobs and the 500-step continuations were
            # separate Beaker allocations. Do not count the idle restart gap.
            if style["restart_after"] == start_step:
                continue
            updates = end_step - start_step
            wall_seconds = end_time - start_time
            if updates <= 0 or wall_seconds <= 0:
                raise RuntimeError(
                    f"invalid checkpoint interval for {arm}: "
                    f"{start_step}->{end_step}, {wall_seconds=}"
                )
            gpu_count = 8
            rows.append(
                {
                    "kind": "training",
                    "arm": arm,
                    "step": end_step,
                    "interval_start_step": start_step,
                    "interval_updates": updates,
                    "hardware": "NVIDIA B200",
                    "gpu_count": gpu_count,
                    "wall_seconds": wall_seconds,
                    "gpu_hours": wall_seconds * gpu_count / 3600,
                    "gpu_hours_per_50_updates": (
                        wall_seconds * gpu_count / 3600 * 50 / updates
                    ),
                    "completion_tokens": "",
                    "gpu_seconds_per_million_completion_tokens": "",
                    "actual_expert_kernel_k": style["actual_k"],
                    "experiment_id": "",
                    "experiment_name": "",
                    "source": str(root / f"iter_{end_step - 1:07d}" / "metadata.json"),
                }
            )
    return rows


def shallowest_metrics(experiment_dir: Path) -> Path | None:
    candidates = list(experiment_dir.rglob("metrics.json"))
    if not candidates:
        return None
    return min(candidates, key=lambda path: len(path.parts))


def completion_tokens(metrics_path: Path) -> int | None:
    telemetry_files = sorted(metrics_path.parent.rglob("*inference.jsonl"))
    if not telemetry_files:
        return None
    total = 0
    batch_count = 0
    for path in telemetry_files:
        with path.open() as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("type") != "batch":
                    continue
                total += int(payload["data"]["total_completion_tokens"])
                batch_count += 1
    if batch_count == 0 or total <= 0:
        return None
    return total


def load_eval_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for experiment_dir in sorted(path for path in RESULTS.iterdir() if path.is_dir()):
        metrics_path = shallowest_metrics(experiment_dir)
        if metrics_path is None:
            continue
        payload = json.loads(metrics_path.read_text())
        name = payload.get("config", {}).get("metrics", {}).get("experiment_name", "")
        match = EVAL_NAME_PATTERN.fullmatch(name)
        if match is None:
            continue
        if payload.get("errors"):
            raise RuntimeError(f"metric errors in {metrics_path}: {payload['errors']}")
        tokens = completion_tokens(metrics_path)
        if tokens is None:
            continue

        run = match.group("run")
        step = int(match.group("step"))
        key = (run, step)
        if key in seen:
            raise RuntimeError(f"duplicate eval telemetry for {key}")
        seen.add(key)

        duration = float(payload["experiment_duration_seconds"])
        gpu_count = 4
        gpu_seconds = duration * gpu_count
        rows.append(
            {
                "kind": "evaluation",
                "arm": run,
                "step": step,
                "interval_start_step": "",
                "interval_updates": "",
                "hardware": "NVIDIA H100 80GB HBM3",
                "gpu_count": gpu_count,
                "wall_seconds": duration,
                "gpu_hours": gpu_seconds / 3600,
                "gpu_hours_per_50_updates": "",
                "completion_tokens": tokens,
                "gpu_seconds_per_million_completion_tokens": gpu_seconds / tokens * 1_000_000,
                "actual_expert_kernel_k": EVAL_STYLES[run]["actual_k"],
                "experiment_id": experiment_dir.name,
                "experiment_name": name,
                "source": str(metrics_path),
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = (
        "kind",
        "arm",
        "step",
        "interval_start_step",
        "interval_updates",
        "hardware",
        "gpu_count",
        "wall_seconds",
        "gpu_hours",
        "gpu_hours_per_50_updates",
        "completion_tokens",
        "gpu_seconds_per_million_completion_tokens",
        "actual_expert_kernel_k",
        "experiment_id",
        "experiment_name",
        "source",
    )
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.85)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)


def plot(training_rows: list[dict[str, object]], eval_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 14.8), sharex=False)
    fig.suptitle(
        "Qwen3-30B-A3B DAPO RL: Observed Compute Cost",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )

    ax = axes[0]
    for arm, style in TRAINING_RUNS.items():
        arm_rows = sorted(
            (row for row in training_rows if row["arm"] == arm),
            key=lambda row: int(row["step"]),
        )
        if not arm_rows:
            continue
        ax.plot(
            [int(row["step"]) for row in arm_rows],
            [float(row["gpu_hours_per_50_updates"]) for row in arm_rows],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.3,
            markersize=7,
        )
    ax.set_title(
        "Training · 8×B200 checkpoint-to-checkpoint wall time × GPU count · restart gaps excluded",
        fontsize=11,
        color="#555A63",
    )
    ax.set_ylabel("GPU-hours per 50 RL updates  ↓", fontweight="semibold")
    ax.set_xlabel("RL training step", fontweight="semibold")
    ax.legend(frameon=False, ncol=2, fontsize=9.5)
    style_axis(ax)

    for ax, metric, ylabel, title in (
        (
            axes[1],
            "gpu_hours",
            "GPU-hours per complete eval  ↓",
            "Evaluation · 4×H100 experiment wall time × GPU count · includes startup and output-length effects",
        ),
        (
            axes[2],
            "gpu_seconds_per_million_completion_tokens",
            "GPU-seconds / 1M completion tokens  ↓",
            "Evaluation efficiency · same measured cost normalized by recorded generated tokens",
        ),
    ):
        for run, style in EVAL_STYLES.items():
            run_rows = sorted(
                (row for row in eval_rows if row["arm"] == run),
                key=lambda row: int(row["step"]),
            )
            if not run_rows:
                continue
            line = ax.plot(
                [int(row["step"]) for row in run_rows],
                [float(row[metric]) for row in run_rows],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6.5,
            )[0]
            if style["linestyle"] == "--":
                line.set_markerfacecolor("white")
                line.set_markeredgewidth(1.6)
        ax.set_title(title, fontsize=11, color="#555A63")
        ax.set_ylabel(ylabel, fontweight="semibold")
        ax.set_xlabel("RL training step", fontweight="semibold")
        style_axis(ax)

    axes[1].legend(frameon=False, ncol=2, fontsize=8.8)
    fig.text(
        0.5,
        0.006,
        "All values are observed measurements; 54 evals with stored inference telemetry are shown. "
        "Reference-scaled olmo-eval curves still dispatch K=8 zero-masked slots.",
        ha="center",
        fontsize=10,
        color="#555A63",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98), h_pad=2.3)
    PLOTS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            PLOTS / f"slime_qwen3_base_dapo_observed_compute_costs.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    training_rows = load_training_rows()
    eval_rows = load_eval_rows()
    rows = sorted(
        training_rows + eval_rows,
        key=lambda row: (str(row["kind"]), str(row["arm"]), int(row["step"])),
    )
    write_csv(rows)
    plot(training_rows, eval_rows)
    print(
        f"Wrote {len(training_rows)} training intervals and {len(eval_rows)} eval measurements "
        f"to {CSV_PATH}"
    )


if __name__ == "__main__":
    main()

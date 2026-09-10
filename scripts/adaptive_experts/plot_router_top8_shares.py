#!/usr/bin/env python3
"""Plot Qwen3's normalized top-eight router mixture across MoE layers."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASKS = ("gpqa_diamond", "ifeval_ood", "math500")
NUM_LAYERS = 48
NUM_SELECTED = 8


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    results = repo / "results" / "router_profile"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        nargs="+",
        default=[
            results
            / "01KXBSST8ZP8VBZ469GWB1CWCR.partial"
            / "workers"
            / "rank2"
            / "samples.jsonl",
            *sorted(
                (
                    results
                    / "01KXD4JDGY5ZC96XHB5E3VKKDC"
                    / "workers"
                ).glob("rank*/samples.jsonl")
            ),
        ],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "notes" / "plots" / "qwen3_k8_top8_share_by_layer.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=(
            repo
            / "notes"
            / "router_profile"
            / "qwen3_k8_top8_share_by_layer.csv"
        ),
    )
    return parser.parse_args()


def load_layer_shares(paths: list[Path]) -> tuple[np.ndarray, dict[str, int]]:
    """Return task-balanced cumulative shares with prompts balanced inside each task."""
    by_example_layer: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
    examples_by_task: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                task, _ = row["example_key"].split(":", maxsplit=1)
                if task not in TASKS:
                    continue
                top8 = np.asarray(row["top16_probs"][:NUM_SELECTED], dtype=np.float64)
                cumulative_share = np.cumsum(top8) / top8.sum()
                layer = int(row["layer"])
                by_example_layer[(task, row["example_key"], layer)].append(cumulative_share)
                examples_by_task[task].add(row["example_key"])

    task_layer_means = []
    for task in TASKS:
        layer_means = []
        for layer in range(NUM_LAYERS):
            example_means = [
                np.mean(rows, axis=0)
                for (row_task, _example, row_layer), rows in by_example_layer.items()
                if row_task == task and row_layer == layer
            ]
            if not example_means:
                raise ValueError(f"no samples for task={task}, layer={layer}")
            layer_means.append(np.mean(example_means, axis=0))
        task_layer_means.append(layer_means)

    return np.mean(np.asarray(task_layer_means), axis=0), {
        task: len(examples_by_task[task]) for task in TASKS
    }


def write_csv(path: Path, cumulative: np.ndarray) -> None:
    individual = np.diff(
        np.concatenate((np.zeros((NUM_LAYERS, 1)), cumulative), axis=1), axis=1
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = (
            ["layer"]
            + [f"cumulative_top{k}_percent" for k in range(1, NUM_SELECTED + 1)]
            + [f"rank{k}_percent" for k in range(1, NUM_SELECTED + 1)]
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for layer in range(NUM_LAYERS):
            row: dict[str, float | int] = {"layer": layer}
            for k in range(NUM_SELECTED):
                row[f"cumulative_top{k + 1}_percent"] = cumulative[layer, k] * 100
                row[f"rank{k + 1}_percent"] = individual[layer, k] * 100
            writer.writerow(row)


def plot(path: Path, cumulative: np.ndarray, prompt_counts: dict[str, int]) -> None:
    layers = np.arange(NUM_LAYERS)
    cumulative_percent = cumulative * 100
    individual_percent = np.diff(
        np.concatenate((np.zeros((NUM_LAYERS, 1)), cumulative_percent), axis=1), axis=1
    )
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, NUM_SELECTED))

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.5), sharex=True)
    for k, color in enumerate(colors, start=1):
        style = "--" if k == NUM_SELECTED else "-"
        axes[0].plot(
            layers,
            cumulative_percent[:, k - 1],
            color=color,
            linewidth=1.8,
            linestyle=style,
            label=f"Top {k}",
        )
        axes[1].plot(
            layers,
            individual_percent[:, k - 1],
            color=color,
            linewidth=1.8,
            label=f"Rank {k}",
        )

    axes[0].set_title("Cumulative share retained from the K=8 mixture", loc="left")
    axes[0].set_ylabel("Cumulative gate weight (%)")
    axes[0].set_ylim(0, 103)
    axes[0].legend(ncol=4, frameon=False, loc="lower right")

    axes[1].set_title("Individual selected expert-rank share", loc="left")
    axes[1].set_xlabel("MoE layer index")
    axes[1].set_ylabel("Gate weight (%)")
    axes[1].set_ylim(0, max(30, individual_percent.max() * 1.08))
    axes[1].legend(ncol=4, frameon=False, loc="upper right")
    axes[1].set_xticks([*range(0, NUM_LAYERS, 4), NUM_LAYERS - 1])
    axes[1].set_xlim(0, NUM_LAYERS - 1)

    for axis in axes:
        axis.grid(True, alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    counts = ", ".join(f"{task}={prompt_counts[task]}" for task in TASKS)
    fig.suptitle(
        "Qwen3-30B-A3B: normalized router mixture within the selected top eight",
        fontsize=15,
        fontweight="semibold",
        x=0.08,
        ha="left",
    )
    fig.text(
        0.08,
        0.925,
        "Prompt-balanced within task and task-balanced across saved "
        f"token-layer samples ({counts})",
        fontsize=9.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0.04, 0.03, 0.98, 0.91), h_pad=2.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cumulative, prompt_counts = load_layer_shares(args.samples)
    write_csv(args.csv_output, cumulative)
    plot(args.output, cumulative, prompt_counts)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.svg')}")
    print(f"wrote {args.csv_output}")
    print(f"prompt counts: {prompt_counts}")


if __name__ == "__main__":
    main()

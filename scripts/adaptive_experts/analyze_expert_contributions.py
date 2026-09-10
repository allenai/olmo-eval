#!/usr/bin/env python3
"""Summarize and plot Qwen3 expert activations and weighted contributions."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PHASES = ("prompt", "reasoning", "final")
RANK_METRICS = (
    "gate_weight",
    "expert_output_norm",
    "weighted_contribution_norm",
    "contribution_norm_share",
    "projection_share",
    "leave_one_out_relative_l2",
    "leave_one_out_cosine",
    "contribution_rank_position",
)
K_METRICS = (
    "cumulative_gate_share",
    "cumulative_contribution_norm_share",
    "counterfactual_cosine",
    "counterfactual_relative_l2",
    "counterfactual_norm_ratio",
)
SCALAR_METRICS = (
    "moe_output_norm",
    "residual_norm",
    "moe_to_residual_norm_ratio",
    "coherence_ratio",
    "bottom4_contribution_norm_share",
    "bottom4_projection_share",
    "bottom4_top4_inversion_fraction",
    "bottom4_any_exceeds_top4",
    "gate_contribution_top4_overlap",
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "notes" / "expert_contribution",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=repo / "notes" / "plots",
    )
    return parser.parse_args()


def empty_stats(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(shape, dtype=np.int64),
        "rank_sum": np.zeros((*shape, 8, len(RANK_METRICS)), dtype=np.float64),
        "k_sum": np.zeros((*shape, 8, len(K_METRICS)), dtype=np.float64),
        "scalar_sum": np.zeros((*shape, len(SCALAR_METRICS)), dtype=np.float64),
    }


def load_root_stats(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            key: data[key]
            for key in ("count", "rank_sum", "k_sum", "scalar_sum")
        }


def load_task_stats(results: Path, shape: tuple[int, int]) -> dict[str, dict[str, np.ndarray]]:
    task_stats: dict[str, dict[str, np.ndarray]] = defaultdict(lambda: empty_stats(shape))
    paths = sorted((results / "workers").glob("rank*/examples.jsonl.gz"))
    if not paths:
        raise ValueError(f"no per-example files found beneath {results}")
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                stats = task_stats[row["task"]]
                count = np.asarray(row["count"], dtype=np.int64)
                stats["count"] += count
                stats["rank_sum"] += np.asarray(row["rank_means"]) * count[..., None, None]
                stats["k_sum"] += np.asarray(row["k_means"]) * count[..., None, None]
                stats["scalar_sum"] += np.asarray(row["scalar_means"]) * count[..., None]
    return dict(task_stats)


def load_example_stats(results: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    paths = sorted((results / "workers").glob("rank*/examples.jsonl.gz"))
    if not paths:
        raise ValueError(f"no per-example files found beneath {results}")
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                count = np.asarray(row["count"], dtype=np.int64)
                examples.append(
                    {
                        "task": row["task"],
                        "count": count,
                        "rank_sum": np.asarray(row["rank_means"]) * count[..., None, None],
                        "k_sum": np.asarray(row["k_means"]) * count[..., None, None],
                        "scalar_sum": np.asarray(row["scalar_means"]) * count[..., None],
                    }
                )
    return examples


def metric_index(names: tuple[str, ...], name: str) -> int:
    return names.index(name)


def collapse(
    stats: dict[str, np.ndarray], phases: tuple[str, ...], *, keep_layer: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase_ids = [PHASES.index(phase) for phase in phases]
    count = stats["count"][phase_ids].sum(axis=0)
    rank_sum = stats["rank_sum"][phase_ids].sum(axis=0)
    k_sum = stats["k_sum"][phase_ids].sum(axis=0)
    scalar_sum = stats["scalar_sum"][phase_ids].sum(axis=0)
    if keep_layer:
        denominator = np.maximum(count, 1)
        return (
            count,
            rank_sum / denominator[:, None, None],
            k_sum / denominator[:, None, None],
            scalar_sum / denominator[:, None],
        )
    denominator = max(int(count.sum()), 1)
    return (
        np.asarray(count.sum()),
        rank_sum.sum(axis=0) / denominator,
        k_sum.sum(axis=0) / denominator,
        scalar_sum.sum(axis=0) / denominator,
    )


def compact_summary(stats: dict[str, np.ndarray], phases: tuple[str, ...]) -> dict[str, Any]:
    count, rank, by_k, scalar = collapse(stats, phases)
    gate = rank[:, metric_index(RANK_METRICS, "gate_weight")]
    contribution_share = rank[:, metric_index(RANK_METRICS, "contribution_norm_share")]
    output_norm = rank[:, metric_index(RANK_METRICS, "expert_output_norm")]
    weighted_norm = rank[:, metric_index(RANK_METRICS, "weighted_contribution_norm")]
    return {
        "token_layer_observations": int(count),
        "gate_weight_by_rank": gate.tolist(),
        "contribution_norm_share_by_rank": contribution_share.tolist(),
        "expert_output_norm_by_rank": output_norm.tolist(),
        "expert_output_norm_relative_to_rank1": (output_norm / output_norm[0]).tolist(),
        "weighted_contribution_norm_by_rank": weighted_norm.tolist(),
        "contribution_rank_position_by_gate_rank": rank[
            :, metric_index(RANK_METRICS, "contribution_rank_position")
        ].tolist(),
        "leave_one_out_relative_l2_by_rank": rank[
            :, metric_index(RANK_METRICS, "leave_one_out_relative_l2")
        ].tolist(),
        "cumulative_gate_share_by_k": by_k[
            :, metric_index(K_METRICS, "cumulative_gate_share")
        ].tolist(),
        "cumulative_contribution_norm_share_by_k": by_k[
            :, metric_index(K_METRICS, "cumulative_contribution_norm_share")
        ].tolist(),
        "counterfactual_cosine_by_k": by_k[
            :, metric_index(K_METRICS, "counterfactual_cosine")
        ].tolist(),
        "counterfactual_relative_l2_by_k": by_k[
            :, metric_index(K_METRICS, "counterfactual_relative_l2")
        ].tolist(),
        "counterfactual_norm_ratio_by_k": by_k[
            :, metric_index(K_METRICS, "counterfactual_norm_ratio")
        ].tolist(),
        "scalars": {name: float(scalar[index]) for index, name in enumerate(SCALAR_METRICS)},
    }


def bootstrap_over_examples(
    examples: list[dict[str, Any]], *, resamples: int = 2_000, seed: int = 20260716
) -> dict[str, Any]:
    """Bootstrap headline generation metrics with responses as the sampling unit."""
    phase_ids = [PHASES.index(phase) for phase in ("reasoning", "final")]
    counts = np.asarray(
        [example["count"][phase_ids].sum() for example in examples], dtype=np.float64
    )
    rank_sums = np.asarray(
        [example["rank_sum"][phase_ids].sum(axis=(0, 1)) for example in examples]
    )
    k_sums = np.asarray(
        [example["k_sum"][phase_ids].sum(axis=(0, 1)) for example in examples]
    )
    scalar_sums = np.asarray(
        [example["scalar_sum"][phase_ids].sum(axis=(0, 1)) for example in examples]
    )

    rng = np.random.default_rng(seed)
    draws = np.zeros((resamples, len(examples)), dtype=np.float64)
    for task in sorted({example["task"] for example in examples}):
        indices = [index for index, example in enumerate(examples) if example["task"] == task]
        draws[:, indices] = rng.multinomial(
            len(indices), np.full(len(indices), 1 / len(indices)), size=resamples
        )
    denominators = draws @ counts
    boot_rank = np.einsum("bn,nrm->brm", draws, rank_sums) / denominators[:, None, None]
    boot_k = np.einsum("bn,nkm->bkm", draws, k_sums) / denominators[:, None, None]
    boot_scalar = np.einsum("bn,nm->bm", draws, scalar_sums) / denominators[:, None]

    total = counts.sum()
    point_rank = rank_sums.sum(axis=0) / total
    point_k = k_sums.sum(axis=0) / total
    point_scalar = scalar_sums.sum(axis=0) / total
    rank_output_norm = metric_index(RANK_METRICS, "expert_output_norm")
    k_gate = metric_index(K_METRICS, "cumulative_gate_share")
    k_contribution = metric_index(K_METRICS, "cumulative_contribution_norm_share")
    k_cosine = metric_index(K_METRICS, "counterfactual_cosine")
    k_relative_l2 = metric_index(K_METRICS, "counterfactual_relative_l2")
    k_norm_ratio = metric_index(K_METRICS, "counterfactual_norm_ratio")

    values = {
        "rank8_activation_norm_relative_to_rank1": (
            point_rank[7, rank_output_norm] / point_rank[0, rank_output_norm],
            boot_rank[:, 7, rank_output_norm] / boot_rank[:, 0, rank_output_norm],
        ),
        "k4_gate_share": (point_k[3, k_gate], boot_k[:, 3, k_gate]),
        "k4_contribution_norm_share": (
            point_k[3, k_contribution],
            boot_k[:, 3, k_contribution],
        ),
        "k4_counterfactual_cosine": (point_k[3, k_cosine], boot_k[:, 3, k_cosine]),
        "k4_counterfactual_relative_l2": (
            point_k[3, k_relative_l2],
            boot_k[:, 3, k_relative_l2],
        ),
        "k4_counterfactual_norm_ratio": (
            point_k[3, k_norm_ratio],
            boot_k[:, 3, k_norm_ratio],
        ),
    }
    for name in (
        "moe_to_residual_norm_ratio",
        "coherence_ratio",
        "bottom4_contribution_norm_share",
        "bottom4_projection_share",
        "bottom4_top4_inversion_fraction",
        "bottom4_any_exceeds_top4",
        "gate_contribution_top4_overlap",
    ):
        index = metric_index(SCALAR_METRICS, name)
        values[name] = (point_scalar[index], boot_scalar[:, index])

    return {
        "sampling_unit": "response, stratified by task",
        "resamples": resamples,
        "seed": seed,
        "weighting_within_resample": "token-layer",
        "metrics": {
            name: {
                "estimate": float(point),
                "ci95": np.quantile(samples, (0.025, 0.975)).tolist(),
            }
            for name, (point, samples) in values.items()
        },
    }


def write_layer_csv(path: Path, stats: dict[str, np.ndarray]) -> None:
    count, rank, by_k, scalar = collapse(stats, ("reasoning", "final"), keep_layer=True)
    fields = ["layer", "token_count"]
    fields += [f"gate_rank{rank_id}_share" for rank_id in range(1, 9)]
    fields += [f"contribution_rank{rank_id}_norm_share" for rank_id in range(1, 9)]
    fields += [f"expert_rank{rank_id}_output_norm" for rank_id in range(1, 9)]
    for k in range(1, 9):
        fields += [
            f"k{k}_gate_share",
            f"k{k}_contribution_norm_share",
            f"k{k}_counterfactual_cosine",
            f"k{k}_counterfactual_relative_l2",
            f"k{k}_counterfactual_norm_ratio",
        ]
    fields += list(SCALAR_METRICS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in range(len(count)):
            row: dict[str, int | float] = {"layer": layer, "token_count": int(count[layer])}
            for rank_id in range(8):
                row[f"gate_rank{rank_id + 1}_share"] = rank[
                    layer, rank_id, metric_index(RANK_METRICS, "gate_weight")
                ]
                row[f"contribution_rank{rank_id + 1}_norm_share"] = rank[
                    layer, rank_id, metric_index(RANK_METRICS, "contribution_norm_share")
                ]
                row[f"expert_rank{rank_id + 1}_output_norm"] = rank[
                    layer, rank_id, metric_index(RANK_METRICS, "expert_output_norm")
                ]
            for k in range(8):
                row[f"k{k + 1}_gate_share"] = by_k[
                    layer, k, metric_index(K_METRICS, "cumulative_gate_share")
                ]
                row[f"k{k + 1}_contribution_norm_share"] = by_k[
                    layer, k, metric_index(K_METRICS, "cumulative_contribution_norm_share")
                ]
                row[f"k{k + 1}_counterfactual_cosine"] = by_k[
                    layer, k, metric_index(K_METRICS, "counterfactual_cosine")
                ]
                row[f"k{k + 1}_counterfactual_relative_l2"] = by_k[
                    layer, k, metric_index(K_METRICS, "counterfactual_relative_l2")
                ]
                row[f"k{k + 1}_counterfactual_norm_ratio"] = by_k[
                    layer, k, metric_index(K_METRICS, "counterfactual_norm_ratio")
                ]
            for index, name in enumerate(SCALAR_METRICS):
                row[name] = scalar[layer, index]
            writer.writerow(row)


def style_axis(axis: plt.Axes) -> None:
    axis.grid(True, alpha=0.22)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=210, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_by_rank(path: Path, stats: dict[str, np.ndarray]) -> None:
    _, rank, _, _ = collapse(stats, ("reasoning", "final"))
    ranks = np.arange(1, 9)
    gate = rank[:, metric_index(RANK_METRICS, "gate_weight")]
    contribution = rank[:, metric_index(RANK_METRICS, "contribution_norm_share")]
    output_norm = rank[:, metric_index(RANK_METRICS, "expert_output_norm")]
    contribution_rank = rank[:, metric_index(RANK_METRICS, "contribution_rank_position")]
    leave_one_out = rank[:, metric_index(RANK_METRICS, "leave_one_out_relative_l2")]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3))
    axes[0, 0].plot(ranks, gate * 100, marker="o", linewidth=2, label="Gate weight")
    axes[0, 0].plot(
        ranks,
        contribution * 100,
        marker="s",
        linewidth=2,
        label="Weighted contribution norm",
    )
    axes[0, 0].set(title="Mixture share", ylabel="Share within selected top 8 (%)")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(ranks, output_norm / output_norm[0], marker="o", linewidth=2)
    axes[0, 1].axhline(1, color="#777777", linestyle="--", linewidth=1)
    axes[0, 1].set(title="Unweighted expert activation size", ylabel="Norm / rank-1 norm")

    axes[1, 0].plot(ranks, contribution_rank, marker="o", linewidth=2)
    axes[1, 0].plot(ranks, ranks, color="#777777", linestyle="--", label="No reordering")
    axes[1, 0].set(
        title="Contribution ordering after activation size",
        xlabel="Router rank",
        ylabel="Mean contribution-norm rank",
    )
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(ranks, leave_one_out, marker="o", linewidth=2)
    axes[1, 1].set(
        title="Effect of dropping one expert and renormalizing",
        xlabel="Dropped router rank",
        ylabel="Relative L2 change in MoE output",
    )
    for axis in axes.flat:
        axis.set_xticks(ranks)
        style_axis(axis)
    fig.suptitle(
        "Qwen3-30B-A3B K=8: activation size versus routed contribution",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=2.0)
    save_figure(fig, path)


def plot_by_layer(path: Path, stats: dict[str, np.ndarray]) -> None:
    _, _, by_k, _ = collapse(stats, ("reasoning", "final"), keep_layer=True)
    layers = np.arange(by_k.shape[0])
    selected_k = (1, 2, 4, 6, 8)
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.9, len(selected_k)))
    panels = (
        ("cumulative_gate_share", "Gate mass retained", "Share"),
        ("cumulative_contribution_norm_share", "Contribution norm retained", "Share"),
        ("counterfactual_cosine", "Direction agreement with K=8 output", "Cosine"),
        ("counterfactual_relative_l2", "Difference from K=8 output", "Relative L2"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        metric_id = metric_index(K_METRICS, metric)
        for k, color in zip(selected_k, colors, strict=True):
            axis.plot(layers, by_k[:, k - 1, metric_id], color=color, label=f"K={k}")
        axis.set(title=title, ylabel=ylabel)
        axis.set_xlim(0, len(layers) - 1)
        style_axis(axis)
    axes[0, 0].legend(ncol=3, frameon=False)
    axes[1, 0].set_xlabel("MoE layer index")
    axes[1, 1].set_xlabel("MoE layer index")
    fig.suptitle(
        "Qwen3-30B-A3B: local top-K counterfactuals across layers",
        fontsize=15,
        fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.2, w_pad=2.0)
    save_figure(fig, path)


def main() -> None:
    args = parse_args()
    root = load_root_stats(args.results / "summary.npz")
    tasks = load_task_stats(args.results, root["count"].shape)
    examples = load_example_stats(args.results)
    payload = {
        "aggregation": "token-layer weighted",
        "phase_token_counts": {
            phase: int(root["count"][index].sum() // root["count"].shape[1])
            for index, phase in enumerate(PHASES)
        },
        "generation": compact_summary(root, ("reasoning", "final")),
        "prompt": compact_summary(root, ("prompt",)),
        "reasoning": compact_summary(root, ("reasoning",)),
        "final": compact_summary(root, ("final",)),
        "by_task_generation": {
            task: compact_summary(stats, ("reasoning", "final"))
            for task, stats in sorted(tasks.items())
        },
        "bootstrap_over_responses": bootstrap_over_examples(examples),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "qwen3_k8_expert_contribution_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    csv_path = args.output_dir / "qwen3_k8_expert_contribution_by_layer.csv"
    write_layer_csv(csv_path, root)
    plot_by_rank(args.plot_dir / "qwen3_k8_expert_contribution_by_rank.png", root)
    plot_by_layer(args.plot_dir / "qwen3_k8_expert_contribution_by_layer.png", root)
    print(f"wrote {summary_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()

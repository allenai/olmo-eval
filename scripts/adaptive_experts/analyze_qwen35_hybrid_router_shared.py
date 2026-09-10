#!/usr/bin/env python3
"""Summarize and plot Qwen3.5 router/shared-expert profiles by layer type."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "notes" / "qwen35_router_shared",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "generated", "prompt", "reasoning", "final"),
        default="all",
    )
    parser.add_argument(
        "--weighting",
        choices=("prompt_task_balanced", "token"),
        default="prompt_task_balanced",
        help="Balance prompts within tasks and tasks equally, or weight every token equally.",
    )
    return parser.parse_args()


def phase_indices(names: list[str], selection: str) -> list[int]:
    if selection == "all":
        wanted = names
    elif selection == "generated":
        wanted = ["reasoning", "final"]
    else:
        wanted = [selection]
    return [names.index(name) for name in wanted]


def combined_mean(data: np.lib.npyio.NpzFile, stem: str, phases: list[int]) -> np.ndarray:
    numerator = data[f"{stem}_sum"][phases].sum(axis=0)
    count = data["count"][phases].sum(axis=0)
    suffix_dims = numerator.ndim - count.ndim
    denominator = count[(...,) + (None,) * suffix_dims]
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def combine_example_family(row: dict[str, Any], family: str, phases: list[int]) -> np.ndarray:
    means = np.asarray(row[f"{family}_means"], dtype=np.float64)
    count = np.asarray(row["count"], dtype=np.float64)
    selected_means = means[phases]
    selected_count = count[phases]
    suffix_dims = selected_means.ndim - selected_count.ndim
    expanded_count = selected_count[(...,) + (None,) * suffix_dims]
    numerator = np.sum(selected_means * expanded_count, axis=0)
    denominator = np.sum(selected_count, axis=0)
    expanded_denominator = denominator[(...,) + (None,) * suffix_dims]
    return np.divide(
        numerator,
        expanded_denominator,
        out=np.full_like(numerator, np.nan),
        where=expanded_denominator > 0,
    )


def prompt_task_balanced_means(
    results: Path, phases: list[int]
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    by_task: dict[str, dict[str, list[np.ndarray]]] = {}
    prompt_counts: dict[str, int] = {}
    paths = sorted((results / "workers").glob("rank*/examples.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no per-example files under {results / 'workers'}")
    for path in paths:
        with gzip.open(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                task = row["task"]
                task_values = by_task.setdefault(
                    task,
                    {family: [] for family in ("router_rank", "router_k", "router_scalar", "branch_scalar")},
                )
                for family in task_values:
                    task_values[family].append(combine_example_family(row, family, phases))
                prompt_counts[task] = prompt_counts.get(task, 0) + 1
    balanced: dict[str, np.ndarray] = {}
    for family in ("router_rank", "router_k", "router_scalar", "branch_scalar"):
        task_means = [
            np.nanmean(np.stack(values[family]), axis=0)
            for values in by_task.values()
            if values[family]
        ]
        balanced[family] = np.nanmean(np.stack(task_means), axis=0)
    return balanced, prompt_counts


def metric_index(metadata: dict[str, Any], family: str, name: str) -> int:
    return metadata[f"{family}_metrics"].index(name)


def shade_full_attention(axis: plt.Axes, layer_types: list[str]) -> None:
    first = True
    for layer, layer_type in enumerate(layer_types, start=1):
        if layer_type == "full_attention":
            axis.axvspan(
                layer - 0.48,
                layer + 0.48,
                color="#f2c14e",
                alpha=0.16,
                linewidth=0,
                label="Full attention layer" if first else None,
                zorder=0,
            )
            first = False


def write_layer_csv(
    path: Path,
    *,
    layer_types: list[str],
    rank: np.ndarray,
    cumulative: np.ndarray,
    router: np.ndarray,
    branch: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    fields = ["layer", "layer_type"]
    fields += [f"selected_rank_{rank + 1}_weight" for rank in range(8)]
    fields += [f"selected_top_{rank + 1}_cumulative_share" for rank in range(8)]
    fields += metadata["router_scalar_metrics"]
    fields += metadata["branch_scalar_metrics"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer, layer_type in enumerate(layer_types):
            row: dict[str, Any] = {"layer": layer + 1, "layer_type": layer_type}
            row.update(
                {f"selected_rank_{i + 1}_weight": rank[layer, i, 1] for i in range(8)}
            )
            row.update(
                {
                    f"selected_top_{i + 1}_cumulative_share": cumulative[layer, i, 1]
                    for i in range(8)
                }
            )
            row.update(
                {
                    name: router[layer, i]
                    for i, name in enumerate(metadata["router_scalar_metrics"])
                }
            )
            row.update(
                {
                    name: branch[layer, i]
                    for i, name in enumerate(metadata["branch_scalar_metrics"])
                }
            )
            writer.writerow(row)


def layer_type_summary(
    *,
    layer_types: list[str],
    rank: np.ndarray,
    cumulative: np.ndarray,
    router: np.ndarray,
    branch: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer_type in ("linear_attention", "full_attention"):
        mask = np.asarray([value == layer_type for value in layer_types])
        values = {
            "layers": int(mask.sum()),
            "selected_rank_weights": np.nanmean(rank[mask, :, 1], axis=0).tolist(),
            "selected_cumulative_shares": np.nanmean(cumulative[mask, :, 1], axis=0).tolist(),
            "router": {
                name: float(np.nanmean(router[mask, i]))
                for i, name in enumerate(metadata["router_scalar_metrics"])
            },
            "shared_branch": {
                name: float(np.nanmean(branch[mask, i]))
                for i, name in enumerate(metadata["branch_scalar_metrics"])
            },
        }
        result[layer_type] = values

    linear = result["linear_attention"]
    full = result["full_attention"]
    result["full_minus_linear"] = {
        "selected_rank_weights": (
            np.asarray(full["selected_rank_weights"])
            - np.asarray(linear["selected_rank_weights"])
        ).tolist(),
        "selected_cumulative_shares": (
            np.asarray(full["selected_cumulative_shares"])
            - np.asarray(linear["selected_cumulative_shares"])
        ).tolist(),
        "router": {
            name: full["router"][name] - linear["router"][name]
            for name in full["router"]
        },
        "shared_branch": {
            name: full["shared_branch"][name] - linear["shared_branch"][name]
            for name in full["shared_branch"]
        },
    }
    return result


def make_plot(
    path: Path,
    *,
    phase: str,
    layer_types: list[str],
    rank: np.ndarray,
    cumulative: np.ndarray,
    router: np.ndarray,
    branch: np.ndarray,
    metadata: dict[str, Any],
    weighting: str,
    prompt_counts: dict[str, int] | None,
) -> None:
    layers = np.arange(1, len(layer_types) + 1)
    colors = plt.colormaps["viridis"](np.linspace(0.06, 0.9, 8))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)

    for k in (1, 2, 4, 6):
        axes[0, 0].plot(
            layers,
            100 * cumulative[:, k - 1, 1],
            color=colors[k - 1],
            linewidth=1.8,
            marker="o",
            markersize=2.5,
            label=f"Top {k}",
        )
    axes[0, 0].set_title("Cumulative share within native K=8")
    axes[0, 0].set_ylabel("Selected weight (%)")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].legend(ncol=2, frameon=False)

    for expert_rank in range(8):
        axes[0, 1].plot(
            layers,
            100 * rank[:, expert_rank, 1],
            color=colors[expert_rank],
            linewidth=1.5,
            label=f"Rank {expert_rank + 1}",
        )
    axes[0, 1].set_title("Individual selected-expert weights")
    axes[0, 1].set_ylabel("Selected weight (%)")
    axes[0, 1].legend(ncol=2, frameon=False, fontsize=8)

    raw_mass = router[:, metric_index(metadata, "router_scalar", "raw_top8_mass")]
    effective = router[:, metric_index(metadata, "router_scalar", "selected_effective_experts")]
    axes[1, 0].plot(layers, 100 * raw_mass, color="#3a7d44", linewidth=2, label="Raw top-8 mass")
    axes[1, 0].set_title("Router concentration against all 256 experts")
    axes[1, 0].set_ylabel("Raw softmax mass (%)", color="#3a7d44")
    axes[1, 0].tick_params(axis="y", labelcolor="#3a7d44")
    twin = axes[1, 0].twinx()
    twin.plot(layers, effective, color="#6a4c93", linewidth=1.8, label="Effective experts within K=8")
    twin.set_ylabel("Effective experts", color="#6a4c93")
    twin.tick_params(axis="y", labelcolor="#6a4c93")

    shared_metrics = (
        ("shared_gate", "Shared gate", "#0072b2"),
        ("shared_branch_norm_fraction", "Shared norm / branch-norm sum", "#d55e00"),
        ("shared_projection_share", "Shared projection share", "#009e73"),
    )
    for name, label, color in shared_metrics:
        values = branch[:, metric_index(metadata, "branch_scalar", name)]
        axes[1, 1].plot(layers, 100 * values, linewidth=1.8, color=color, label=label)
    axes[1, 1].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 1].set_title("Shared-expert contribution after gating")
    axes[1, 1].set_ylabel("Percent")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        shade_full_attention(axis, layer_types)
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlim(0.5, len(layer_types) + 0.5)
        axis.set_xticks([1, *range(4, len(layer_types) + 1, 4)])
    for axis in axes[1]:
        axis.set_xlabel("MoE layer (1-based; gold bands are full-attention layers)")

    fig.suptitle(
        f"Qwen3.5-35B-A3B native K=8 router and shared expert — {phase} tokens",
        x=0.06,
        ha="left",
        fontsize=15,
        fontweight="semibold",
    )
    if weighting == "prompt_task_balanced":
        counts = ", ".join(f"{task}={count}" for task, count in sorted((prompt_counts or {}).items()))
        subtitle = f"Prompt-balanced within task, then task-balanced ({counts})"
    else:
        subtitle = "Token-weighted across the complete saved trajectories"
    fig.text(0.06, 0.935, subtitle, fontsize=9.5, color="#555555")
    fig.tight_layout(rect=(0.03, 0.02, 0.99, 0.92), h_pad=2.2, w_pad=2.0)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.results / "summary.json").open() as handle:
        summary_metadata = json.load(handle)
    with (args.results / "metadata.json").open() as handle:
        run_metadata = json.load(handle)
    data = np.load(args.results / "summary.npz")
    selected_phases = phase_indices(summary_metadata["phases"], args.phase)
    prompt_counts = None
    if args.weighting == "prompt_task_balanced":
        balanced, prompt_counts = prompt_task_balanced_means(args.results, selected_phases)
        rank = balanced["router_rank"]
        cumulative = balanced["router_k"]
        router = balanced["router_scalar"]
        branch = balanced["branch_scalar"]
    else:
        rank = combined_mean(data, "router_rank", selected_phases)
        cumulative = combined_mean(data, "router_k", selected_phases)
        router = combined_mean(data, "router_scalar", selected_phases)
        branch = combined_mean(data, "branch_scalar", selected_phases)
    layer_types = run_metadata["layer_types"]

    stem = f"qwen35_k8_router_shared_by_layer_{args.phase}_{args.weighting}"
    write_layer_csv(
        args.output / f"{stem}.csv",
        layer_types=layer_types,
        rank=rank,
        cumulative=cumulative,
        router=router,
        branch=branch,
        metadata=summary_metadata,
    )
    comparison = layer_type_summary(
        layer_types=layer_types,
        rank=rank,
        cumulative=cumulative,
        router=router,
        branch=branch,
        metadata=summary_metadata,
    )
    comparison["analysis"] = {
        "phase": args.phase,
        "weighting": args.weighting,
        "prompt_counts": prompt_counts,
    }
    (args.output / f"qwen35_k8_attention_type_comparison_{args.phase}_{args.weighting}.json").write_text(
        json.dumps(comparison, indent=2)
    )
    make_plot(
        args.output / f"{stem}.png",
        phase=args.phase,
        layer_types=layer_types,
        rank=rank,
        cumulative=cumulative,
        router=router,
        branch=branch,
        metadata=summary_metadata,
        weighting=args.weighting,
        prompt_counts=prompt_counts,
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize and plot the Qwen3.6 above-native-K diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qwen36_expanded_k_diagnostic import METRICS, PHASES


def weighted_mean(values: np.ndarray, counts: np.ndarray, mask: np.ndarray) -> float:
    weights = counts * mask
    return float((values * weights).sum() / max(weights.sum(), 1))


def phase_layer_means(data: Any, metric: str, phase: str) -> np.ndarray:
    metric_index = METRICS.index(metric)
    phase_index = PHASES.index(phase)
    count = data["count"][phase_index]
    return data["metric_sum"][phase_index, :, metric_index] / np.maximum(count, 1)


def add_full_attention_bands(axis: plt.Axes, layer_types: list[str]) -> None:
    for layer, layer_type in enumerate(layer_types, start=1):
        if layer_type == "full_attention":
            axis.axvspan(layer - 0.5, layer + 0.5, color="#d9a400", alpha=0.09, lw=0)


def aggregate(data: Any, layer_types: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    layer_masks = {
        "all_layers": np.ones(len(layer_types), dtype=float),
        "linear_attention": np.asarray([x == "linear_attention" for x in layer_types], dtype=float),
        "full_attention": np.asarray([x == "full_attention" for x in layer_types], dtype=float),
    }
    phase_masks = {
        "all_tokens": np.ones(len(PHASES), dtype=float),
        **{phase: np.asarray([candidate == phase for candidate in PHASES], dtype=float) for phase in PHASES},
    }
    for phase_name, phase_mask in phase_masks.items():
        result[phase_name] = {}
        for layer_name, layer_mask in layer_masks.items():
            combined_mask = phase_mask[:, None] * layer_mask[None, :]
            counts = data["count"]
            values = data["metric_sum"] / np.maximum(counts[..., None], 1)
            result[phase_name][layer_name] = {
                metric: weighted_mean(values[..., index], counts, combined_mask)
                for index, metric in enumerate(METRICS)
            }
    return result


def render_markdown(summary: dict[str, Any], probes: list[dict[str, Any]]) -> str:
    lines = [
        "# Qwen3.6 above-native-K diagnostic results",
        "",
        "Gold bands in the plot mark full-attention layers; other layers use DeltaNet.",
        "",
        "| Tokens | Layers | raw top-8 mass | raw top-32 mass | ref K32 weight sum | normalized K32 top-8 share | tail/native routed norm | ref local delta | normalized local delta | K32-top8-only delta | native replay control | tail/top-8 cosine |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for phase in ("all_tokens", "prompt", "generated"):
        for layer_type in ("all_layers", "linear_attention", "full_attention"):
            row = summary[phase][layer_type]
            lines.append(
                "| "
                + " | ".join(
                    (
                        phase,
                        layer_type,
                        f"{100 * row['raw_top8_mass']:.2f}%",
                        f"{100 * row['raw_top32_mass']:.2f}%",
                        f"{row['reference_k32_weight_sum']:.3f}",
                        f"{100 * row['normalized_k32_top8_share']:.2f}%",
                        f"{100 * row['tail_to_native_routed_norm_ratio']:.2f}%",
                        f"{100 * row['reference_delta_relative_l2']:.2f}%",
                        f"{100 * row['normalized_delta_relative_l2']:.2f}%",
                        f"{100 * row['k32_top8_only_relative_l2']:.3f}%",
                        f"{100 * row['native_recompute_control_relative_l2']:.4f}%",
                        f"{row['tail_vs_k32_top8_cosine']:.3f}",
                    )
                )
                + " |"
            )
    lines.extend(("", "## End-to-end short probes", ""))
    for index, probe in enumerate(probes):
        lines.append(f"Worker {index}:")
        lines.append("")
        lines.append("| Policy | top-1 match | top-10 overlap | cosine | KL(native || policy) | mean abs logit delta |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for policy in ("native_repeat", "keep8", "reference32", "normalized32"):
            row = probe[policy]
            lines.append(
                f"| {policy} | {row['top1_match']} | {row['top10_overlap']}/10 | "
                f"{row['cosine']:.6f} | {row['native_to_candidate_kl']:.6f} | "
                f"{row['mean_abs_error']:.5f} |"
            )
        lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> None:
    data = np.load(args.results / "summary.npz")
    metadata = json.loads((args.results / "metadata.json").read_text())
    layer_types = metadata["layer_types"]
    layers = np.arange(1, len(layer_types) + 1)
    summary = aggregate(data, layer_types)
    probes = []
    for path in sorted((args.results / "workers").glob("rank*/metadata.json")):
        probes.append(json.loads(path.read_text())["end_to_end_probe"])

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "summary.md").write_text(render_markdown(summary, probes))

    colors = {"prompt": "#2468a2", "generated": "#d95f02"}
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for axis in axes.flat:
        add_full_attention_bands(axis, layer_types)
        axis.grid(alpha=0.22)

    for phase in PHASES:
        color = colors[phase]
        axes[0, 0].plot(
            layers,
            100 * phase_layer_means(data, "raw_top8_mass", phase),
            color=color,
            label=f"{phase}: top 8",
        )
        axes[0, 0].plot(
            layers,
            100 * phase_layer_means(data, "raw_top32_mass", phase),
            color=color,
            linestyle="--",
            label=f"{phase}: top 32",
        )
        axes[0, 1].plot(
            layers,
            phase_layer_means(data, "reference_k32_weight_sum", phase),
            color=color,
            label=f"{phase}: ref total weight",
        )
        axes[0, 1].plot(
            layers,
            phase_layer_means(data, "normalized_k32_top8_share", phase),
            color=color,
            linestyle="--",
            label=f"{phase}: normalized top-8 share",
        )
        axes[1, 0].plot(
            layers,
            100 * phase_layer_means(data, "reference_delta_relative_l2", phase),
            color=color,
            label=f"{phase}: reference-scaled",
        )
        axes[1, 0].plot(
            layers,
            100 * phase_layer_means(data, "normalized_delta_relative_l2", phase),
            color=color,
            linestyle="--",
            label=f"{phase}: normalized",
        )
        axes[1, 1].plot(
            layers,
            100 * phase_layer_means(data, "tail_to_native_routed_norm_ratio", phase),
            color=color,
            label=f"{phase}: tail/native routed",
        )
        axes[1, 1].plot(
            layers,
            100 * phase_layer_means(data, "shared_to_native_moe_norm_ratio", phase),
            color=color,
            linestyle="--",
            label=f"{phase}: shared/native MoE",
        )

    axes[0, 0].set_title("Raw all-256 router probability captured")
    axes[0, 0].set_ylabel("Probability mass (%)")
    axes[0, 1].set_title("What K=32 scaling does to the routed mixture")
    axes[0, 1].set_ylabel("Multiplier / share")
    axes[1, 0].set_title("Local K=32 MoE-update change vs native K=8")
    axes[1, 0].set_ylabel("Relative L2 change (%)")
    axes[1, 1].set_title("Tail and shared-branch magnitudes")
    axes[1, 1].set_ylabel("Norm ratio (%)")
    for axis in axes[1, :]:
        axis.set_xlabel("MoE layer (1-based)")
    for axis in axes.flat:
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Qwen3.6-35B-A3B · Native K=8 vs Expanded K=32 on Matched Tokens",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"qwen36_expanded_k_diagnostic.{suffix}", dpi=180)
    plt.close(fig)

    rank_fig, rank_axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    ranks = np.arange(1, 33)
    for phase_index, phase in enumerate(PHASES):
        count = data["count"][phase_index]
        mean_rank_probability = data["rank_probability_sum"][phase_index].sum(axis=0) / max(
            count.sum(), 1
        )
        cumulative = np.cumsum(mean_rank_probability)
        rank_axes[0].plot(ranks, 100 * cumulative, color=colors[phase], label=phase)
        rank_axes[1].plot(
            ranks,
            100 * cumulative / cumulative[-1],
            color=colors[phase],
            label=phase,
        )
    for axis in rank_axes:
        for selected_k in (8, 16, 32):
            axis.axvline(selected_k, color="#555555", alpha=0.3, linestyle=":")
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24, 28, 32])
        axis.set_xlabel("Router rank K")
        axis.grid(alpha=0.22)
        axis.legend()
    rank_axes[0].set_title("Cumulative raw all-256 router mass")
    rank_axes[0].set_ylabel("Raw probability mass (%)")
    rank_axes[1].set_title("Cumulative share within the selected top 32")
    rank_axes[1].set_ylabel("Share of top-32 mass (%)")
    rank_fig.suptitle(
        "Qwen3.6-35B-A3B · Router Mass by Rank on Matched Native-K=8 Tokens",
        fontsize=14,
        fontweight="bold",
    )
    rank_fig.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "svg"):
        rank_fig.savefig(args.output / f"qwen36_expanded_k_router_mass_curve.{suffix}", dpi=180)
    plt.close(rank_fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    analyze(parse_args())

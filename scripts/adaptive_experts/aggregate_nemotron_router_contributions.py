#!/usr/bin/env python3
"""Merge and summarize completed Nemotron router/contribution profile shards.

The profiler stores additive token-layer sufficient statistics in ``summary.npz``.
This script validates the four shard schemas, merges those statistics exactly, and
writes overall, phase, layer, rank, and Qwen-comparison tables and plots.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ARRAY_NAMES = (
    "count",
    "router_k_sum",
    "router_k_sumsq",
    "router_scalar_sum",
    "router_scalar_sumsq",
    "rank_sum",
    "rank_sumsq",
    "contribution_k_sum",
    "contribution_k_sumsq",
    "contribution_scalar_sum",
    "contribution_scalar_sumsq",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_examples(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_and_merge(
    inputs: list[Path],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict]]:
    if not inputs:
        raise ValueError("at least one input shard is required")
    schema_keys = (
        "phases",
        "k_values",
        "router_k_metrics",
        "router_scalar_metrics",
        "rank_metrics",
        "contribution_k_metrics",
        "contribution_scalar_metrics",
    )
    schema: dict[str, Any] | None = None
    merged: dict[str, np.ndarray] = {}
    examples: list[dict] = []
    seen_examples: set[tuple[str, str]] = set()
    seen_shards: set[int] = set()
    model: str | None = None
    layer_indices: list[int] | None = None

    for root in inputs:
        summary_path = root / "summary.json"
        npz_path = root / "summary.npz"
        metadata_path = root / "metadata.json"
        examples_path = root / "examples.jsonl.gz"
        for path in (summary_path, npz_path, metadata_path, examples_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        payload = load_json(summary_path)
        current_schema = {key: payload[key] for key in schema_keys}
        if schema is None:
            schema = current_schema
        elif schema != current_schema:
            raise ValueError(f"profile schema mismatch in {root}")

        metadata = load_json(metadata_path)
        shard_index = int(metadata["shard_index"])
        if shard_index in seen_shards:
            raise ValueError(f"duplicate shard index {shard_index}")
        seen_shards.add(shard_index)
        if model is None:
            model = metadata["model"]
            layer_indices = metadata["moe_model_layer_indices"]
        elif model != metadata["model"] or layer_indices != metadata["moe_model_layer_indices"]:
            raise ValueError(f"model/layer metadata mismatch in {root}")
        if metadata["native_k"] != 22 or metadata["routed_scaling_factor"] != 5.0:
            raise ValueError(f"unexpected routing configuration in {root}")
        validation = metadata["validation"]
        if validation["max_routed_reconstruction_relative_l2"] > 0.03:
            raise ValueError(f"routed reconstruction validation failed in {root}")
        if validation["max_total_reconstruction_relative_l2"] > 0.03:
            raise ValueError(f"total reconstruction validation failed in {root}")

        with np.load(npz_path) as arrays:
            if set(arrays.files) != set(ARRAY_NAMES):
                raise ValueError(f"unexpected arrays in {npz_path}: {arrays.files}")
            for name in ARRAY_NAMES:
                value = arrays[name]
                if name not in merged:
                    merged[name] = np.zeros_like(value)
                if merged[name].shape != value.shape:
                    raise ValueError(f"shape mismatch for {name} in {npz_path}")
                merged[name] += value

        rows = load_examples(examples_path)
        if len(rows) != int(metadata["examples"]):
            raise ValueError(f"example count mismatch in {root}")
        for row in rows:
            key = (str(row["task"]), str(row["doc_id"]))
            if key in seen_examples:
                raise ValueError(f"duplicate example {key}")
            seen_examples.add(key)
            examples.append(row)

    assert schema is not None and model is not None and layer_indices is not None
    expected_shards = set(range(len(inputs)))
    if seen_shards != expected_shards:
        raise ValueError(f"expected shard indices {expected_shards}, got {seen_shards}")
    if len(examples) != 60:
        raise ValueError(f"expected 60 examples, got {len(examples)}")
    task_counts: dict[str, int] = {}
    for row in examples:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1
    if set(task_counts.values()) != {20} or len(task_counts) != 3:
        raise ValueError(f"expected 20 examples for each of three tasks, got {task_counts}")

    schema.update(
        {
            "model": model,
            "layer_indices": layer_indices,
            "shard_indices": sorted(seen_shards),
            "example_count": len(examples),
            "task_counts": task_counts,
            "aggregation": "token-layer weighted additive sufficient statistics",
            "uncertainty_note": (
                "Stored summaries do not preserve per-example profile statistics; values are exact "
                "token-layer means, not prompt-bootstrap confidence intervals."
            ),
        }
    )
    return schema, merged, examples


def moments(
    arrays: dict[str, np.ndarray],
    name: str,
    *,
    reduce_axes: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = arrays["count"].sum(axis=reduce_axes)
    total = arrays[f"{name}_sum"].sum(axis=reduce_axes)
    total_sq = arrays[f"{name}_sumsq"].sum(axis=reduce_axes)
    extra_dims = total.ndim - count.ndim
    denominator = np.maximum(count, 1)[(...,) + (None,) * extra_dims]
    mean = total / denominator
    variance = np.maximum(total_sq / denominator - mean**2, 0)
    return mean, variance, count


def full_summary(schema: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    count = np.maximum(arrays["count"], 1)

    def one(name: str) -> dict[str, Any]:
        values = arrays[f"{name}_sum"]
        extra_dims = values.ndim - count.ndim
        denominator = count[(...,) + (None,) * extra_dims]
        mean = values / denominator
        variance = np.maximum(arrays[f"{name}_sumsq"] / denominator - mean**2, 0)
        return {"mean": mean.tolist(), "variance": variance.tolist()}

    return {
        **schema,
        "count": arrays["count"].tolist(),
        "router_k": one("router_k"),
        "router_scalar": one("router_scalar"),
        "rank": one("rank"),
        "contribution_k": one("contribution_k"),
        "contribution_scalar": one("contribution_scalar"),
    }


def metric_index(schema: dict[str, Any], family: str, name: str) -> int:
    return schema[f"{family}_metrics"].index(name)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def k_rows(
    schema: dict[str, Any],
    router: np.ndarray,
    contribution: np.ndarray,
    *,
    prefix: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    prefix = prefix or {}
    rows = []
    for position, k in enumerate(schema["k_values"]):
        row = {**prefix, "k": k}
        families = (
            ("router_k", router[position]),
            ("contribution_k", contribution[position]),
        )
        for family, values in families:
            for metric, value in zip(schema[f"{family}_metrics"], values, strict=True):
                row[metric] = float(value)
        rows.append(row)
    return rows


def rank_rows(
    schema: dict[str, Any], rank: np.ndarray, *, prefix: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    prefix = prefix or {}
    return [
        {
            **prefix,
            "rank": position + 1,
            **{
                metric: float(value)
                for metric, value in zip(schema["rank_metrics"], values, strict=True)
            },
        }
        for position, values in enumerate(rank)
    ]


def scalar_row(
    schema: dict[str, Any], router: np.ndarray, contribution: np.ndarray, **prefix: Any
) -> dict[str, Any]:
    return {
        **prefix,
        **{
            metric: float(value)
            for metric, value in zip(schema["router_scalar_metrics"], router, strict=True)
        },
        **{
            metric: float(value)
            for metric, value in zip(
                schema["contribution_scalar_metrics"], contribution, strict=True
            )
        },
    }


def qwen_comparison_rows(
    schema: dict[str, Any],
    overall_contribution_k: np.ndarray,
    qwen_path: Path | None,
) -> list[dict[str, Any]]:
    if qwen_path is None:
        return []
    qwen = load_json(qwen_path)["generation"]
    qwen_ks = list(range(1, 9))
    common_ks = sorted(set(qwen_ks) & set(schema["k_values"]))
    metrics = schema["contribution_k_metrics"]
    rows = []
    for k in common_ks:
        npos = schema["k_values"].index(k)
        qpos = qwen_ks.index(k)
        rows.extend(
            [
                {
                    "model": "Qwen3-30B-A3B",
                    "native_k": 8,
                    "k": k,
                    "gate_share": qwen["cumulative_gate_share_by_k"][qpos],
                    "contribution_norm_share": qwen[
                        "cumulative_contribution_norm_share_by_k"
                    ][qpos],
                    "reference_routed_cosine": "",
                    "renormalized_routed_cosine": qwen["counterfactual_cosine_by_k"][qpos],
                    "reference_total_cosine": "",
                    "renormalized_total_cosine": "",
                },
                {
                    "model": "Nemotron-3-Super",
                    "native_k": 22,
                    "k": k,
                    "gate_share": overall_contribution_k[
                        npos, metrics.index("cumulative_gate_share")
                    ],
                    "contribution_norm_share": overall_contribution_k[
                        npos, metrics.index("cumulative_contribution_norm_share")
                    ],
                    "reference_routed_cosine": overall_contribution_k[
                        npos, metrics.index("reference_routed_cosine")
                    ],
                    "renormalized_routed_cosine": overall_contribution_k[
                        npos, metrics.index("renormalized_routed_cosine")
                    ],
                    "reference_total_cosine": overall_contribution_k[
                        npos, metrics.index("reference_total_cosine")
                    ],
                    "renormalized_total_cosine": overall_contribution_k[
                        npos, metrics.index("renormalized_total_cosine")
                    ],
                },
            ]
        )
    return rows


def save_plots(
    output: Path,
    schema: dict[str, Any],
    overall_router_k: np.ndarray,
    overall_rank: np.ndarray,
    overall_contribution_k: np.ndarray,
    by_layer_contribution_k: np.ndarray,
) -> None:
    k = np.asarray(schema["k_values"])
    cm = schema["contribution_k_metrics"]
    rm = schema["router_k_metrics"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    axes[0].plot(
        k,
        overall_router_k[:, rm.index("selected_score_share")],
        marker="o",
        label="router score share",
    )
    axes[0].plot(
        k,
        overall_contribution_k[:, cm.index("cumulative_contribution_norm_share")],
        marker="s",
        label="contribution-norm share",
    )
    axes[0].set_ylabel("Cumulative share")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(frameon=False)

    for label, metric, style in (
        ("routed, reference preserving", "reference_routed_cosine", "-"),
        ("routed, renormalized", "renormalized_routed_cosine", "--"),
        ("routed+shared, reference preserving", "reference_total_cosine", "-"),
        ("routed+shared, renormalized", "renormalized_total_cosine", "--"),
    ):
        axes[1].plot(
            k,
            overall_contribution_k[:, cm.index(metric)],
            linestyle=style,
            marker="o",
            label=label,
        )
    axes[1].set_ylabel("Cosine to native K=22 update")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8)

    for label, metric, style in (
        ("routed, reference preserving", "reference_routed_relative_l2", "-"),
        ("routed, renormalized", "renormalized_routed_relative_l2", "--"),
        ("routed+shared, reference preserving", "reference_total_relative_l2", "-"),
        ("routed+shared, renormalized", "renormalized_total_relative_l2", "--"),
    ):
        axes[2].plot(
            k,
            overall_contribution_k[:, cm.index(metric)],
            linestyle=style,
            marker="o",
            label=label,
        )
    axes[2].set_ylabel("Relative L2 to native K=22 update")
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Retained routed experts")
        ax.set_xticks(k)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Nemotron-3-Super Router Mass and Local Contribution Counterfactuals",
        fontweight="bold",
    )
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            output / f"nemotron_router_contribution_overall.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    ranks = np.arange(1, overall_rank.shape[0] + 1)
    rank_metrics = schema["rank_metrics"]
    raw_norm = overall_rank[:, rank_metrics.index("expert_output_norm")]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(
        ranks,
        overall_rank[:, rank_metrics.index("gate_share")],
        marker="o",
        label="gate share",
    )
    ax.plot(
        ranks,
        overall_rank[:, rank_metrics.index("contribution_norm_share")],
        marker="s",
        label="contribution-norm share",
    )
    ax.plot(ranks, raw_norm / raw_norm[0], marker="^", label="raw output norm / rank 1")
    ax.set_xlabel("Selected router rank")
    ax.set_ylabel("Share / relative magnitude")
    ax.set_xticks(ranks)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Nemotron-3-Super Native K=22 Profile by Router Rank", fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            output / f"nemotron_router_contribution_by_rank.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    layers = np.asarray(schema["layer_indices"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for selected_k in (11, 17):
        pos = schema["k_values"].index(selected_k)
        axes[0].plot(
            layers,
            by_layer_contribution_k[:, pos, cm.index("cumulative_gate_share")],
            label=f"K={selected_k} gate share",
        )
        axes[0].plot(
            layers,
            by_layer_contribution_k[
                :, pos, cm.index("cumulative_contribution_norm_share")
            ],
            linestyle="--",
            label=f"K={selected_k} contribution share",
        )
        axes[1].plot(
            layers,
            by_layer_contribution_k[:, pos, cm.index("reference_total_cosine")],
            label=f"K={selected_k} reference preserving",
        )
        axes[1].plot(
            layers,
            by_layer_contribution_k[:, pos, cm.index("renormalized_total_cosine")],
            linestyle="--",
            label=f"K={selected_k} renormalized",
        )
    axes[0].set_ylabel("Cumulative share")
    axes[1].set_ylabel("Cosine to native routed+shared update")
    axes[1].set_xlabel("MoE layer index")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("Nemotron Layer-Resolved K=11 and K=17 Counterfactuals", fontweight="bold")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            output / f"nemotron_router_contribution_by_layer.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-summary", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    schema, arrays, examples = validate_and_merge(args.input)
    np.savez_compressed(args.output / "nemotron_router_contribution_merged.npz", **arrays)
    (args.output / "nemotron_router_contribution_merged.json").write_text(
        json.dumps(full_summary(schema, arrays))
    )

    overall_router_k, _, overall_count = moments(arrays, "router_k", reduce_axes=(0, 1))
    overall_router_scalar, _, _ = moments(arrays, "router_scalar", reduce_axes=(0, 1))
    overall_rank, _, _ = moments(arrays, "rank", reduce_axes=(0, 1))
    overall_contribution_k, _, _ = moments(arrays, "contribution_k", reduce_axes=(0, 1))
    overall_contribution_scalar, _, _ = moments(
        arrays, "contribution_scalar", reduce_axes=(0, 1)
    )
    by_layer_router_k, _, by_layer_count = moments(arrays, "router_k", reduce_axes=(0,))
    by_layer_router_scalar, _, _ = moments(arrays, "router_scalar", reduce_axes=(0,))
    by_layer_rank, _, _ = moments(arrays, "rank", reduce_axes=(0,))
    by_layer_contribution_k, _, _ = moments(arrays, "contribution_k", reduce_axes=(0,))
    by_layer_contribution_scalar, _, _ = moments(
        arrays, "contribution_scalar", reduce_axes=(0,)
    )
    by_phase_router_k, _, by_phase_count = moments(arrays, "router_k", reduce_axes=(1,))
    by_phase_router_scalar, _, _ = moments(arrays, "router_scalar", reduce_axes=(1,))
    by_phase_rank, _, _ = moments(arrays, "rank", reduce_axes=(1,))
    by_phase_contribution_k, _, _ = moments(arrays, "contribution_k", reduce_axes=(1,))
    by_phase_contribution_scalar, _, _ = moments(
        arrays, "contribution_scalar", reduce_axes=(1,)
    )

    k_fields = ["k", *schema["router_k_metrics"], *schema["contribution_k_metrics"]]
    rank_fields = ["rank", *schema["rank_metrics"]]
    scalar_fields = [
        "token_layer_count",
        *schema["router_scalar_metrics"],
        *schema["contribution_scalar_metrics"],
    ]
    write_csv(
        args.output / "nemotron_router_contribution_by_k.csv",
        k_fields,
        k_rows(schema, overall_router_k, overall_contribution_k),
    )
    write_csv(
        args.output / "nemotron_router_contribution_by_rank.csv",
        rank_fields,
        rank_rows(schema, overall_rank),
    )
    write_csv(
        args.output / "nemotron_router_contribution_scalars.csv",
        scalar_fields,
        [
            scalar_row(
                schema,
                overall_router_scalar,
                overall_contribution_scalar,
                token_layer_count=int(overall_count),
            )
        ],
    )

    layer_k_rows: list[dict[str, Any]] = []
    layer_rank_rows: list[dict[str, Any]] = []
    layer_scalar_rows: list[dict[str, Any]] = []
    for position, layer in enumerate(schema["layer_indices"]):
        prefix = {"layer": layer, "token_count": int(by_layer_count[position])}
        layer_k_rows.extend(
            k_rows(
                schema,
                by_layer_router_k[position],
                by_layer_contribution_k[position],
                prefix=prefix,
            )
        )
        layer_rank_rows.extend(rank_rows(schema, by_layer_rank[position], prefix=prefix))
        layer_scalar_rows.append(
            scalar_row(
                schema,
                by_layer_router_scalar[position],
                by_layer_contribution_scalar[position],
                **prefix,
            )
        )
    write_csv(
        args.output / "nemotron_router_contribution_by_layer_k.csv",
        ["layer", "token_count", *k_fields],
        layer_k_rows,
    )
    write_csv(
        args.output / "nemotron_router_contribution_by_layer_rank.csv",
        ["layer", "token_count", *rank_fields],
        layer_rank_rows,
    )
    write_csv(
        args.output / "nemotron_router_contribution_by_layer_scalars.csv",
        [
            "layer",
            "token_count",
            *schema["router_scalar_metrics"],
            *schema["contribution_scalar_metrics"],
        ],
        layer_scalar_rows,
    )

    phase_k_rows: list[dict[str, Any]] = []
    phase_rank_rows: list[dict[str, Any]] = []
    phase_scalar_rows: list[dict[str, Any]] = []
    for position, phase in enumerate(schema["phases"]):
        prefix = {"phase": phase, "token_layer_count": int(by_phase_count[position])}
        phase_k_rows.extend(
            k_rows(
                schema,
                by_phase_router_k[position],
                by_phase_contribution_k[position],
                prefix=prefix,
            )
        )
        phase_rank_rows.extend(rank_rows(schema, by_phase_rank[position], prefix=prefix))
        phase_scalar_rows.append(
            scalar_row(
                schema,
                by_phase_router_scalar[position],
                by_phase_contribution_scalar[position],
                **prefix,
            )
        )
    write_csv(
        args.output / "nemotron_router_contribution_by_phase_k.csv",
        ["phase", "token_layer_count", *k_fields],
        phase_k_rows,
    )
    write_csv(
        args.output / "nemotron_router_contribution_by_phase_rank.csv",
        ["phase", "token_layer_count", *rank_fields],
        phase_rank_rows,
    )
    write_csv(
        args.output / "nemotron_router_contribution_by_phase_scalars.csv",
        ["phase", *scalar_fields],
        phase_scalar_rows,
    )

    comparison = qwen_comparison_rows(schema, overall_contribution_k, args.qwen_summary)
    if comparison:
        write_csv(
            args.output / "qwen_nemotron_common_k_comparison.csv",
            list(comparison[0]),
            comparison,
        )

    overview = {
        **schema,
        "profiled_prompt_tokens": int(sum(row["prompt_tokens"] for row in examples)),
        "profiled_generated_tokens": int(sum(row["generated_tokens"] for row in examples)),
        "overall": scalar_row(
            schema,
            overall_router_scalar,
            overall_contribution_scalar,
            token_layer_count=int(overall_count),
        ),
        "by_k": k_rows(schema, overall_router_k, overall_contribution_k),
        "by_rank": rank_rows(schema, overall_rank),
    }
    (args.output / "nemotron_router_contribution_overview.json").write_text(
        json.dumps(overview, indent=2)
    )
    save_plots(
        args.output,
        schema,
        overall_router_k,
        overall_rank,
        overall_contribution_k,
        by_layer_contribution_k,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "shards": len(args.input),
                "examples": len(examples),
                "token_layer_count": int(overall_count),
                "files": len(list(args.output.iterdir())),
            }
        )
    )


if __name__ == "__main__":
    main()

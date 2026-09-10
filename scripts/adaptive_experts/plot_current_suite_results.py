#!/usr/bin/env python3
"""Plot the current four-eval Qwen and GPT-OSS expert-count sweeps."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from plot_results import COLORS, GPTOSS_K, GPTOSS_SCORES, POSTTRAIN_K, POSTTRAIN_SCORES

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "adaptive_experts"
LEDGER = ROOT / "notes" / "beaker_jobs.jsonl"
OUTPUT = ROOT / "notes" / "plots"
CSV_PATH = ROOT / "notes" / "current_suite_expert_sweeps.csv"

QWEN_NORMALIZED_GROUP = "adaptive-experts-qwen-normalized-capability-backfill-20260717"
QWEN_REFERENCE_GROUP = "adaptive-experts-qwen-reference-scaled-expanded-20260717"
QWEN_CAPABILITY_GROUP = "adaptive-experts-qwen-capability-policies-20260717"
QWEN_IFBENCH32K_GROUP = "adaptive-experts-qwen-capability-ifbench32k-20260717"
QWEN_ROUTING_GROUP = "adaptive-experts-qwen-routing-policies-20260717"
GPT_NORMALIZED_GROUP = "adaptive-experts-gptoss-normalized-capability-backfill-20260717"
GPT_REFERENCE_GROUP = "adaptive-experts-gptoss-reference-scaled-expanded-20260717"
DOLCI_CORRECTED_GROUP = "adaptive-experts-qwen-dolci-terminal-eos-v2-fullsuite-20260719"

METRICS = ("MATH-500 ↑", "GPQA Diamond ↑", "IFBench macro (32k) ↑", "HumanEval pass@1 ↑")
STYLES = {
    "MATH-500 ↑": (COLORS["blue"], "o"),
    "GPQA Diamond ↑": (COLORS["orange"], "s"),
    "IFBench macro (32k) ↑": (COLORS["green"], "^"),
    "HumanEval pass@1 ↑": (COLORS["vermillion"], "D"),
}


@dataclass(frozen=True)
class Stat:
    mean: float
    sd: float
    n: int


def read_ledger() -> list[dict]:
    with LEDGER.open() as f:
        return [json.loads(line) for line in f if line.strip()]


LEDGER_ROWS = read_ledger()


def result_path(experiment_id: str) -> Path | None:
    candidates = list((RESULTS / experiment_id).rglob("metrics.json"))
    return candidates[0] if candidates else None


def valid_ids(group: str, pattern: str) -> list[str]:
    regex = re.compile(pattern)
    ids = []
    for row in LEDGER_ROWS:
        if row.get("group") != group or not regex.search(row.get("run_tag", "")):
            continue
        experiment_id = row.get("beaker_experiment_id")
        if experiment_id and result_path(experiment_id) is not None:
            ids.append(experiment_id)
    return ids


def grouped_ids(group: str, pattern: str) -> dict[int, list[str]]:
    regex = re.compile(pattern)
    grouped: dict[int, list[str]] = {}
    for row in LEDGER_ROWS:
        if row.get("group") != group:
            continue
        match = regex.search(row.get("run_tag", ""))
        if match is None:
            continue
        experiment_id = row.get("beaker_experiment_id")
        if experiment_id and result_path(experiment_id) is not None:
            grouped.setdefault(int(match.group(1)), []).append(experiment_id)
    return grouped


def read_metrics(experiment_id: str) -> dict:
    path = result_path(experiment_id)
    if path is None:
        raise FileNotFoundError(experiment_id)
    with path.open() as f:
        payload = json.load(f)
    if payload["errors"]:
        raise RuntimeError(f"{experiment_id} has metric errors: {payload['errors']}")
    return payload


def score(experiment_id: str, metric: str) -> float:
    tasks = {task["task"]: task for task in read_metrics(experiment_id)["tasks"]}
    if metric == "MATH-500 ↑":
        return 100 * tasks["math500:chat"]["metrics"]["accuracy"]["minerva_math_flex"]
    if metric == "GPQA Diamond ↑":
        return 100 * tasks["gpqa_diamond:qwen3_thinking"]["metrics"]["accuracy"]["multiple_choice"]
    if metric == "HumanEval pass@1 ↑":
        return 100 * tasks["humaneval:chat:pass_at_1:qwen3_thinking"]["metrics"]["pass_at_1"]["code_exec"]
    if metric == "IFBench macro (32k) ↑":
        names = (
            "ifeval_ood",
            "ifeval_mt_wildchat_unused_withRewrite",
            "ifeval_mt_ood_wildchat_unused_withRewrite",
        )
        return 100 * float(
            np.mean([tasks[name]["metrics"]["prompt_level_loose_acc"]["ifeval"] for name in names])
        )
    raise KeyError(metric)


def summarize(values: list[float]) -> Stat:
    array = np.asarray(values, dtype=float)
    return Stat(float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0, len(array))


def summarize_ids(experiment_ids: list[str], metric: str) -> Stat:
    return summarize([score(experiment_id, metric) for experiment_id in experiment_ids])


def existing_stat(scores: dict, expert_counts: np.ndarray, k: int, metric: str, n: int = 3) -> Stat:
    mean, sd = scores[metric]
    index = int(np.where(expert_counts == k)[0][0])
    return Stat(float(mean[index]), float(sd[index]), n)


def qwen_normalized() -> dict[int, dict[str, Stat]]:
    backfill = grouped_ids(QWEN_NORMALIZED_GROUP, r"qwen-normalized-k(\d+)-r\d+")
    output: dict[int, dict[str, Stat]] = {}
    for k in range(3, 17):
        n_core = 2 if k in {13, 14, 15} else 3
        output[k] = {
            "MATH-500 ↑": existing_stat(POSTTRAIN_SCORES, POSTTRAIN_K, k, "MATH-500 ↑", n_core),
            "GPQA Diamond ↑": existing_stat(POSTTRAIN_SCORES, POSTTRAIN_K, k, "GPQA Diamond ↑", n_core),
        }
        if k == 4:
            ifbench_ids = valid_ids(QWEN_IFBENCH32K_GROUP, r"capability32k-normalized-k4-r\d+")
            human_ids = valid_ids(QWEN_CAPABILITY_GROUP, r"capability-normalized-k4-r\d+")
        elif k == 8:
            ifbench_ids = valid_ids(QWEN_IFBENCH32K_GROUP, r"capability32k-native-k8-r\d+")
            human_ids = valid_ids(QWEN_CAPABILITY_GROUP, r"capability-native-k8-r\d+")
        else:
            ifbench_ids = human_ids = backfill[k]
        output[k]["IFBench macro (32k) ↑"] = summarize_ids(ifbench_ids, "IFBench macro (32k) ↑")
        output[k]["HumanEval pass@1 ↑"] = summarize_ids(human_ids, "HumanEval pass@1 ↑")
    return output


def qwen_reference(normalized: dict[int, dict[str, Stat]]) -> dict[int, dict[str, Stat]]:
    full = grouped_ids(QWEN_REFERENCE_GROUP, r"qwen-reference-k(\d+)-r\d+")
    output: dict[int, dict[str, Stat]] = {}
    for k in range(3, 17):
        if k == 8:
            output[k] = normalized[k]
            continue
        if k == 4:
            math_gpqa_ids = valid_ids(QWEN_ROUTING_GROUP, r"routing-reference-k4-r\d+")
            ifbench_ids = valid_ids(QWEN_IFBENCH32K_GROUP, r"capability32k-reference-k4-r\d+")
            human_ids = valid_ids(QWEN_CAPABILITY_GROUP, r"capability-reference-k4-r\d+")
            output[k] = {
                "MATH-500 ↑": summarize_ids(math_gpqa_ids, "MATH-500 ↑"),
                "GPQA Diamond ↑": summarize_ids(math_gpqa_ids, "GPQA Diamond ↑"),
                "IFBench macro (32k) ↑": summarize_ids(ifbench_ids, "IFBench macro (32k) ↑"),
                "HumanEval pass@1 ↑": summarize_ids(human_ids, "HumanEval pass@1 ↑"),
            }
            continue
        output[k] = {metric: summarize_ids(full[k], metric) for metric in METRICS}
    return output


def gpt_normalized() -> dict[int, dict[str, Stat]]:
    backfill = grouped_ids(GPT_NORMALIZED_GROUP, r"gptoss-normalized-k(\d+)-r\d+-timeout1800")
    output: dict[int, dict[str, Stat]] = {}
    for k, experiment_ids in sorted(backfill.items()):
        output[k] = {
            "MATH-500 ↑": existing_stat(GPTOSS_SCORES, GPTOSS_K, k, "MATH-500 ↑"),
            "GPQA Diamond ↑": existing_stat(GPTOSS_SCORES, GPTOSS_K, k, "GPQA Diamond ↑"),
            "IFBench macro (32k) ↑": summarize_ids(experiment_ids, "IFBench macro (32k) ↑"),
            "HumanEval pass@1 ↑": summarize_ids(experiment_ids, "HumanEval pass@1 ↑"),
        }
    return output


def gpt_reference(normalized: dict[int, dict[str, Stat]]) -> dict[int, dict[str, Stat]]:
    full = grouped_ids(GPT_REFERENCE_GROUP, r"gptoss-reference-k(\d+)-r\d+-timeout1800")
    output = {k: {metric: summarize_ids(ids, metric) for metric in METRICS} for k, ids in full.items()}
    if 4 in normalized:
        output[4] = normalized[4]
    return dict(sorted(output.items()))


def corrected_dolci(pattern: str) -> dict[int, dict[str, Stat]]:
    completed = grouped_ids(DOLCI_CORRECTED_GROUP, pattern)
    return {
        k: {metric: summarize_ids(experiment_ids, metric) for metric in METRICS}
        for k, experiment_ids in sorted(completed.items())
    }


def write_csv(models: dict[str, dict[str, dict[int, dict[str, Stat]]]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("model", "routing", "k", "metric", "n", "mean", "sd"))
        writer.writeheader()
        for model, modes in models.items():
            for mode, points in modes.items():
                for k, metrics in points.items():
                    for metric, stat in metrics.items():
                        writer.writerow(
                            {
                                "model": model,
                                "routing": mode,
                                "k": k,
                                "metric": metric,
                                "n": stat.n,
                                "mean": f"{stat.mean:.6f}",
                                "sd": f"{stat.sd:.6f}",
                            }
                        )


def plot_model(
    model: str,
    title: str,
    sample_note: str,
    default_k: int,
    normalized: dict[int, dict[str, Stat]],
    reference: dict[int, dict[str, Stat]],
    stem: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), sharey=True)
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.99)
    fig.text(
        0.5,
        0.945,
        "Current suite: MATH-500, GPQA Diamond, IFBench macro at 32k, and standard HumanEval; "
        f"shading is ±1 sample SD\n{sample_note}",
        ha="center",
        va="top",
        color="#555A63",
        fontsize=10.2,
    )
    for ax, panel_title, points in zip(
        axes,
        ("Normalized top-K", f"K={default_k}-reference scaled (not renormalized)"),
        (normalized, reference),
        strict=True,
    ):
        ks = np.asarray(sorted(points))
        for metric in METRICS:
            color, marker = STYLES[metric]
            means = np.asarray([points[k][metric].mean for k in ks])
            sds = np.asarray([points[k][metric].sd for k in ks])
            ax.fill_between(ks, np.maximum(0, means - sds), np.minimum(100, means + sds), color=color, alpha=0.13)
            ax.plot(ks, means, color=color, marker=marker, linewidth=2.2, markersize=6, label=metric)
        padding = max(0.35, (ks.max() - ks.min()) * 0.04)
        ax.set_xlim(ks.min() - padding, ks.max() + padding)
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylim(0, 102)
        ax.set_title(panel_title, fontweight="semibold", pad=10)
        ax.set_xlabel("Active experts per token", fontweight="semibold")
        ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.8)
        ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.6)
        if ks.min() <= default_k <= ks.max():
            ax.axvline(default_k, color="#73777F", linestyle="--", linewidth=1.1, alpha=0.7, zorder=0)
    axes[0].set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.79, wspace=0.08)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", dpi=220 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    qwen_norm = qwen_normalized()
    qwen_ref = qwen_reference(qwen_norm)
    gpt_norm = gpt_normalized()
    gpt_ref = gpt_reference(gpt_norm)
    dolci_norm = corrected_dolci(r"dolci-v2-norm-k(\d+)-r\d+")
    dolci_ref = corrected_dolci(r"dolci-v2-ref-k(\d+)-r\d+")
    models = {
        "Qwen3-30B-A3B hybrid thinking": {"normalized": qwen_norm, "reference_scaled": qwen_ref},
        "GPT-OSS-120B": {"normalized": gpt_norm, "reference_scaled": gpt_ref},
        "Qwen3-30B-A3B corrected Dolci/OLMo-3 SFT": {
            "normalized": dolci_norm,
            "reference_scaled": dolci_ref,
        },
    }
    write_csv(models)
    plot_model(
        "Qwen3-30B-A3B hybrid thinking",
        "Qwen3-30B-A3B Hybrid Thinking: Current-Suite Performance vs Active Experts",
        "MATH/GPQA use n=2 at K=13–15; every other metric point uses n=3",
        8,
        qwen_norm,
        qwen_ref,
        "qwen3_hybrid_current_suite_expert_sweep",
    )
    plot_model(
        "GPT-OSS-120B",
        "GPT-OSS-120B: Current-Suite Performance vs Active Experts",
        "Every point is the mean of three complete-evaluation replicates",
        4,
        gpt_norm,
        gpt_ref,
        "gptoss_120b_current_suite_expert_sweep",
    )
    plot_model(
        "Qwen3-30B-A3B corrected Dolci/OLMo-3 SFT",
        "Corrected Qwen/Dolci OLMo-3 SFT: Current-Suite Performance",
        "Every point is the mean of three complete-evaluation replicates",
        8,
        dolci_norm,
        dolci_ref,
        "qwen3_dolci_terminal_eos_v2_current_suite_expert_sweep",
    )


if __name__ == "__main__":
    main()

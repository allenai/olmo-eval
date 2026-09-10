#!/usr/bin/env python3
"""Plot raw replicate ranges for the original Qwen3 hybrid-thinking sweep."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "adaptive_experts"
OUTPUT_DIR = REPO_ROOT / "notes" / "plots"
CSV_PATH = REPO_ROOT / "notes" / "qwen3_posttraining_replicate_ranges.csv"

# These are the same curated runs used for the original mean/SD Qwen3 plot. The
# earlier short-cap pilot runs are intentionally excluded. K=13--15 have n=2;
# every other K has n=3.
RUN_IDS = {
    1: ("01KX56NQMVH3TVZKGAM9DPGF05", "01KX6KNJBM5KRF349TF72QWCFY", "01KX6KQHSH0PYEYP9Q69FNAEZ4"),
    2: ("01KX56NZQVWQBKXH6VP6CXV345", "01KX6KNTYNVNE16WXK0DNTGGP5", "01KX6KQTXKYB6ENVAV2Q30KDMS"),
    3: ("01KX7GFW9RCAPT76WTTBRWT0SR", "01KX7GDJXSF8XTP79Y5WK0BCPE", "01KX7GH43M5RXNRJPPTJN62TT1"),
    4: ("01KX56P8M9TD1MPMCT582AXZ87", "01KX6KP4V9Z43MR9RJPD4YV6JN", "01KX6KR3YH8YET0JZACDSQA8XP"),
    5: ("01KX6KSNE491THB9DT4Y9PPJ7W", "01KX6KTWW70DP1684C2CR10TCH", "01KX6KW6Z0GV5KHQTAMP1KA35J"),
    6: ("01KX6KSYGH7H8RQPEGJFYC3PEW", "01KX6KV5MDA3AAG9ET53W91FEE", "01KX6KWKTN48AHZPYN5A9GEPFS"),
    7: ("01KX6KT77JRKJSW67J2NY5BFQQ", "01KX6KVEM81QYVVBJZK6JTTS0V", "01KX6KX1QMYC8KEN1FXV56S2TZ"),
    8: ("01KX56PH4KDPVQFSBB3ZHDEMB7", "01KX6KPE77ANFQHN4D4MP1VSEZ", "01KX6KRCDK4ZSA4VZ4TABXG3CD"),
    9: ("01KX7GG47N7XP7K8NYW9NE7B73", "01KX7GDX9FWDR43N8SDD232ZZK", "01KX7GHBWTRAJ5DVSAYASRDHTD"),
    10: ("01KX7GGCK1WZX5XKEW3VHHDCB4", "01KX7GE5PFF8FJX62HHVS9Z0B9", "01KX7GHKQNRSH6JP8QX6A449QF"),
    11: ("01KX7GGMF6J8VV1YJ8F1REKFQ3", "01KX7GEK926AYKMV7R7JE5AP1N", "01KX7GHVSSYXR16CSZ8WW8X997"),
    12: ("01KX7GGVVD8NQS8W2RFNAWNWXS", "01KX7GF02PRGQR0SNF23BX230K", "01KX7GJ4S0513935Q6P96HD3CE"),
    13: ("01KX7VBBDJDTWW1YK3010QB7R0", "01KX7VCEF5H37P48MBQ2QY8R3R"),
    14: ("01KX7VBM2Z141G67RM7GB8C0R9", "01KX7VCQJ441PPKSMCJ5TD9P1Q"),
    15: ("01KX7VBW0H13C928JGVQ9D3GV1", "01KX7VD8PC4VMCWSM9VYMEJGMB"),
    16: ("01KX56PSK9WJMQ68BTQ3P54P5D", "01KX6KPPW8A7PPF9GKT8VT7PXQ", "01KX6KRMQ39F6SYMDM12KSJJBP"),
    32: ("01KX5ACWAEQSRM0E54YFBB4Z8H", "01KX6KPZFHG49NF7MBQX0QGXFP", "01KX6KRXF9V7TER9WWTMPCNFHZ"),
}

TASK_METRICS = {
    "MATH-500 ↑": ("math500:chat", "accuracy", "minerva_math_flex"),
    "GPQA Diamond ↑": ("gpqa_diamond:qwen3_thinking", "accuracy", "multiple_choice"),
    "IFEval OOD ↑": ("ifeval_ood:qwen3_thinking", "prompt_level_loose_acc", "ifeval"),
}

STYLES = {
    "MATH-500 ↑": ("#0072B2", "o"),
    "GPQA Diamond ↑": ("#E69F00", "s"),
    "IFEval OOD ↑": ("#009E73", "^"),
    "Macro average ↑": ("#D55E00", "D"),
}


def load_run(experiment_id: str) -> dict[str, float]:
    path = RESULTS_DIR / experiment_id / "metrics.json"
    with path.open() as f:
        payload = json.load(f)
    if payload["errors"]:
        raise RuntimeError(f"{experiment_id} has errors: {payload['errors']}")

    tasks = {task["task"]: task for task in payload["tasks"]}
    values = {}
    for label, (task_name, metric, scorer) in TASK_METRICS.items():
        values[label] = 100.0 * tasks[task_name]["metrics"][metric][scorer]
    values["Macro average ↑"] = float(np.mean(list(values.values())))
    return values


def collect() -> dict[str, dict[int, list[float]]]:
    values: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for k, experiment_ids in RUN_IDS.items():
        for experiment_id in experiment_ids:
            for label, score in load_run(experiment_id).items():
                values[label][k].append(score)
    return values


def write_summary(values: dict[str, dict[int, list[float]]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("metric", "k", "n", "mean", "min", "max", "range"))
        writer.writeheader()
        for label in STYLES:
            for k in RUN_IDS:
                samples = np.asarray(values[label][k])
                writer.writerow(
                    {
                        "metric": label,
                        "k": k,
                        "n": len(samples),
                        "mean": f"{samples.mean():.6f}",
                        "min": f"{samples.min():.6f}",
                        "max": f"{samples.max():.6f}",
                        "range": f"{np.ptp(samples):.6f}",
                    }
                )


def plot(values: dict[str, dict[int, list[float]]]) -> None:
    ks = np.asarray(list(RUN_IDS))
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    fig.suptitle(
        "Qwen3-30B-A3B Hybrid Thinking: Replicate Variation vs Active Experts",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Shading is the observed min–max range; dotted lines are replicate means "
        "(n=3 except K=13–15, n=2)",
        ha="center",
        va="top",
        color="#555A63",
        fontsize=10.5,
    )

    for label, (color, marker) in STYLES.items():
        samples = [np.asarray(values[label][k]) for k in ks]
        means = np.asarray([sample.mean() for sample in samples])
        minima = np.asarray([sample.min() for sample in samples])
        maxima = np.asarray([sample.max() for sample in samples])
        ax.fill_between(ks, minima, maxima, color=color, alpha=0.17, linewidth=0)
        ax.plot(
            ks,
            means,
            label=label,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.3,
            linestyle=(0, (2.0, 2.0)),
        )

    padding = max(0.5, (ks.max() - ks.min()) * 0.03)
    ax.set_xlim(ks.min() - padding, ks.max() + padding)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_ylim(0, 102)
    ax.set_xlabel("Active experts per token", fontweight="semibold")
    ax.set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.6)
    ax.axvline(8, color="#73777F", linestyle="--", linewidth=1.2, alpha=0.75, zorder=0)
    ax.text(8, 3.0, "default K=8", rotation=90, ha="right", va="bottom", color="#666B73", fontsize=9)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.6,
    )

    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.87)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(
            OUTPUT_DIR / f"posttraining_expert_sweep_replicate_range.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def main() -> None:
    values = collect()
    write_summary(values)
    plot(values)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot completed Qwen3.6 TBLite and TB2.1 TMax-default expert sweeps.

The manifests below point only at the validated 2026-08-15 protocol: native
TMax task timeouts, Qwen thinking-mode generation settings, and corrected
nested router overrides above K=8. A run is emitted only when its complete
100-task TBLite or 89-task TB2.1 metric file exists. In-progress and invalid
historical K>8 runs therefore cannot enter either CSV or plot.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


OLMO_EVAL_ROOT = Path(__file__).resolve().parents[2]
NOTES = OLMO_EVAL_ROOT / "notes"
PLOTS = NOTES / "plots"
TBLITE_RESULTS = Path(
    "/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-reference-sweep-20260815"
)
TBLITE_EXPANDED_RESULTS = Path(
    "/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-expanded-reference-sweep-20260816"
)
TBLITE_EXTREME_RESULTS = Path(
    "/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-extreme-reference-sweep-20260816"
)
TBLITE_INTERMEDIATE_RESULTS = Path(
    "/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tblite-intermediate-reference-sweep-20260817"
)
TB21_RESULTS = Path(
    "/weka/oe-adapt-default/jacobm/tmax-eval/qwen36-tb21-reference-sweep-20260815"
)
EXPERT_COUNTS = (4, 6, 8, 10, 12)
TB21_EXPANDED_EXPERT_COUNTS = (14, 16, 32, 64)


@dataclass(frozen=True)
class RunSpec:
    benchmark: str
    expert_count: int
    replicate: int
    expected_tasks: int
    components: tuple[Path, ...]


def current_name(benchmark: str, k: int, replicate: int) -> str:
    routing = "native" if k == 8 else "reference"
    task_count = 100 if benchmark == "tblite" else 89
    seed = 4201 + replicate
    if benchmark == "tblite":
        name = (
            f"qwen36-tblite-k{k}-{routing}-tmax-default-full{task_count}-"
            f"rep{replicate}-seed{seed}-tp2-dp4-c4-h100-20260815"
        )
        if k == 6 and replicate == 3:
            name += "-rerun1"
    else:
        name = (
            f"qwen36-tb21-k{k}-{routing}-tblite-protocol-full{task_count}-"
            f"rep{replicate}-seed{seed}-tp2-dp4-c4-h100-20260815"
        )
        if replicate > 1:
            name += "-replicate-expansion1"
    if k > 8 and (benchmark == "tblite" or replicate == 1):
        name += "-kgt8-nestedfix1"
    return name


RUNS = tuple(
    RunSpec(
        "tblite",
        k,
        replicate,
        100,
        (TBLITE_RESULTS / current_name("tblite", k, replicate),),
    )
    for k in EXPERT_COUNTS
    for replicate in (1, 2, 3)
) + tuple(
    RunSpec(
        "tblite",
        k,
        replicate,
        100,
        (
            TBLITE_EXPANDED_RESULTS
            / (
                f"qwen36-tblite-k{k}-reference-tmax-default-full100-"
                f"rep{replicate}-seed{4201 + replicate}-tp2-dp4-c4-h100-"
                "20260816-expanded1"
            ),
        ),
    )
    for k in (2, 14, 16, 32)
    for replicate in (1, 2, 3)
) + tuple(
    RunSpec(
        "tblite",
        k,
        replicate,
        100,
        (
            TBLITE_EXTREME_RESULTS
            / (
                f"qwen36-tblite-k{k}-reference-tmax-default-full100-"
                f"rep{replicate}-seed{4201 + replicate}-tp2-dp4-c4-h100-"
                "20260816-extreme-reference-full1"
            ),
        ),
    )
    for k in (64, 128, 256)
    for replicate in (1, 2, 3)
) + tuple(
    RunSpec(
        "tblite",
        k,
        replicate,
        100,
        (
            TBLITE_INTERMEDIATE_RESULTS
            / (
                f"qwen36-tblite-k{k}-reference-tmax-default-full100-"
                f"rep{replicate}-seed{4201 + replicate}-tp2-dp4-c4-h100-"
                "20260817-intermediate1"
            ),
        ),
    )
    for k in (18, 20, 22, 24, 26, 28, 30)
    for replicate in (1, 2, 3)
) + tuple(
    RunSpec(
        "tb21",
        k,
        replicate,
        89,
        (TB21_RESULTS / current_name("tb21", k, replicate),),
    )
    for k in EXPERT_COUNTS
    for replicate in (1, 2, 3)
) + tuple(
    RunSpec(
        "tb21",
        k,
        replicate,
        89,
        (
            TB21_RESULTS
            / (
                f"qwen36-tb21-k{k}-reference-tblite-protocol-full89-"
                f"rep{replicate}-seed{4201 + replicate}-tp2-dp4-c4-h100-"
                "20260817-extreme1"
            ),
        ),
    )
    for k in TB21_EXPANDED_EXPERT_COUNTS
    for replicate in (1, 2, 3)
)


def count_exceptions(component_dir: Path) -> int:
    # Each TMax component writes its aggregate result under a nested directory
    # with the same basename. Avoid a recursive Weka traversal here: as the
    # sweep grows, rglob() can spend minutes waiting on unrelated directory
    # metadata even though there is exactly one result file per component.
    result_path = component_dir / component_dir.name / "result.json"
    if not result_path.is_file():
        result_path = component_dir / "result.json"
    if not result_path.is_file():
        return 0
    payload = json.loads(result_path.read_text())
    stats = payload.get("stats", {})
    if "n_errored_trials" in stats:
        return int(stats["n_errored_trials"])
    return int(payload.get("exception_info") is not None)


def load_complete_run(spec: RunSpec) -> dict[str, object] | None:
    payloads: list[dict] = []
    component_dirs: list[Path] = []
    for component in spec.components:
        component_dir = component
        metrics_path = component_dir / "metrics.json"
        if not metrics_path.is_file():
            return None
        payloads.append(json.loads(metrics_path.read_text()))
        component_dirs.append(component_dir)

    task_names: set[str] = set()
    task_payloads: list[dict] = []
    for component, payload in zip(spec.components, payloads, strict=True):
        per_task = payload.get("per_task", {})
        if int(payload.get("n_tasks", -1)) != len(per_task):
            raise RuntimeError(f"Inconsistent task count in {component}")
        overlap = task_names & set(per_task)
        if overlap:
            raise RuntimeError(f"Overlapping tasks in {spec}: {sorted(overlap)[:3]}")
        task_names.update(per_task)
        task_payloads.extend(per_task.values())

    if len(task_names) != spec.expected_tasks:
        return None

    mean_reward = statistics.fmean(float(task["mean_reward"]) for task in task_payloads)
    pass_at_1 = sum(int(task["n_correct"]) for task in task_payloads) / spec.expected_tasks
    return {
        "benchmark": spec.benchmark,
        "expert_count": spec.expert_count,
        "routing": "native" if spec.expert_count == 8 else "reference-scaled",
        "replicate": spec.replicate,
        "mean_reward": mean_reward,
        "pass_at_1": pass_at_1,
        "n_tasks": spec.expected_tasks,
        "n_task_errors": sum(count_exceptions(path) for path in component_dirs),
        "agent": "Vanillux2Agent",
        "agent_timeout_sec": "task-default",
        "command_timeout_sec": "task-default",
        "components": ";".join(str(path) for path in spec.components),
    }


def load_rows() -> dict[str, list[dict[str, object]]]:
    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for spec in RUNS:
        row = load_complete_run(spec)
        if row is not None:
            rows[spec.benchmark].append(row)
    for benchmark in rows:
        rows[benchmark].sort(key=lambda row: (int(row["expert_count"]), int(row["replicate"])))
    return rows


def write_csv(benchmark: str, rows: list[dict[str, object]]) -> Path:
    output_path = NOTES / f"qwen36_{benchmark}_finished_runs.csv"
    fieldnames = [
        "benchmark",
        "expert_count",
        "routing",
        "replicate",
        "mean_reward",
        "pass_at_1",
        "n_tasks",
        "n_task_errors",
        "agent",
        "agent_timeout_sec",
        "command_timeout_sec",
        "components",
    ]
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def plot_benchmark(
    benchmark: str,
    rows: list[dict[str, object]],
    *,
    max_k: int | None = None,
) -> tuple[Path, Path]:
    descriptions = {
        "tblite": {
            "title": "Qwen3.6-35B-A3B · TB-Lite 2.0 Expert Sweep",
            "subtitle": (
                "Complete 100-task TMax-default runs only · corrected K>8 routing · "
                "partial attempts excluded"
            ),
            "ylabel": "TB-Lite mean reward (%)  ↑",
            "stem": "qwen36_tblite_finished_expert_sweep",
            "empty": "No complete valid runs yet",
        },
        "tb21": {
            "title": "Qwen3.6-35B-A3B · Terminal-Bench 2.1 Expert Sweep",
            "subtitle": (
                "Complete 89-task TMax-default runs only · corrected K>8 routing · "
                "in-progress attempts excluded"
            ),
            "ylabel": "Terminal-Bench 2.1 pass@1 (%)  ↑",
            "stem": "qwen36_tb21_finished_expert_sweep",
            "empty": "No complete valid TMax-default runs yet",
        },
    }
    config = descriptions[benchmark]
    if max_k is not None:
        rows = [row for row in rows if int(row["expert_count"]) <= max_k]
    by_k: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        metric = "mean_reward" if benchmark == "tblite" else "pass_at_1"
        by_k[int(row["expert_count"])].append(100 * float(row[metric]))

    fig, ax = plt.subplots(figsize=(11.6, 6.5))
    present_k = sorted(by_k)
    annotated_k = (
        {2, 4, 8, 16, 32, 64, 128}
        if benchmark == "tblite"
        else {4, 12, 32, 64}
    )
    if benchmark == "tb21" and max_k == 16:
        annotated_k = {4, 8, 12, 16}
    if present_k:
        means = [statistics.fmean(by_k[k]) for k in present_k]
        ax.plot(
            present_k,
            means,
            color="#293241",
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="Mean of complete runs",
            zorder=4,
        )
        for k, mean in zip(present_k, means, strict=True):
            values = by_k[k]
            if len(values) == 1:
                offsets = [0.0]
            else:
                width = 0.20
                offsets = [(-width / 2) + width * index / (len(values) - 1) for index in range(len(values))]
                ax.vlines(k, min(values), max(values), color="#0072B2", alpha=0.55, linewidth=2)
            ax.scatter(
                [k + offset for offset in offsets],
                values,
                color="#0072B2",
                s=62,
                alpha=0.82,
                edgecolors="white",
                linewidths=0.8,
                zorder=5,
                label="Individual complete runs" if k == present_k[0] else None,
            )
            if k in annotated_k:
                ax.annotate(
                    f"{mean:.2f}% (n={len(values)})",
                    (k, mean),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=10.5,
                    fontweight="semibold",
                )
        ax.legend(frameon=False, loc="lower right")
    else:
        ax.text(
            0.5,
            0.52,
            str(config["empty"]),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=16,
            color="#626B73",
            fontweight="semibold",
        )

    if benchmark == "tblite" and present_k:
        tick_k = [k for k in present_k if k in {2, 4, 8, 12, 16, 20, 24, 28, 32, 64, 128}]
    else:
        tick_k = present_k if present_k else list(EXPERT_COUNTS)
    padding = max(0.75, (max(tick_k) - min(tick_k)) * 0.03)
    ax.set_xticks(tick_k)
    ax.set_xlim(min(tick_k) - padding, max(tick_k) + padding)
    if benchmark == "tb21" and max_k == 16:
        ax.set_ylim(20, 40)
    else:
        ax.set_ylim(0, 100)
    ax.set_xlabel("Active routed experts", fontweight="semibold")
    ax.set_ylabel(str(config["ylabel"]), fontweight="semibold")
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.9)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(str(config["title"]), fontsize=17, fontweight="bold", y=0.98)
    subtitle = str(config["subtitle"])
    if max_k is not None:
        subtitle += f" · shown through K={max_k}"
    ax.set_title(subtitle, fontsize=10.5, color="#555A63", pad=14)
    fig.tight_layout()

    PLOTS.mkdir(parents=True, exist_ok=True)
    output_stem = str(config["stem"])
    if max_k is not None:
        output_stem += f"_through_k{max_k}"
    stem = PLOTS / output_stem
    png_path = stem.with_suffix(".png")
    svg_path = stem.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    rows = load_rows()
    for benchmark in ("tblite", "tb21"):
        benchmark_rows = rows.get(benchmark, [])
        csv_path = write_csv(benchmark, benchmark_rows)
        png_path, svg_path = plot_benchmark(benchmark, benchmark_rows)
        print(
            f"{benchmark}: {len(benchmark_rows)} complete runs; wrote "
            f"{csv_path}, {png_path}, and {svg_path}"
        )

    for benchmark, max_k in (("tblite", 32), ("tb21", 16)):
        png_path, svg_path = plot_benchmark(
            benchmark,
            rows.get(benchmark, []),
            max_k=max_k,
        )
        print(f"{benchmark} through K={max_k}: wrote {png_path} and {svg_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot the constituent evaluations behind the Qwen3 Base OLMoBase summaries."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

RUNS = {
    1: "01KX597HRNRSJD6Q028SGR7X1J",
    2: "01KX597TE7GSPMXKQAHDKJ0TZG",
    3: "01KX7GKDS69Y3P61MPTJD9BPWD",
    4: "01KX598334MGBFC3RN01PR27RH",
    5: "01KX6M1VV8H27K4J2CVDV64Y82",
    6: "01KX6M2634H40FR7WZP0WGCSZ9",
    7: "01KX6M2QW5FG1BNH5DHW7F1PM1",
    8: "01KX598BVQG62BXW751MJMEZTG",
    9: "01KX7GKR8K19TYT3Z26EA5RVQC",
    10: "01KX7GM1MGHZYWYG9BWE55JABZ",
    11: "01KX7GMAB9E6JEZ669XR33CEZA",
    12: "01KX7GMM4TCFTVA32T7RPTNZHB",
    16: "01KX598MHGPFNET048MXHQGX64",
}


@dataclass(frozen=True)
class Panel:
    title: str
    aggregate: str
    components: tuple[str, ...]
    lower_is_better: bool = False


PANELS = (
    Panel(
        "MCQA STEM",
        "olmobase:mcqa_stem",
        (
            "arc:mc:olmo3base",
            "mmlu:stem:mc:olmo3base",
            "medmcqa:mc:olmo3base",
            "medqa_en:mc:olmo3base",
            "sciq:mc:olmo3base",
        ),
    ),
    Panel(
        "MCQA non-STEM",
        "olmobase:mcqa_non_stem",
        (
            "mmlu:humanities:mc:olmo3base",
            "mmlu:other:mc:olmo3base",
            "mmlu:social_sciences:mc:olmo3base",
            "csqa:mc_olmo3base",
            "piqa:mc_olmo3base",
            "socialiqa:mc_olmo3base",
            "coqa:mc:olmo3base",
            "drop:mc:olmo3base",
            "jeopardy:mc:olmo3base",
            "naturalqs:mc:olmo3base",
            "squad:mc:olmo3base",
        ),
    ),
    Panel(
        "Generation",
        "olmobase:gen",
        (
            "hellaswag:rc:olmo3base",
            "lambada:olmo3base",
            "winogrande:rc:olmo3base",
            "basic_skills:rc:olmo3base",
            "drop:gen:olmo3base",
            "jeopardy:gen:olmo3base",
            "naturalqs:gen:olmo3base",
            "squad:gen:olmo3base",
            "coqa:gen:olmo3base",
        ),
    ),
    Panel(
        "Math",
        "olmobase:math",
        (
            "gsm8k:olmo3base",
            "gsm_symb:olmo3base",
            "minerva_math:olmo3base",
        ),
    ),
    Panel(
        "Easy QA reading comprehension",
        "olmobase:easy:qa:rc",
        (
            "arc:rc:olmo3base",
            "mmlu:rc:olmo3base",
            "csqa:rc:olmo3base",
            "hellaswag:rc:olmo3base",
            "winogrande:rc:olmo3base",
            "socialiqa:rc:olmo3base",
            "piqa:rc:olmo3base",
            "coqa:rc:olmo3base",
            "drop:rc:olmo3base",
            "jeopardy:rc:olmo3base",
            "naturalqs:rc:olmo3base",
            "squad:rc:olmo3base",
            "sciq:rc:olmo3base",
            "qasper_yesno:rc:olmo3base",
            "basic_skills:rc:olmo3base",
            "lab_bench_dbqa:olmo3base",
            "lab_bench_protocolqa:olmo3base",
            "lambada",
            "medmcqa:rc:olmo3base",
            "medqa_en:rc:olmo3base",
            "sciriff_yesno:rc:olmo3base",
        ),
    ),
    Panel(
        "Easy QA bits per byte",
        "olmobase:easy:qa:bpb",
        (
            "arc:bpb:olmo3base",
            "mmlu:bpb",
            "csqa:bpb:olmo3base",
            "hellaswag:bpb:olmo3base",
            "winogrande:bpb:olmo3base",
            "socialiqa:bpb:olmo3base",
            "piqa:bpb:olmo3base",
            "coqa:bpb:olmo3base",
            "drop:bpb:olmo3base",
            "jeopardy:bpb:olmo3base",
            "naturalqs:bpb:olmo3base",
            "squad:bpb:olmo3base",
            "sciq:bpb:olmo3base",
            "qasper_yesno:bpb:olmo3base",
            "basic_skills:bpb:olmo3base",
            "lab_bench_dbqa:bpb:olmo3base",
            "lab_bench_protocolqa:bpb:olmo3base",
            "lambada:bpb:olmo3base",
            "medmcqa:bpb:olmo3base",
            "medqa_en:bpb:olmo3base",
            "sciriff_yesno:bpb:olmo3base",
        ),
        lower_is_better=True,
    ),
    Panel(
        "Minerva Math bits per byte",
        "olmobase:easy:math:bpb",
        tuple(
            f"minerva_math_{subset}:bpb:olmo3base"
            for subset in (
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            )
        ),
        lower_is_better=True,
    ),
    Panel(
        "Code bits per byte",
        "olmobase:easy:code:bpb",
        (
            "codex_humaneval:bpb:olmo3base",
            "mbpp:bpb:olmo3base",
            "mt_mbpp:bpb:olmo3base",
        ),
        lower_is_better=True,
    ),
)


SPECIAL_LABELS = {
    "arc": "ARC",
    "mmlu": "MMLU",
    "medmcqa": "MedMCQA",
    "medqa_en": "MedQA",
    "sciq": "SciQ",
    "csqa": "CommonsenseQA",
    "piqa": "PIQA",
    "socialiqa": "SocialIQA",
    "coqa": "CoQA",
    "drop": "DROP",
    "jeopardy": "Jeopardy",
    "naturalqs": "NaturalQuestions",
    "squad": "SQuAD",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoGrande",
    "basic_skills": "Basic Skills",
    "qasper_yesno": "Qasper Yes/No",
    "lab_bench_dbqa": "LAB-Bench DBQA",
    "lab_bench_protocolqa": "LAB-Bench ProtocolQA",
    "lambada": "LAMBADA",
    "sciriff_yesno": "SciRIFF Yes/No",
    "gsm8k": "GSM8K",
    "gsm_symb": "GSM-Symbolic",
    "minerva_math": "Minerva Math",
    "codex_humaneval": "HumanEval",
    "mbpp": "MBPP",
    "mt_mbpp": "Multilingual MBPP",
}


def display_name(key: str) -> str:
    base = key.split(":", maxsplit=1)[0]
    for prefix, label in sorted(SPECIAL_LABELS.items(), key=lambda item: -len(item[0])):
        if base == prefix:
            name = label
            break
        if base.startswith(prefix + "_"):
            suffix = base[len(prefix) + 1 :].replace("_", " ").title()
            name = f"{label}: {suffix}"
            break
    else:
        name = base.replace("_", " ").title()

    if ":stem:" in key:
        name += " STEM"
    elif ":humanities:" in key:
        name += " humanities"
    elif ":social_sciences:" in key:
        name += " social sciences"
    elif ":other:" in key:
        name += " other"
    return name


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo / "results" / "adaptive_experts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "notes" / "plots" / "pretraining_individual_evals_expert_sweep.png",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=(
            repo
            / "notes"
            / "pretraining_individual_evals_expert_sweep.csv"
        ),
    )
    return parser.parse_args()


def load_summaries(results_dir: Path) -> dict[int, dict[str, float]]:
    summaries = {}
    for k, experiment_id in RUNS.items():
        path = results_dir / experiment_id / "metrics.json"
        payload = json.loads(path.read_text())
        if payload.get("errors"):
            raise ValueError(f"{experiment_id} has errors: {payload['errors']}")
        summaries[k] = {
            key: float(value["score"]) for key, value in payload["summary"].items()
        }
    return summaries


def write_csv(path: Path, summaries: dict[int, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = (
            "k",
            "suite",
            "evaluation",
            "display_name",
            "kind",
            "direction",
            "score",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel in PANELS:
            for k in RUNS:
                for evaluation, kind in (
                    *((component, "constituent") for component in panel.components),
                    (panel.aggregate, "aggregate"),
                ):
                    writer.writerow(
                        {
                            "k": k,
                            "suite": panel.aggregate,
                            "evaluation": evaluation,
                            "display_name": (
                                "Aggregate" if kind == "aggregate" else display_name(evaluation)
                            ),
                            "kind": kind,
                            "direction": "lower" if panel.lower_is_better else "higher",
                            "score": summaries[k][evaluation],
                        }
                    )


def configure_axis(axis: plt.Axes, *, lower_is_better: bool) -> None:
    ks = np.asarray(tuple(RUNS))
    axis.set_xlim(0.5, 16.5)
    axis.set_xticks(ks)
    axis.get_xaxis().set_major_formatter(ScalarFormatter())
    axis.axvline(8, color="#777777", linestyle=":", linewidth=1.0, alpha=0.8)
    axis.grid(axis="both", color="#D9DDE3", linewidth=0.65, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    if lower_is_better:
        axis.set_yscale("log")
        axis.set_ylabel("Bits per byte ↓")
    else:
        axis.set_ylim(0, 100)
        axis.set_ylabel("Score (%) ↑")


def plot(path: Path, summaries: dict[int, dict[str, float]]) -> None:
    ks = np.asarray(tuple(RUNS))
    fig, axes = plt.subplots(4, 2, figsize=(19, 24))
    for axis, panel in zip(axes.flat, PANELS, strict=True):
        colors = sns.color_palette("husl", n_colors=len(panel.components))
        for component, color in zip(panel.components, colors, strict=True):
            values = np.asarray([summaries[k][component] for k in ks])
            if not panel.lower_is_better:
                values *= 100
            axis.plot(
                ks,
                values,
                color=color,
                marker="o",
                markersize=3.4,
                linewidth=1.45,
                alpha=0.9,
                label=display_name(component),
            )

        aggregate = np.asarray([summaries[k][panel.aggregate] for k in ks])
        if not panel.lower_is_better:
            aggregate *= 100
        axis.plot(
            ks,
            aggregate,
            color="#202124",
            linestyle="--",
            marker="D",
            markersize=4.2,
            linewidth=2.5,
            label="Aggregate reference",
            zorder=10,
        )
        configure_axis(axis, lower_is_better=panel.lower_is_better)
        axis.set_title(panel.title, loc="left", fontsize=13, fontweight="semibold")
        axis.set_xlabel("Active experts per token")
        columns = 3 if len(panel.components) >= 9 else 2
        axis.legend(
            loc="best",
            frameon=True,
            framealpha=0.88,
            fontsize=6.5,
            ncol=columns,
            handlelength=2.0,
            columnspacing=0.8,
        )

    fig.suptitle(
        "Qwen3-30B-A3B-Base: constituent OLMoBase evaluations vs active experts",
        fontsize=20,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.975,
        "One consolidated full-suite run per K. Colored lines are constituents; "
        "dashed black lines are suite aggregates. Math is stochastic; BPB uses a log scale.",
        ha="center",
        va="top",
        color="#555A63",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.945), h_pad=3.0, w_pad=2.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.results_dir)
    write_csv(args.csv_output, summaries)
    plot(args.output, summaries)
    print(f"wrote {args.output}")
    print(f"wrote {args.output.with_suffix('.svg')}")
    print(f"wrote {args.csv_output}")


if __name__ == "__main__":
    main()

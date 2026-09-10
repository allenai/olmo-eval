#!/usr/bin/env python3
"""Plot the normalized Nemotron 3 Super expert-count sweep."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    11: [
        "01KXRRC639BXGACTNNSYV6PEFF",
        "01KXRRD5DR2QXVG3YZJYMX04Q1",
        "01KXRRE5RQT3X8P26S7YXVXMD8",
    ],
    13: [
        "01KXRWPY7MT098PJWFTXF3HXYD",
        "01KXRWRB7AVEPKXX39E6RZDRCH",
        "01KXRWSJZJD77ZDGJ5EFF195YJ",
    ],
    15: [
        "01KXRWQ6HQD5692HQB88DZ3KRV",
        "01KXRWRK57EEWHMKCDPXMZZPH9",
        "01KXRWSVA89KYBSQFHX6DX9CRD",
    ],
    17: [
        "01KXRWQEKJZ86AQAE0XF7E42PY",
        "01KXRWRTTXQT4WS3B9F26RWJ3Z",
        "01KXRWT4100K9N02C5KD0ZWH3X",
    ],
    19: [
        "01KXRWQPHFX7SVFMJ2FFZSXJCB",
        "01KXRWS2SZE075XVDP3QPT8H6N",
        "01KXRWTC48F70DQQR59AC6RS8H",
    ],
    21: [
        "01KXRWQYM3JP4MB54KAHS53GDM",
        "01KXRWSAY6QFDRCB35M78XQ1N9",
        "01KXRWTMC4Q1GC72WV310D427Y",
    ],
    22: [
        "01KXRRBYEBQZRNH267A9F77RX1",
        "01KXRRCXGTE0H8JMW001Q13KE0",
        "01KXRRDXWFARQX0QJ7EMBS3J2T",
    ],
}


def read_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    scores: dict[str, float] = {}
    for task in payload["tasks"]:
        if task["task"].startswith("math500"):
            scores["MATH-500"] = task["metrics"]["accuracy"]["minerva_math_flex"] * 100
        elif task["task"].startswith("gpqa_diamond"):
            scores["GPQA Diamond"] = (
                task["metrics"]["accuracy"]["multiple_choice"] * 100
            )
        elif task["task"].startswith("ifeval_ood"):
            scores["IFEval OOD"] = (
                task["metrics"]["prompt_level_loose_acc"]["ifeval"] * 100
            )
    if set(scores) != {"MATH-500", "GPQA Diamond", "IFEval OOD"}:
        raise ValueError(f"missing expected metrics in {path}")
    scores["Macro average"] = float(np.mean(list(scores.values())))
    return scores


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    results = repo / "results" / "adaptive_experts"
    output = repo / "notes" / "plots" / "nemotron3_super_posttraining_expert_sweep"
    expert_counts = np.array(sorted(RUNS))
    metric_names = ["MATH-500", "GPQA Diamond", "IFEval OOD", "Macro average"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#7A5195"]
    markers = ["o", "s", "^", "D"]

    means: dict[str, list[float]] = {name: [] for name in metric_names}
    standard_deviations: dict[str, list[float]] = {name: [] for name in metric_names}
    for expert_count in expert_counts:
        run_scores = [
            read_scores(results / experiment_id / "metrics.json")
            for experiment_id in RUNS[int(expert_count)]
        ]
        for metric_name in metric_names:
            values = np.array([scores[metric_name] for scores in run_scores])
            means[metric_name].append(float(values.mean()))
            standard_deviations[metric_name].append(float(values.std(ddof=1)))

    fig, axis = plt.subplots(figsize=(11.5, 7.2))
    for metric_name, color, marker in zip(
        metric_names, colors, markers, strict=True
    ):
        axis.errorbar(
            expert_counts,
            means[metric_name],
            yerr=standard_deviations[metric_name],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            capsize=3,
            label=f"{metric_name} ↑",
        )

    axis.axvline(
        22,
        color="#73777F",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label="Checkpoint default K=22",
    )
    axis.set_title("Nemotron 3 Super: normalized active-expert sweep")
    axis.set_xlabel("Active routed experts (K)")
    axis.set_ylabel("Performance (%)")
    axis.set_xticks(expert_counts)
    axis.set_xlim(10.5, 22.5)
    axis.set_ylim(55, 100)
    axis.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.6)
    axis.legend(frameon=False, ncol=2, loc="lower right")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

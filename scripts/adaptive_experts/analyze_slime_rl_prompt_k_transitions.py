#!/usr/bin/env python3
"""Analyze matched-prompt MATH-500 outcomes across expert counts.

This consumes the three-replicate step-350 plateau manifest and the prediction
files already downloaded from Beaker. It intentionally uses reference-scaled
routing only: those are the runs we kept for the low-K plateau comparison.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EVAL_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = EVAL_ROOT.parent
MANIFEST = EVAL_ROOT / "notes/slime_qwen3_base_dapo_step350_plateau_replicates.csv"
RESULTS = WORKSPACE_ROOT / "results/rl_checkpoint_evals"
OUT_CSV = EVAL_ROOT / "notes/slime_qwen3_base_dapo_prompt_k_transitions.csv"
OUT_MD = EVAL_ROOT / "notes/slime_qwen3_base_dapo_prompt_k_transitions.md"
OUT_PNG = EVAL_ROOT / "notes/plots/slime_qwen3_base_dapo_prompt_k_heatmaps.png"
OUT_SVG = EVAL_ROOT / "notes/plots/slime_qwen3_base_dapo_prompt_k_heatmaps.svg"

RUN_LABELS = {
    "k8-normalized": "trained K=8 normalized",
    "k6-reference-k8": "trained K=6 reference-scaled",
    "k4-reference-k8": "trained K=4 reference-scaled",
}


def accuracy(record: dict) -> int:
    metrics = record["instance_metrics"]
    value = metrics.get("accuracy", {}).get("minerva_math_flex")
    if value is None:
        value = metrics["minerva_math_flex"]["minerva_math_flex"]
    return int(float(value) >= 0.5)


def prediction_path(experiment_id: str) -> Path:
    matches = list((RESULTS / experiment_id).glob("**/*math500*predictions.jsonl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one MATH-500 prediction file for {experiment_id}; found {matches}"
        )
    return matches[0]


def load_results() -> tuple[list[str], list[int], dict[str, dict[int, dict[int, list[int]]]]]:
    rows = list(csv.DictReader(MANIFEST.open()))
    rows = [row for row in rows if row["mode"] == "reference"]
    runs = [run for run in RUN_LABELS if any(row["run"] == run for row in rows)]
    ks = sorted({int(row["k"]) for row in rows})
    values: dict[str, dict[int, dict[int, list[int]]]] = {
        run: {k: {} for k in ks} for run in runs
    }

    for row in rows:
        run = row["run"]
        if run not in values:
            continue
        k = int(row["k"])
        with prediction_path(row["experiment_id"]).open() as f:
            records = [json.loads(line) for line in f]
        if len(records) != int(row["num_instances"]):
            raise RuntimeError(f"Unexpected row count for {row['experiment_id']}")
        for record in records:
            prompt_id = int(record["native_id"])
            values[run][k].setdefault(prompt_id, []).append(accuracy(record))

    expected_prompts = None
    for run in runs:
        for k in ks:
            prompt_ids = set(values[run][k])
            if expected_prompts is None:
                expected_prompts = prompt_ids
            if prompt_ids != expected_prompts:
                raise RuntimeError(f"Prompt mismatch for {run}, K={k}")
            bad = [p for p, outcomes in values[run][k].items() if len(outcomes) != 3]
            if bad:
                raise RuntimeError(f"Expected three replicates for {run}, K={k}: {bad[:5]}")
    return runs, ks, values


def pattern_for(matrix: np.ndarray, ks: list[int], selected: tuple[int, ...]) -> np.ndarray:
    indices = [ks.index(k) for k in selected]
    majority = matrix[:, indices] >= (2 / 3)
    return np.array(["".join(str(int(v)) for v in row) for row in majority])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze_run(ks: list[int], per_k: dict[int, dict[int, list[int]]]) -> tuple[np.ndarray, dict]:
    prompt_ids = sorted(per_k[ks[0]])
    matrix = np.array(
        [[np.mean(per_k[k][prompt_id]) for k in ks] for prompt_id in prompt_ids],
        dtype=float,
    )
    majority = matrix >= (2 / 3)
    monotone = np.all(np.diff(majority.astype(int), axis=1) >= 0, axis=1)
    all_correct = np.all(majority, axis=1)
    all_wrong = np.all(~majority, axis=1)
    threshold = monotone & ~all_correct & ~all_wrong
    nonmonotone = ~monotone
    replicate_disagreement = (matrix > 0) & (matrix < 1)
    down_reversal = np.any(np.diff(majority.astype(int), axis=1) < 0, axis=1)
    up_transition = np.any(np.diff(majority.astype(int), axis=1) > 0, axis=1)

    selected = (4, 6, 8)
    patterns = pattern_for(matrix, ks, selected)
    pattern_counts = Counter(patterns)
    selected_indices = [ks.index(k) for k in selected]
    k4 = matrix[:, ks.index(4)]
    k8 = matrix[:, ks.index(8)]
    selected_corrs = {
        f"{a}-{b}": corr(matrix[:, ks.index(a)], matrix[:, ks.index(b)])
        for a, b in ((4, 6), (6, 8), (4, 8))
    }

    stats = {
        "prompt_ids": prompt_ids,
        "scores": {k: float(matrix[:, i].mean()) for i, k in enumerate(ks)},
        "all_correct": int(all_correct.sum()),
        "all_wrong": int(all_wrong.sum()),
        "monotone_threshold": int(threshold.sum()),
        "nonmonotone": int(nonmonotone.sum()),
        "any_replicate_disagreement": int(np.any(replicate_disagreement, axis=1).sum()),
        "cell_replicate_disagreement": float(replicate_disagreement.mean()),
        "any_down_reversal": int(down_reversal.sum()),
        "any_up_transition": int(up_transition.sum()),
        "patterns": pattern_counts,
        "selected_corrs": selected_corrs,
        "selected_mean_abs_step": float(
            np.abs(np.diff(matrix[:, selected_indices], axis=1)).mean()
        ),
        "k4_k8_equal": float(np.mean(k4 == k8)),
        "k8_better_than_k4": float(np.mean(k8 > k4)),
        "k4_better_than_k8": float(np.mean(k4 > k8)),
        "k8_minus_k4": float(np.mean(k8 - k4)),
    }
    return matrix, stats


def write_prompt_csv(
    runs: list[str], ks: list[int], matrices: dict[str, np.ndarray], stats: dict[str, dict]
) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        fieldnames = ["run", "prompt_id", *[f"p_correct_k{k}" for k in ks], "majority_pattern"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            matrix = matrices[run]
            for row_idx, prompt_id in enumerate(stats[run]["prompt_ids"]):
                row = {"run": run, "prompt_id": prompt_id}
                row.update({f"p_correct_k{k}": matrix[row_idx, i] for i, k in enumerate(ks)})
                row["majority_pattern"] = "".join(
                    str(int(value >= (2 / 3))) for value in matrix[row_idx]
                )
                writer.writerow(row)


def write_report(runs: list[str], ks: list[int], stats: dict[str, dict]) -> None:
    lines = [
        "# Prompt-level MATH-500 transitions across expert counts",
        "",
        "This analysis matches the same 500 prompts across every K and uses all three",
        "replicates of the step-350 **reference-scaled** plateau sweep. A prompt is called",
        "correct at a K when at least two of three generations are correct. The individual",
        "replicates are independent samples, so the analysis treats 0/3 through 3/3 as an",
        "empirical success rate rather than pairing replicate 1 across K.",
        "",
        "## Summary",
        "",
        "| Training run | Always correct | Always wrong | Clean K threshold | Non-monotonic | Any replicate disagreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        s = stats[run]
        lines.append(
            f"| {RUN_LABELS[run]} | {s['all_correct']} ({s['all_correct']/5:.1f}%) "
            f"| {s['all_wrong']} ({s['all_wrong']/5:.1f}%) "
            f"| {s['monotone_threshold']} ({s['monotone_threshold']/5:.1f}%) "
            f"| {s['nonmonotone']} ({s['nonmonotone']/5:.1f}%) "
            f"| {s['any_replicate_disagreement']} ({s['any_replicate_disagreement']/5:.1f}%) |"
        )

    lines += [
        "",
        "`Clean K threshold` means a majority-correct sequence that changes from wrong to",
        "correct at most once over K=2…12. `Non-monotonic` means a majority-correct prompt",
        "later becomes majority-wrong at a larger K at least once.",
        "",
        "## The policy-relevant K=4/6/8 slice",
        "",
        "Patterns are majority correctness at K=4, K=6, and K=8 respectively.",
        "",
        "| Training run | 000 | 001 | 011 | 111 | Non-monotonic patterns | Corr. 4↔6 | Corr. 6↔8 | Corr. 4↔8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    monotonic_patterns = {"000", "001", "011", "111"}
    for run in runs:
        s = stats[run]
        p = s["patterns"]
        nonmono = sum(count for pattern, count in p.items() if pattern not in monotonic_patterns)
        c = s["selected_corrs"]
        lines.append(
            f"| {RUN_LABELS[run]} | {p['000']} | {p['001']} | {p['011']} | {p['111']} "
            f"| {nonmono} | {c['4-6']:.3f} | {c['6-8']:.3f} | {c['4-8']:.3f} |"
        )

    lines += ["", "Mean scores from the same prediction files:", ""]
    lines.append("| Training run | " + " | ".join(f"K={k}" for k in ks) + " |")
    lines.append("|---|" + "---:|" * len(ks))
    for run in runs:
        lines.append(
            f"| {RUN_LABELS[run]} | "
            + " | ".join(f"{stats[run]['scores'][k]:.3f}" for k in ks)
            + " |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "There is a real but small monotonic signal in the policy-relevant K=4/6/8 range.",
        "Across the three checkpoints, 73.6--78.0% of prompts are majority-correct at every",
        "one of K=4, K=6, and K=8, while 10.6--12.6% are wrong at all three. Only 4.2--6.8%",
        "show a clean transition that actually needs K=6 or K=8. Another 5.4--7.0% have a",
        "non-monotonic majority pattern. Prompt difficulty is nevertheless fairly stable:",
        "the K=4/K=8 empirical-success correlations are 0.735--0.797.",
        "",
        "Looking directly at the three-sample empirical probabilities rather than majority",
        "labels makes the limited separation especially clear. K=4 and K=8 have identical",
        "success rates on 69--79% of prompts. K=8 is better on 11.6--16.0%, while K=4 is",
        "better on 7.8--15.0%; the net average K=8 advantage is only 0.9--3.1 percentage",
        "points, depending on the trained checkpoint.",
        "",
        "Generation variance is material. Even within only K=4/6/8, 27.8--38.6% of prompts",
        "have at least one K where the three generations disagree. Over the full K=2--12",
        "sweep that rises to 67.0--75.6%. A hard label such as ‘this prompt needs K=6’ from",
        "one generation would therefore be noisy and highly imbalanced. If we pursue a prompt",
        "selector, it should use multi-sample success probabilities/confidence bounds and first",
        "report an oracle quality-versus-average-K curve. Direct cost-aware RL remains especially",
        "well motivated because it need not pretend every prompt has one deterministic minimum K.",
        "",
        f"![Matched-prompt heatmaps]({OUT_PNG.resolve()})",
        "",
        f"Per-prompt probabilities: `{OUT_CSV.resolve()}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")


def plot_heatmaps(runs: list[str], ks: list[int], matrices: dict[str, np.ndarray]) -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(runs), figsize=(15, 8), sharey=True, constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]
    image = None
    for ax, run in zip(axes, runs):
        matrix = matrices[run]
        # Stable difficulty first, then the location of correctness gains. This keeps the
        # exact same prompt set while making threshold-like and irregular bands visible.
        weights = np.linspace(1.0, 1.5, matrix.shape[1])
        order = np.lexsort((matrix @ weights, matrix.mean(axis=1)))
        image = ax.imshow(matrix[order], aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(RUN_LABELS[run])
        ax.set_xlabel("Experts used at evaluation (K)")
        ax.set_xticks(range(len(ks)), ks)
        ax.set_yticks([])
    axes[0].set_ylabel("Same 500 MATH-500 prompts\n(sorted independently for visibility)")
    cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Empirical correctness across 3 generations")
    fig.suptitle("Prompt-level performance across K (reference-scaled routing)", fontsize=15)
    fig.savefig(OUT_PNG, dpi=180)
    fig.savefig(OUT_SVG)
    plt.close(fig)


def main() -> None:
    runs, ks, values = load_results()
    matrices = {}
    stats = {}
    for run in runs:
        matrices[run], stats[run] = analyze_run(ks, values[run])
    write_prompt_csv(runs, ks, matrices, stats)
    plot_heatmaps(runs, ks, matrices)
    write_report(runs, ks, stats)
    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_PNG)


if __name__ == "__main__":
    main()

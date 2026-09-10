#!/usr/bin/env python3
"""Plot the completed adaptive-expert evaluation sweeps."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "notes" / "plots"

# Color-blind-friendly palette based on Okabe-Ito.
COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#292929",
}

PRETRAIN_K = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16])
PRETRAIN_TICKS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16])
PRETRAIN_SCORES = {
    "MCQA STEM ↑": np.array(
        [22.70, 32.34, 66.27, 75.22, 79.71, 77.13, 79.52, 69.69, 74.55, 68.36, 72.89, 44.39, 39.60]
    ),
    "MCQA non-STEM ↑": np.array(
        [25.37, 31.63, 64.73, 76.96, 82.26, 79.95, 81.91, 72.30, 77.32, 71.26, 75.45, 46.57, 41.72]
    ),
    "Generation ↑": np.array(
        [10.31, 20.45, 48.66, 62.17, 64.63, 62.33, 65.28, 59.78, 64.70, 60.99, 64.72, 39.93, 36.38]
    ),
    "Math ↑": np.array(
        [0.57, 0.95, 24.33, 45.62, 58.93, 60.19, 66.15, 65.02, 66.96, 66.44, 67.93, 60.35, 58.47]
    ),
    "Easy QA RC ↑": np.array(
        [24.98, 40.38, 57.66, 69.04, 74.39, 71.80, 74.27, 65.61, 70.11, 64.73, 68.61, 42.47, 38.46]
    ),
}
PRETRAIN_MATH_SD = np.array(
    [0.01, 0.02, 0.82, 1.19, 0.81, 2.40, 1.86, 6.45, 4.42, 7.60, 5.06, 18.53, 20.61]
)
PRETRAIN_BPB = {
    "Easy QA BPB ↓": np.array(
        [
            4.2936,
            1.6105,
            0.9457,
            0.7571,
            0.7164,
            0.6904,
            0.6771,
            0.6615,
            0.6605,
            0.6524,
            0.6485,
            0.6180,
            0.6071,
        ]
    ),
    "Easy math BPB ↓": np.array(
        [
            4.9600,
            0.8427,
            0.4249,
            0.3470,
            0.3214,
            0.3120,
            0.3073,
            0.3063,
            0.3081,
            0.3082,
            0.3088,
            0.3087,
            0.3139,
        ]
    ),
    "Easy code BPB ↓": np.array(
        [
            4.6291,
            0.8391,
            0.4239,
            0.3179,
            0.2885,
            0.2754,
            0.2624,
            0.2581,
            0.2551,
            0.2582,
            0.2599,
            0.2594,
            0.2552,
        ]
    ),
}

POSTTRAIN_K = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32])
POSTTRAIN_TICKS = POSTTRAIN_K
POSTTRAIN_SCORES = {
    "MATH-500 ↑": (
        np.array(
            [
                0.00,
                0.07,
                62.13,
                94.27,
                95.07,
                95.07,
                95.13,
                95.33,
                94.40,
                94.87,
                95.00,
                94.47,
                93.80,
                94.00,
                94.40,
                93.73,
                76.00,
            ]
        ),
        np.array(
            [
                0.00,
                0.12,
                0.76,
                0.23,
                0.70,
                0.23,
                0.31,
                0.12,
                0.53,
                0.61,
                0.35,
                0.23,
                0.00,
                0.28,
                0.57,
                0.12,
                1.44,
            ]
        ),
    ),
    "GPQA Diamond ↑": (
        np.array(
            [
                0.00,
                0.67,
                40.07,
                56.06,
                60.27,
                62.46,
                62.46,
                61.28,
                62.46,
                62.46,
                63.30,
                60.94,
                60.10,
                61.62,
                63.38,
                58.25,
                36.20,
            ]
        ),
        np.array(
            [
                0.00,
                0.29,
                5.56,
                0.51,
                2.54,
                1.77,
                2.04,
                2.04,
                1.77,
                0.29,
                1.27,
                0.29,
                5.00,
                1.43,
                2.50,
                2.96,
                1.05,
            ]
        ),
    ),
    "IFEval OOD ↑": (
        np.array(
            [
                0.00,
                0.00,
                8.44,
                28.89,
                34.44,
                38.11,
                37.78,
                37.78,
                40.56,
                42.44,
                41.11,
                40.67,
                40.67,
                41.00,
                40.83,
                42.44,
                33.67,
            ]
        ),
        np.array(
            [
                0.00,
                0.00,
                2.14,
                1.39,
                0.69,
                0.69,
                1.90,
                0.51,
                1.02,
                1.26,
                0.69,
                2.00,
                1.89,
                0.00,
                0.71,
                1.71,
                1.33,
            ]
        ),
    ),
    "Macro average ↑": (
        np.array(
            [
                0.00,
                0.25,
                36.88,
                59.74,
                63.26,
                65.21,
                65.12,
                64.80,
                65.80,
                66.59,
                66.47,
                65.36,
                64.86,
                65.54,
                66.21,
                64.81,
                48.62,
            ]
        ),
        np.array(
            [
                0.00,
                0.08,
                2.49,
                0.43,
                0.69,
                0.49,
                0.87,
                0.79,
                0.51,
                0.50,
                0.61,
                0.69,
                2.30,
                0.38,
                0.88,
                1.11,
                0.75,
            ]
        ),
    ),
}

GPTOSS_K = np.arange(1, 9)
GPTOSS_TICKS = np.arange(1, 9)
GPTOSS_SCORES = {
    "MATH-500 ↑": (
        np.array([42.8000, 92.0000, 91.8667, 92.4000, 91.8667, 92.4667, 92.0000, 91.9333]),
        np.array([2.1633, 0.5292, 1.1015, 0.3464, 0.9018, 0.9866, 0.5292, 0.7572]),
    ),
    "GPQA Diamond ↑": (
        np.array([29.9663, 70.0337, 74.5791, 70.5387, 72.2222, 70.5387, 70.2020, 68.5185]),
        np.array([3.7228, 0.2916, 1.2710, 1.6235, 0.8748, 1.5430, 0.5051, 4.1134]),
    ),
    "IFEval OOD ↑": (
        np.array([20.2222, 64.6667, 63.2222, 65.1111, 65.0000, 64.0000, 64.8889, 62.1111]),
        np.array([4.2208, 2.6034, 1.9532, 1.1706, 0.8819, 1.8559, 1.8954, 2.6736]),
    ),
    "Macro average ↑": (
        np.array([30.9962, 75.5668, 76.5560, 76.0166, 76.3630, 75.6685, 75.6970, 74.1877]),
        np.array([0.6233, 0.8445, 1.0805, 0.6571, 0.3559, 1.4226, 0.7462, 1.8369]),
    ),
}

GLM_K = np.array([1, 2, 3, 4, 5, 7, 8])
GLM_SCORES = {
    "MATH-500 ↑": (
        np.array([0.0667, 53.6667, 91.0667, 94.7333, 94.8000, 95.6000, 95.5333]),
        np.array([0.1155, 0.9452, 0.7572, 0.3055, 0.4000, 0.3464, 0.4163]),
    ),
    "GPQA Diamond ↑": (
        np.array([0.3367, 22.2222, 55.2189, 64.9832, 65.4882, 59.5960, 60.6061]),
        np.array([0.5832, 4.4029, 2.0411, 1.5430, 1.2710, 4.4029, 2.3144]),
    ),
    "IFEval OOD ↑": (
        np.array([0.1111, 17.2222, 26.4444, 30.5556, 34.1111, 37.4444, 39.3333]),
        np.array([0.1925, 1.8359, 2.2690, 1.3472, 2.3413, 1.5396, 0.5774]),
    ),
    "Macro average ↑": (
        np.array([0.1715, 31.0370, 57.5767, 63.4240, 64.7998, 64.2135, 65.1576]),
        np.array([0.1448, 1.0267, 1.1894, 0.3782, 0.8106, 1.3595, 0.6300]),
    ),
}

# Completed OLMo3-parser runs for the Dolci-think SFT100k checkpoint.
# Superseded Qwen3-parser runs are excluded.
DOLCI_QWEN_K = np.array([2, 4, 6, 8])
DOLCI_QWEN_TICKS = np.array([2, 4, 6, 8])
DOLCI_QWEN_SCORES = {
    "MATH-500 ↑": np.array([0.6, 72.8, 84.4, 84.4]),
    "GPQA Diamond ↑": np.array([12.626263, 42.929293, 50.0, 46.969697]),
    "IFEval OOD ↑": np.array([10.666667, 22.333333, 23.333333, 23.0]),
    "Macro average ↑": np.array([7.96431, 46.020875, 52.577778, 51.456566]),
}


QWEN_AIME_K = POSTTRAIN_K
QWEN_AIME_SCORES = {
    "pass@1 ↑": np.array(
        [
            0.0,
            0.0,
            27.604167,
            62.708333,
            71.666667,
            71.458333,
            73.125,
            72.5,
            70.416667,
            69.270833,
            68.541667,
            67.916667,
            62.291667,
            62.8125,
            60.729167,
            56.145833,
            18.541667,
        ]
    ),
    "pass@4 ↑": np.array(
        [
            0.0,
            0.0,
            39.291157,
            75.700871,
            83.705784,
            84.014368,
            85.349277,
            84.755098,
            83.517798,
            83.1231,
            82.944012,
            82.275491,
            77.73007,
            75.192436,
            71.514461,
            71.655729,
            31.765387,
        ]
    ),
    "pass@8 ↑": np.array(
        [
            0.0,
            0.0,
            44.376409,
            79.004192,
            85.973886,
            86.237938,
            87.618128,
            87.483918,
            86.233968,
            86.662784,
            85.923455,
            85.662061,
            82.726949,
            79.699005,
            74.595851,
            75.667509,
            39.894704,
        ]
    ),
    "pass@16 ↑": np.array(
        [
            0.0,
            0.0,
            48.80127,
            79.958789,
            86.635824,
            86.661222,
            89.164086,
            89.162732,
            86.664367,
            88.316152,
            86.63663,
            86.641289,
            85.717659,
            83.715417,
            77.44325,
            79.677301,
            47.513163,
        ]
    ),
    "pass@32 ↑": np.array(
        [
            0.0,
            0.0,
            53.333333,
            80.0,
            86.666667,
            86.666667,
            90.0,
            90.0,
            86.666667,
            90.0,
            86.666667,
            86.666667,
            86.666667,
            86.666667,
            80.0,
            83.333333,
            53.333333,
        ]
    ),
}

GPTOSS_AIME_K = np.arange(2, 9)
GPTOSS_AIME_SCORES = {
    "pass@1 ↑": np.array(
        [80.520833, 81.979167, 83.020833, 80.520833, 79.791667, 79.166667, 79.6875]
    ),
    "pass@4 ↑": np.array(
        [92.173526, 92.962644, 92.902577, 92.018725, 93.28096, 93.409066, 92.198183]
    ),
    "pass@8 ↑": np.array(
        [93.817598, 94.142767, 94.705464, 94.411149, 95.886594, 95.836378, 94.952241]
    ),
    "pass@16 ↑": np.array(
        [94.988671, 94.999999, 95.860075, 95.848858, 96.633124, 96.632718, 96.464425]
    ),
    "pass@32 ↑": np.array(
        [96.666667, 96.666667, 96.666667, 96.666667, 96.666667, 96.666667, 96.666667]
    ),
}

GLM_AIME_K = np.array([2, 4, 8])
GLM_AIME_SCORES = {
    "pass@1 ↑": np.array([3.4375, 68.020833, 75.416667]),
    "pass@4 ↑": np.array([9.541991, 79.742492, 84.452818]),
    "pass@8 ↑": np.array([13.390368, 82.305759, 87.377499]),
    "pass@16 ↑": np.array([16.11626, 84.594199, 90.479792]),
    "pass@32 ↑": np.array([16.666667, 86.666667, 93.333333]),
}


def configure_axis(ax: plt.Axes, expert_counts: np.ndarray, default_k: int = 8) -> None:
    """Apply shared expert-count axis formatting."""
    padding = max(0.5, (expert_counts.max() - expert_counts.min()) * 0.03)
    ax.set_xlim(expert_counts.min() - padding, expert_counts.max() + padding)
    ax.set_xticks(expert_counts)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(axis="y", color="#D5D9DF", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", color="#E8EAED", linewidth=0.6, alpha=0.6)
    ax.axvline(default_k, color="#73777F", linestyle="--", linewidth=1.2, alpha=0.75, zorder=0)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure in presentation and vector formats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def plot_pretraining() -> None:
    """Create the two-panel OLMoBase summary plot."""
    fig, (ax_score, ax_bpb) = plt.subplots(
        2,
        1,
        figsize=(12.5, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.13},
    )
    fig.suptitle(
        "Qwen3-30B-A3B-Base: OLMoBase Performance vs Active Experts",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "Math shows mean ± 1 sample SD (n=3); other suites are single runs",
        ha="center",
        va="top",
        color="#555A63",
        fontsize=10.5,
    )

    score_styles = [
        ("MCQA STEM ↑", COLORS["blue"], "o"),
        ("MCQA non-STEM ↑", COLORS["orange"], "s"),
        ("Generation ↑", COLORS["green"], "^"),
        ("Math ↑", COLORS["vermillion"], "D"),
        ("Easy QA RC ↑", COLORS["purple"], "P"),
    ]
    for label, color, marker in score_styles:
        values = PRETRAIN_SCORES[label]
        ax_score.plot(
            PRETRAIN_K,
            values,
            label=label,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.2,
        )
        if label == "Math ↑":
            ax_score.fill_between(
                PRETRAIN_K,
                np.maximum(0, values - PRETRAIN_MATH_SD),
                np.minimum(100, values + PRETRAIN_MATH_SD),
                color=color,
                alpha=0.15,
                linewidth=0,
            )

    configure_axis(ax_score, PRETRAIN_TICKS)
    ax_score.set_ylim(0, 90)
    ax_score.set_ylabel("Primary score (%)  ↑", fontweight="semibold")
    ax_score.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.5,
    )
    ax_score.text(
        8,
        2.5,
        "default K=8",
        rotation=90,
        ha="right",
        va="bottom",
        color="#666B73",
        fontsize=9,
    )

    bpb_styles = [
        ("Easy QA BPB ↓", COLORS["blue"], "o"),
        ("Easy math BPB ↓", COLORS["orange"], "s"),
        ("Easy code BPB ↓", COLORS["green"], "^"),
    ]
    for label, color, marker in bpb_styles:
        ax_bpb.plot(
            PRETRAIN_K,
            PRETRAIN_BPB[label],
            label=label,
            color=color,
            marker=marker,
            markersize=6.5,
            linewidth=2.2,
        )

    configure_axis(ax_bpb, PRETRAIN_TICKS)
    ax_bpb.set_yscale("log")
    ax_bpb.set_ylim(0.2, 6.0)
    ax_bpb.set_ylabel("Bits per byte  ↓  (log scale)", fontweight="semibold")
    ax_bpb.set_xlabel("Active experts per token", fontweight="semibold")
    ax_bpb.legend(loc="upper right", frameon=False, ncol=1, handlelength=2.5)

    sns.despine(fig=fig)
    save_figure(fig, "pretraining_expert_sweep")
    plt.close(fig)


def plot_posttraining() -> None:
    """Create the hybrid-thinking evaluation plot."""
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    fig.suptitle(
        "Qwen3-30B-A3B Hybrid Thinking: Performance vs Active Experts",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Means use n=3 except K=13–15 (n=2); shaded regions show ± 1 sample SD",
        ha="center",
        va="top",
        color="#555A63",
        fontsize=10.5,
    )

    styles = [
        ("MATH-500 ↑", COLORS["blue"], "o"),
        ("GPQA Diamond ↑", COLORS["orange"], "s"),
        ("IFEval OOD ↑", COLORS["green"], "^"),
        ("Macro average ↑", COLORS["vermillion"], "D"),
    ]
    for label, color, marker in styles:
        mean, sd = POSTTRAIN_SCORES[label]
        ax.plot(
            POSTTRAIN_K,
            mean,
            label=label,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.4,
        )
        ax.fill_between(
            POSTTRAIN_K,
            np.maximum(0, mean - sd),
            np.minimum(100, mean + sd),
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    configure_axis(ax, POSTTRAIN_TICKS)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Active experts per token", fontweight="semibold")
    ax.set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        columnspacing=1.5,
        handlelength=2.6,
    )
    ax.text(
        8,
        3.0,
        "default K=8",
        rotation=90,
        ha="right",
        va="bottom",
        color="#666B73",
        fontsize=9,
    )

    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.87)
    save_figure(fig, "posttraining_expert_sweep")
    plt.close(fig)


def plot_chat_model(
    *,
    expert_counts: np.ndarray,
    ticks: np.ndarray,
    scores: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]],
    title: str,
    subtitle: str,
    default_k: int,
    stem: str,
) -> None:
    """Plot the shared three-task chat evaluation for one model."""
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", color="#555A63", fontsize=10.5)

    styles = [
        ("MATH-500 ↑", COLORS["blue"], "o"),
        ("GPQA Diamond ↑", COLORS["orange"], "s"),
        ("IFEval OOD ↑", COLORS["green"], "^"),
        ("Macro average ↑", COLORS["vermillion"], "D"),
    ]
    for label, color, marker in styles:
        value = scores[label]
        if isinstance(value, tuple):
            mean, sd = value
        else:
            mean, sd = value, None
        ax.plot(
            expert_counts,
            mean,
            label=label,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.4,
        )
        if sd is not None:
            ax.fill_between(
                expert_counts,
                np.maximum(0, mean - sd),
                np.minimum(100, mean + sd),
                color=color,
                alpha=0.14,
                linewidth=0,
            )

    configure_axis(ax, ticks, default_k=default_k)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Active experts per token", fontweight="semibold")
    ax.set_ylabel("Accuracy / score (%)  ↑", fontweight="semibold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.text(
        default_k,
        4.5,
        f"default K={default_k}",
        rotation=90,
        ha="right",
        va="bottom",
        color="#666B73",
        fontsize=9,
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.86)
    save_figure(fig, stem)
    plt.close(fig)


def plot_aime_model(
    *,
    expert_counts: np.ndarray,
    ticks: np.ndarray,
    scores: dict[str, np.ndarray],
    title: str,
    subtitle: str,
    default_k: int,
    stem: str,
    y_min: float = 0,
) -> None:
    """Plot AIME 2026 pass@k results from 32 samples per problem."""
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", color="#555A63", fontsize=10.5)

    styles = [
        ("pass@1 ↑", COLORS["blue"], "o"),
        ("pass@4 ↑", COLORS["orange"], "s"),
        ("pass@8 ↑", COLORS["green"], "^"),
        ("pass@16 ↑", COLORS["vermillion"], "D"),
        ("pass@32 ↑", COLORS["purple"], "P"),
    ]
    for label, color, marker in styles:
        ax.plot(
            expert_counts,
            scores[label],
            label=label,
            color=color,
            marker=marker,
            markersize=7,
            linewidth=2.4,
        )

    configure_axis(ax, ticks, default_k=default_k)
    ax.set_ylim(y_min, 102)
    ax.set_xlabel("Active experts per token", fontweight="semibold")
    ax.set_ylabel("AIME 2026 score (%)  ↑", fontweight="semibold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    ax.text(
        default_k,
        y_min + 3,
        f"default K={default_k}",
        rotation=90,
        ha="right",
        va="bottom",
        color="#666B73",
        fontsize=9,
    )
    sns.despine(fig=fig)
    fig.subplots_adjust(top=0.86)
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="white", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
        }
    )
    plot_pretraining()
    plot_posttraining()
    plot_chat_model(
        expert_counts=GPTOSS_K,
        ticks=GPTOSS_TICKS,
        scores=GPTOSS_SCORES,
        title="GPT-OSS-120B: Performance vs Active Experts",
        subtitle="Mean ± 1 sample SD (n=3); K=3/5/6/7 use the validated TP=2 eager configuration",
        default_k=4,
        stem="gptoss_120b_posttraining_expert_sweep",
    )
    plot_chat_model(
        expert_counts=GLM_K,
        ticks=GLM_K,
        scores=GLM_SCORES,
        title="GLM-4.5-Air: Performance vs Active Experts",
        subtitle="Mean ± 1 sample SD (n=3); K=8 is the checkpoint default",
        default_k=8,
        stem="glm45_air_posttraining_expert_sweep",
    )
    plot_chat_model(
        expert_counts=DOLCI_QWEN_K,
        ticks=DOLCI_QWEN_TICKS,
        scores=DOLCI_QWEN_SCORES,
        title="Qwen3-30B-A3B Dolci-Think SFT100k: Performance vs Active Experts",
        subtitle="Completed OLMo3-parser runs; scores use full dataset denominators",
        default_k=8,
        stem="qwen3_dolci_think_sft100k_posttraining_expert_sweep",
    )
    plot_aime_model(
        expert_counts=QWEN_AIME_K,
        ticks=QWEN_AIME_K,
        scores=QWEN_AIME_SCORES,
        title="Qwen3-30B-A3B Hybrid Thinking: AIME 2026 vs Active Experts",
        subtitle="30 problems × 32 samples per expert-count condition",
        default_k=8,
        stem="qwen3_hybrid_aime2026_expert_sweep",
    )
    plot_aime_model(
        expert_counts=GPTOSS_AIME_K,
        ticks=GPTOSS_AIME_K,
        scores=GPTOSS_AIME_SCORES,
        title="GPT-OSS-120B: AIME 2026 vs Active Experts",
        subtitle="30 problems × 32 samples; invalid K=1 result excluded",
        default_k=4,
        stem="gptoss_120b_aime2026_expert_sweep",
        y_min=75,
    )
    plot_aime_model(
        expert_counts=GLM_AIME_K,
        ticks=GLM_AIME_K,
        scores=GLM_AIME_SCORES,
        title="GLM-4.5-Air: AIME 2026 vs Active Experts",
        subtitle="30 problems × 32 samples per point",
        default_k=8,
        stem="glm45_air_aime2026_expert_sweep",
    )


if __name__ == "__main__":
    main()

"""Pairwise comparison matrix visualization using seaborn/matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from olmo_eval.analysis.pairwise import PairwiseResult, get_win_rate

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# --- Color palette ---
_BG = "#2b2b2b"
_TEXT_LIGHT = "#e0e0e0"
_TEXT_DIM = "#999999"

# Discrete colormap: green (winning) -> beige (tie) -> coral (losing)
_CMAP_COLORS = ["#d47853", "#d4cfc4", "#8cc5a9", "#4a9a7e"]
_CMAP_BOUNDS = [0.0, 0.40, 0.50, 0.60, 1.01]

_LEGEND_ITEMS = [
    ("#4a9a7e", "wins > 60%"),
    ("#8cc5a9", "wins 50\u201360%"),
    ("#d4cfc4", "near tie"),
    ("#d47853", "loses < 40%"),
]


def build_win_rate_matrix(result: PairwiseResult) -> np.ndarray:
    """Build an NxN matrix of win rates. Diagonal = NaN."""
    n = len(result.models)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_win_rate(result.pairs, i, j)
    return matrix


def plot_pairwise_matrix(
    result: PairwiseResult,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Render the pairwise win-rate matrix as a heatmap."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    n = len(result.models)
    matrix = build_win_rate_matrix(result)
    labels = [m.label.replace("\n", " ") for m in result.models]

    # --- Build discrete colormap ---
    cmap = mcolors.ListedColormap(_CMAP_COLORS)
    norm = mcolors.BoundaryNorm(_CMAP_BOUNDS, cmap.N)

    # --- Figure with dark background ---
    cell_size = 1.2
    bottom_margin = 2.2
    fig_w = max(cell_size * n + 3, 7)
    fig_h = cell_size * n + 3 + bottom_margin

    with plt.rc_context(
        {
            "figure.facecolor": _BG,
            "axes.facecolor": _BG,
            "text.color": _TEXT_LIGHT,
            "axes.labelcolor": _TEXT_LIGHT,
            "xtick.color": _TEXT_LIGHT,
            "ytick.color": _TEXT_LIGHT,
        }
    ):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(bottom=bottom_margin / fig_h, top=0.88)

        # --- Heatmap via seaborn ---
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=True,
            fmt=".0%",
            annot_kws={"fontsize": 13, "fontweight": "bold"},
            linewidths=4,
            linecolor=_BG,
            square=True,
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            mask=np.isnan(matrix),
        )

        # Style diagonal cells
        for i in range(n):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color="#3a3a3a", zorder=2))

        # Fix annotation colors based on cell value
        for text in ax.texts:
            try:
                val = float(text.get_text().rstrip("%")) / 100
            except ValueError:
                continue
            text.set_color("#ffffff" if val >= 0.60 or val < 0.40 else _BG)

        # --- Axis labels ---
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0)

        # --- Title ---
        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold", color=_TEXT_LIGHT)

        # --- Bottom section (figure coordinates) ---
        # Legend
        legend_y = (bottom_margin - 0.3) / fig_h
        n_items = len(_LEGEND_ITEMS)
        legend_w = 0.8
        seg = legend_w / n_items
        x0 = (1 - legend_w) / 2
        for idx, (color, desc) in enumerate(_LEGEND_ITEMS):
            lx = x0 + idx * seg
            fig.patches.append(
                plt.Rectangle(
                    (lx, legend_y),
                    0.015,
                    0.012,
                    facecolor=color,
                    edgecolor="none",
                    transform=fig.transFigure,
                    zorder=5,
                )
            )
            fig.text(
                lx + 0.022,
                legend_y + 0.006,
                desc,
                fontsize=7,
                color=_TEXT_DIM,
                va="center",
            )

        # Divider
        div_y = legend_y - 0.03
        fig.add_artist(
            Line2D(
                [x0, x0 + legend_w],
                [div_y, div_y],
                transform=fig.transFigure,
                color="#444444",
                linewidth=0.5,
            )
        )

        # Summary stats
        n_pairs = n * (n - 1) // 2
        total_ties = sum(p.ties for p in result.pairs)
        total_contested = sum(p.wins_a + p.wins_b for p in result.pairs)
        total = total_ties + total_contested
        avg_tie = total_ties / total if total > 0 else 0.0

        wins: dict[int, int] = {i: 0 for i in range(n)}
        losses: dict[int, int] = {i: 0 for i in range(n)}
        for p in result.pairs:
            wins[p.index_a] += p.wins_a
            losses[p.index_a] += p.wins_b
            wins[p.index_b] += p.wins_b
            losses[p.index_b] += p.wins_a
        best_i = max(
            range(n),
            key=lambda i: wins[i] / (wins[i] + losses[i]) if (wins[i] + losses[i]) > 0 else 0.5,
        )
        best_wr = (
            wins[best_i] / (wins[best_i] + losses[best_i])
            if (wins[best_i] + losses[best_i]) > 0
            else 0.5
        )

        summary_y = div_y - 0.02
        fig.text(
            x0,
            summary_y,
            "SUMMARY",
            fontsize=8,
            fontweight="bold",
            color=_TEXT_DIM,
            va="top",
        )

        stats = [
            ("total pairs", str(n_pairs)),
            ("instances / pair", str(result.instance_count)),
            ("avg tie rate", f"{avg_tie:.0%}"),
            ("top model wins", f"{best_wr:.0%}"),
        ]
        col_w = legend_w / len(stats)
        head_y = summary_y - 0.035
        val_y = head_y - 0.025
        for idx, (heading, value) in enumerate(stats):
            cx = x0 + idx * col_w
            fig.text(cx, head_y, heading, fontsize=7, color=_TEXT_DIM, va="top")
            fig.text(
                cx,
                val_y,
                value,
                fontsize=16,
                fontweight="bold",
                color=_TEXT_LIGHT,
                va="top",
            )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)

    return fig

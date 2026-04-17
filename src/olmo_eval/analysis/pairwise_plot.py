"""Pairwise comparison matrix visualization using matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from olmo_eval.analysis.pairwise import PairwiseResult, get_win_rate

if TYPE_CHECKING:
    from matplotlib.figure import Figure  # type: ignore[ty:unresolved-import]


def build_win_rate_matrix(result: PairwiseResult) -> np.ndarray:
    """Build an NxN matrix of win rates from a PairwiseResult.

    Diagonal entries are np.nan (self-comparison).
    """
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
    """Render the pairwise win-rate matrix as a heatmap.

    Args:
        result: PairwiseResult from compute_pairwise().
        title: Optional figure title.
        save_path: If provided, save to disk and close the figure.
            Otherwise return the Figure for the caller to display or save.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt  # type: ignore[ty:unresolved-import]

    n = len(result.models)
    matrix = build_win_rate_matrix(result)
    labels = [m.label for m in result.models]

    # --- Figure sizing ---
    cell_size = 1.2
    fig_size = cell_size * n + 2
    fig, ax = plt.subplots(figsize=(fig_size, fig_size + 1.2))

    # --- Color map ---
    cmap = plt.cm.RdYlGn.copy()  # type: ignore[attr-defined]
    cmap.set_bad(color="#cccccc")

    im = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    # --- Axis labels ---
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # --- Cell annotations ---
    for i in range(n):
        for j in range(n):
            if i != j:
                val = matrix[i][j]
                text_color = "white" if val < 0.35 or val > 0.75 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                )

    # --- Colorbar ---
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Win Rate")

    # --- Title ---
    if title:
        ax.set_title(title, pad=40, fontsize=12, fontweight="bold")

    # --- Summary stats ---
    n_pairs = n * (n - 1) // 2
    total_ties = sum(p.ties for p in result.pairs)
    total_contested = sum(p.wins_a + p.wins_b for p in result.pairs)
    total_instances_compared = total_ties + total_contested
    avg_tie_rate = total_ties / total_instances_compared if total_instances_compared > 0 else 0.0

    # Best model: highest overall win rate across all pairs
    wins_by_idx: dict[int, int] = {i: 0 for i in range(n)}
    losses_by_idx: dict[int, int] = {i: 0 for i in range(n)}
    for p in result.pairs:
        wins_by_idx[p.index_a] += p.wins_a
        losses_by_idx[p.index_a] += p.wins_b
        wins_by_idx[p.index_b] += p.wins_b
        losses_by_idx[p.index_b] += p.wins_a

    best_idx = max(
        range(n),
        key=lambda i: wins_by_idx[i] / (wins_by_idx[i] + losses_by_idx[i])
        if (wins_by_idx[i] + losses_by_idx[i]) > 0
        else 0.5,
    )
    best_label = result.models[best_idx].label

    summary = (
        f"Pairs: {n_pairs}  |  "
        f"Instances/pair: {result.instance_count}  |  "
        f"Avg tie rate: {avg_tie_rate:.1%}  |  "
        f"Best: {best_label}"
    )
    margin_note = f"  |  Margin: {result.margin}" if result.margin > 0 else ""
    fig.text(
        0.5,
        0.02,
        summary + margin_note,
        ha="center",
        fontsize=8,
        style="italic",
        color="gray",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig

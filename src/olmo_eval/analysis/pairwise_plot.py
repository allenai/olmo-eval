"""Pairwise comparison matrix visualization using seaborn/matplotlib."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from olmo_eval.analysis.pairwise import PairwiseResult, get_se, get_win_rate

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# --- Color palette ---
_BG = "#2b2b2b"
_TEXT_LIGHT = "#e0e0e0"
_TEXT_DIM = "#999999"
_DIVIDER = "#444444"
_DIAGONAL = "#3a3a3a"

# Summary-stat tier colors.
_TIER_GOOD = "#5aa380"
_TIER_OK = "#e2b05a"
_TIER_BAD = "#c85d3b"
_TIER_NEUTRAL = _TEXT_LIGHT

# Continuous gradient: deep red (row loses badly) -> beige (tie) -> deep green
# (row wins decisively). Stops are (position, hex); position is the win-rate
# value the colour is anchored to, in [0, 1].
_CMAP_STOPS: list[tuple[float, str]] = [
    (0.00, "#8f2f18"),
    (0.20, "#c85d3b"),
    (0.40, "#e2a48c"),
    (0.50, "#d4cfc4"),
    (0.60, "#a8d1bb"),
    (0.80, "#5aa380"),
    (1.00, "#1f6b4f"),
]

# Layout constants (inches)
_FOOTER_HEIGHT = 2.00
_RIGHT_PAD = 0.35
_LEFT_PAD = 0.35
_TITLE_PAD = 0.55
# Minimum width reserved for the footer (legend + 4 summary columns)
_MIN_FOOTER_WIDTH = 6.0


def build_win_rate_matrix(result: PairwiseResult) -> np.ndarray:
    """Build an NxN matrix of win rates. Diagonal = NaN."""
    n = len(result.models)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_win_rate(result.pairs, i, j)
    return matrix


def build_se_matrix(result: PairwiseResult) -> np.ndarray:
    """Build an NxN matrix of win-rate standard errors. Diagonal = NaN."""
    n = len(result.models)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_se(result.pairs, i, j)
    return matrix


def _scale_for_n(n: int) -> tuple[float, int, int, int]:
    """Pick cell size and font sizes based on the number of models."""
    if n <= 6:
        return 0.90, 9, 9, 13
    if n <= 10:
        return 0.72, 8, 8, 12
    if n <= 14:
        return 0.60, 7, 7, 11
    if n <= 20:
        return 0.50, 6, 7, 11
    return max(0.42, 10.0 / n), 5, 6, 10


def _max_line_chars(labels: list[str]) -> int:
    return max(len(line) for label in labels for line in label.split("\n"))


def plot_pairwise_matrix(
    result: PairwiseResult,
    title: str | None = None,
    save_path: str | None = None,
) -> Figure:
    """Render the pairwise win-rate matrix as a heatmap."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    n = len(result.models)
    matrix = build_win_rate_matrix(result)
    se_matrix = build_se_matrix(result)
    # Models are already ordered by overall win rate (top-winning first) inside
    # compute_pairwise, so every output format agrees on ordering.
    labels = [m.label for m in result.models]

    cell_size, annot_font, label_font, title_font = _scale_for_n(n)

    # Estimate label extent in inches.  Conservative per-character width at
    # matplotlib's default 100 DPI; this is used only for margin computation.
    char_w = label_font * 0.008
    max_chars = _max_line_chars(labels)
    label_w = max_chars * char_w
    # Rotated 45° x-labels project label_w * sin(45°) onto the vertical axis.
    rotated_projection = label_w * math.sin(math.radians(45))

    heatmap_in = cell_size * n
    left_in = max(label_w + _LEFT_PAD, 1.1)
    # Last column's rotated label overhangs to the right; reserve room for it.
    right_in = max(rotated_projection, _RIGHT_PAD)
    top_in = rotated_projection + _TITLE_PAD + (0.35 if title else 0.0)
    footer_in = _FOOTER_HEIGHT

    # Footer needs a minimum width for legend + stat columns to read cleanly,
    # independent of how narrow the heatmap is.
    footer_width_in = max(heatmap_in, _MIN_FOOTER_WIDTH)
    fig_w = max(left_in + heatmap_in + right_in, left_in + footer_width_in + _RIGHT_PAD)
    fig_h = top_in + heatmap_in + footer_in

    # Continuous colormap built from position-anchored stops.
    positions = [p for p, _ in _CMAP_STOPS]
    colors = [c for _, c in _CMAP_STOPS]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pairwise", list(zip(positions, colors, strict=True))
    )
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

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
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes(
            (
                left_in / fig_w,
                footer_in / fig_h,
                heatmap_in / fig_w,
                heatmap_in / fig_h,
            )
        )

        # --- N×N axis setup ---
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)  # inverted so row 0 is at top
        ax.set_aspect("equal")
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # --- Draw cells as rounded boxes ---
        gap = 0.06  # fraction of cell used as gap between neighbors
        inset = gap / 2
        side = 1.0 - gap
        rounding = side * 0.18

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if i == j or np.isnan(val):
                    color = _DIAGONAL
                    annotate = False
                else:
                    color = cmap(norm(val))
                    annotate = True

                ax.add_patch(
                    FancyBboxPatch(
                        (j + inset, i + inset),
                        side,
                        side,
                        boxstyle=f"round,pad=0,rounding_size={rounding}",
                        facecolor=color,
                        edgecolor="none",
                        linewidth=0,
                        zorder=2,
                    )
                )

                if annotate:
                    # White text on saturated ends, dark on the beige tie band.
                    text_color = "#ffffff" if abs(val - 0.5) > 0.14 else _BG
                    se_val = se_matrix[i, j]
                    se_font = max(7, int(annot_font * 0.80))
                    ax.text(
                        j + 0.5,
                        i + 0.38,
                        f"{val:.1%}",
                        ha="center",
                        va="center",
                        fontsize=annot_font,
                        fontweight="bold",
                        color=text_color,
                        zorder=3,
                    )
                    if not np.isnan(se_val):
                        ax.text(
                            j + 0.5,
                            i + 0.68,
                            f"({se_val:.1%})",
                            ha="center",
                            va="center",
                            fontsize=se_font,
                            color=text_color,
                            zorder=3,
                        )

        # --- Tick labels ---
        ax.set_xticks([i + 0.5 for i in range(n)])
        ax.set_yticks([i + 0.5 for i in range(n)])
        ax.set_xticklabels(
            labels, rotation=45, ha="left", rotation_mode="anchor", fontsize=label_font
        )
        ax.set_yticklabels(labels, rotation=0, fontsize=label_font)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.tick_params(length=0, pad=4)

        if title:
            title_y = 1 - 0.22 / fig_h
            fig.text(
                0.5,
                title_y,
                title,
                fontsize=title_font,
                fontweight="bold",
                color=_TEXT_LIGHT,
                ha="center",
                va="top",
            )

        # --- Footer: laid out in inches from figure bottom, aligned to heatmap ---
        # Footer spans at least _MIN_FOOTER_WIDTH so labels don't overlap for small N.
        heatmap_left_frac = left_in / fig_w
        heatmap_width_frac = footer_width_in / fig_w

        def _y(inches_from_bottom: float) -> float:
            return inches_from_bottom / fig_h

        # Footer stack (y inches from figure bottom)
        legend_header_in = 1.82
        legend_bar_top_in = 1.66
        legend_bar_bottom_in = 1.52
        legend_tick_in = 1.42
        upper_divider_in = 1.22
        summary_label_in = 1.00
        stat_head_in = 0.68
        stat_value_in = 0.34

        # --- Gradient legend bar ---
        bar_width_in = min(footer_width_in * 0.55, 4.5)
        bar_left_in = left_in + (footer_width_in - bar_width_in) / 2
        bar_left_frac = bar_left_in / fig_w
        bar_width_frac = bar_width_in / fig_w
        bar_height_frac = (legend_bar_top_in - legend_bar_bottom_in) / fig_h

        cax = fig.add_axes(
            (bar_left_frac, _y(legend_bar_bottom_in), bar_width_frac, bar_height_frac)
        )
        cax.imshow(
            np.linspace(0, 1, 256)[np.newaxis, :],
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=(0.0, 1.0, 0.0, 1.0),
        )
        cax.set_xticks([])
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_visible(False)

        # Header above bar
        fig.text(
            bar_left_frac + bar_width_frac / 2,
            _y(legend_header_in),
            "WIN RATE",
            fontsize=8,
            fontweight="bold",
            color=_TEXT_DIM,
            ha="center",
            va="center",
        )

        # Tick labels under bar
        for frac, label, ha in (
            (0.0, "0% — row loses", "left"),
            (0.5, "50% — tie", "center"),
            (1.0, "100% — row wins", "right"),
        ):
            fig.text(
                bar_left_frac + frac * bar_width_frac,
                _y(legend_tick_in),
                label,
                fontsize=7,
                color=_TEXT_DIM,
                ha=ha,
                va="center",
            )

        # --- Divider above SUMMARY ---
        fig.add_artist(
            Line2D(
                [heatmap_left_frac, heatmap_left_frac + heatmap_width_frac],
                [_y(upper_divider_in), _y(upper_divider_in)],
                transform=fig.transFigure,
                color=_DIVIDER,
                linewidth=0.6,
            )
        )

        # --- SUMMARY heading ---
        fig.text(
            heatmap_left_frac,
            _y(summary_label_in),
            "SUMMARY",
            fontsize=8,
            fontweight="bold",
            color=_TEXT_DIM,
            va="center",
        )

        # --- Summary stats ---
        from olmo_eval.analysis.eval_power import (
            minimum_detectable_effect,
            required_sample_size,
        )

        # Median paired-difference variance across pairs — representative omega^2
        # for the eval-sizing stats below.
        pair_vars = sorted(p.var_paired_diff for p in result.pairs)
        median_var = pair_vars[len(pair_vars) // 2] if pair_vars else 0.0
        shared_n = result.instance_count

        # MDE at the matrix's shared-instance count.
        mde: float | None
        if shared_n > 0 and median_var > 0:
            mde = minimum_detectable_effect(n=shared_n, omega2=median_var, alpha=0.05, power=0.80)
            mde_str = f"{mde:.1%}"
        else:
            mde = None
            mde_str = "—"

        # Sample size needed to resolve a 3-percentage-point gap.
        n_for_3pp: int | None
        if median_var > 0:
            n_for_3pp = required_sample_size(mde=0.03, omega2=median_var)
            n_for_3pp_str = f"{n_for_3pp:,}"
        else:
            n_for_3pp = None
            n_for_3pp_str = "—"

        # Effective sample size per pair:
        #     n_eff = n_shared × (σ_A² + σ_B²) / Var(d)
        # High ratio means pairing compresses uncertainty; near 1× means
        # comparisons inherit the full unpaired noise.
        n_eff_values: list[int] = []
        for p in result.pairs:
            if p.var_paired_diff > 0 and p.var_marginal_sum > 0:
                n_eff_values.append(round(shared_n * p.var_marginal_sum / p.var_paired_diff))
        n_eff_values.sort()
        median_n_eff: int | None = n_eff_values[len(n_eff_values) // 2] if n_eff_values else None
        median_n_eff_str = f"{median_n_eff:,}" if median_n_eff is not None else "—"

        # Tier helpers — good / ok / bad thresholds.
        def _tier_mde(v: float | None) -> str:
            if v is None:
                return _TIER_NEUTRAL
            if v <= 0.03:
                return _TIER_GOOD
            if v <= 0.10:
                return _TIER_OK
            return _TIER_BAD

        def _tier_shared_n(v: int) -> str:
            if v >= 500:
                return _TIER_GOOD
            if v >= 100:
                return _TIER_OK
            return _TIER_BAD

        def _tier_n_for_target(required: int | None, baseline: int) -> str:
            if required is None or baseline <= 0:
                return _TIER_NEUTRAL
            ratio = required / baseline
            if ratio <= 1.0:
                return _TIER_GOOD
            if ratio <= 3.0:
                return _TIER_OK
            return _TIER_BAD

        def _tier_n_eff(eff: int | None, baseline: int) -> str:
            if eff is None or baseline <= 0:
                return _TIER_NEUTRAL
            ratio = eff / baseline
            if ratio >= 2.0:
                return _TIER_GOOD
            if ratio >= 1.2:
                return _TIER_OK
            return _TIER_BAD

        stats: list[tuple[str, str, str]] = [
            ("n for 3pp MDE", n_for_3pp_str, _tier_n_for_target(n_for_3pp, shared_n)),
            ("median n_eff", median_n_eff_str, _tier_n_eff(median_n_eff, shared_n)),
            ("shared n / pair", str(shared_n), _tier_shared_n(shared_n)),
            ("MDE @ 80% power", mde_str, _tier_mde(mde)),
        ]
        col_w = heatmap_width_frac / len(stats)
        for idx, (heading, value, value_color) in enumerate(stats):
            cx = heatmap_left_frac + idx * col_w
            fig.text(cx, _y(stat_head_in), heading, fontsize=8, color=_TEXT_DIM, va="center")
            fig.text(
                cx,
                _y(stat_value_in),
                value,
                fontsize=13,
                fontweight="bold",
                color=value_color,
                va="center",
            )

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=_BG)
        plt.close(fig)

    return fig

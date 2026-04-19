"""Matplotlib renderer for pairwise win-rate matrices."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from olmo_eval.analysis.pairwise import PairwiseResult, get_se, get_win_rate

if TYPE_CHECKING:
    from matplotlib.colors import ColorType
    from matplotlib.figure import Figure

_BG = "#2b2b2b"
_TEXT_LIGHT = "#e0e0e0"
_TEXT_DIM = "#999999"
_DIVIDER = "#444444"
_DIAGONAL = "#3a3a3a"

_TIER_GOOD = "#5aa380"
_TIER_OK = "#e2b05a"
_TIER_BAD = "#c85d3b"
_TIER_NEUTRAL = _TEXT_LIGHT

# Win-rate colormap anchors.
_CMAP_STOPS: list[tuple[float, str]] = [
    (0.00, "#8f2f18"),
    (0.20, "#c85d3b"),
    (0.40, "#e2a48c"),
    (0.50, "#d4cfc4"),
    (0.60, "#a8d1bb"),
    (0.80, "#5aa380"),
    (1.00, "#1f6b4f"),
]

_FOOTER_HEIGHT = 1.58
_RIGHT_PAD = 0.35
_LEFT_PAD = 0.35
_TITLE_PAD = 0.55
# Keep the footer readable when the matrix is narrow.
_MIN_FOOTER_WIDTH = 6.0
_LEGEND_GAP = 0.28
_LEGEND_BAR_WIDTH = 0.26
_LEGEND_LABEL_GAP = 0.08
_LEGEND_LABEL_WIDTH = 0.45
_LEGEND_HEIGHT_FRACTION = 0.62
_LEGEND_MIN_HEIGHT = 1.20
_NONSIG_BLEND = 0.55
_NONSIG_BORDER_WIDTH = 0.8
_SIGNIFICANCE_Z = 2.0


def build_win_rate_matrix(result: PairwiseResult) -> np.ndarray:
    """Return the NxN win-rate matrix with NaN on the diagonal."""
    n = len(result.models)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_win_rate(result.pairs, i, j)
    return matrix


def build_se_matrix(result: PairwiseResult) -> np.ndarray:
    """Return the NxN standard-error matrix with NaN on the diagonal."""
    n = len(result.models)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = get_se(result.pairs, i, j)
    return matrix


def _scale_for_n(n: int) -> tuple[float, int, int, int]:
    """Choose cell and font sizes from matrix size."""
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
    # The result is already row-sorted; keep every output format aligned.
    labels = [m.label for m in result.models]

    cell_size, annot_font, label_font, title_font = _scale_for_n(n)

    char_w = label_font * 0.008
    max_chars = _max_line_chars(labels)
    label_w = max_chars * char_w
    # Reserve vertical space for rotated x-labels.
    rotated_projection = label_w * math.sin(math.radians(45))
    legend_right_in = (
        _LEGEND_GAP + _LEGEND_BAR_WIDTH + _LEGEND_LABEL_GAP + _LEGEND_LABEL_WIDTH + _RIGHT_PAD
    )

    heatmap_in = cell_size * n
    left_in = max(label_w + _LEFT_PAD, 1.1)
    right_in = max(rotated_projection, legend_right_in, _RIGHT_PAD)
    top_in = rotated_projection + _TITLE_PAD + (0.35 if title else 0.0)
    footer_in = _FOOTER_HEIGHT

    plot_section_width_in = left_in + heatmap_in + right_in
    footer_width_in = max(heatmap_in, _MIN_FOOTER_WIDTH)
    footer_section_width_in = footer_width_in + _LEFT_PAD + _RIGHT_PAD
    fig_w = max(plot_section_width_in, footer_section_width_in)
    fig_h = top_in + heatmap_in + footer_in

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
        # Center the matrix/legend panel and the footer independently.
        plot_section_left_in = (fig_w - plot_section_width_in) / 2
        heatmap_left_in = plot_section_left_in + left_in
        ax = fig.add_axes(
            (
                heatmap_left_in / fig_w,
                footer_in / fig_h,
                heatmap_in / fig_w,
                heatmap_in / fig_h,
            )
        )

        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)  # row 0 at top
        ax.set_aspect("equal")
        ax.set_facecolor(_BG)
        for spine in ax.spines.values():
            spine.set_visible(False)

        gap = 0.06  # cell padding
        inset = gap / 2
        side = 1.0 - gap
        rounding = side * 0.18
        neutral_rgba = mcolors.to_rgba(cmap(norm(0.5)))
        nonsig_edge_rgba = mcolors.to_rgba(_TEXT_DIM, alpha=0.55)

        def _is_nonsignificant(win_rate: float, se: float) -> bool:
            return abs(win_rate - 0.5) <= _SIGNIFICANCE_Z * se

        def _blend_with_neutral(color: ColorType) -> tuple[float, float, float, float]:
            r, g, b, a = mcolors.to_rgba(color)
            return (
                (1.0 - _NONSIG_BLEND) * r + _NONSIG_BLEND * neutral_rgba[0],
                (1.0 - _NONSIG_BLEND) * g + _NONSIG_BLEND * neutral_rgba[1],
                (1.0 - _NONSIG_BLEND) * b + _NONSIG_BLEND * neutral_rgba[2],
                (1.0 - _NONSIG_BLEND) * a + _NONSIG_BLEND * neutral_rgba[3],
            )

        def _text_color_for_fill(fill_color: ColorType) -> str:
            r, g, b = mcolors.to_rgb(fill_color)
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return _BG if luminance >= 0.66 else _TEXT_LIGHT

        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                if i == j or np.isnan(val):
                    color = _DIAGONAL
                    annotate = False
                    se_val = np.nan
                    edgecolor = "none"
                    linewidth = 0.0
                else:
                    se_val = se_matrix[i, j]
                    color = cmap(norm(val))
                    edgecolor = "none"
                    linewidth = 0.0
                    if not np.isnan(se_val) and _is_nonsignificant(val, se_val):
                        color = _blend_with_neutral(color)
                        edgecolor = nonsig_edge_rgba
                        linewidth = _NONSIG_BORDER_WIDTH
                    annotate = True

                ax.add_patch(
                    FancyBboxPatch(
                        (j + inset, i + inset),
                        side,
                        side,
                        boxstyle=f"round,pad=0,rounding_size={rounding}",
                        facecolor=color,
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                        zorder=2,
                    )
                )

                if annotate:
                    text_color = _text_color_for_fill(color)
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

        footer_left_in = (fig_w - footer_width_in) / 2
        footer_left_frac = footer_left_in / fig_w
        footer_width_frac = footer_width_in / fig_w
        footer_center_frac = (footer_left_in + footer_width_in / 2) / fig_w

        def _y(inches_from_bottom: float) -> float:
            return inches_from_bottom / fig_h

        upper_divider_in = 1.26
        summary_label_in = 0.95
        stat_head_in = 0.60
        stat_value_in = 0.22

        legend_height_in = min(
            max(heatmap_in * _LEGEND_HEIGHT_FRACTION, _LEGEND_MIN_HEIGHT),
            max(heatmap_in - 0.35, _LEGEND_MIN_HEIGHT),
        )
        legend_bottom_in = footer_in + (heatmap_in - legend_height_in) / 2
        legend_left_in = heatmap_left_in + heatmap_in + _LEGEND_GAP
        cax = fig.add_axes(
            (
                legend_left_in / fig_w,
                _y(legend_bottom_in),
                _LEGEND_BAR_WIDTH / fig_w,
                legend_height_in / fig_h,
            )
        )
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])
        cbar = fig.colorbar(scalar_mappable, cax=cax, orientation="vertical")
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.set_ticklabels(["0%", "50%", "100%"])
        cbar.ax.set_facecolor(_BG)
        cbar.ax.set_frame_on(False)
        cbar.ax.set_title("WIN RATE", fontsize=8, fontweight="bold", color=_TEXT_DIM, pad=8)
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.tick_params(length=0, pad=4, labelsize=7, colors=_TEXT_DIM)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

        fig.add_artist(
            Line2D(
                [footer_left_frac, footer_left_frac + footer_width_frac],
                [_y(upper_divider_in), _y(upper_divider_in)],
                transform=fig.transFigure,
                color=_DIVIDER,
                linewidth=0.6,
            )
        )

        fig.text(
            footer_center_frac,
            _y(summary_label_in),
            "SUMMARY",
            fontsize=8,
            fontweight="bold",
            color=_TEXT_DIM,
            ha="center",
            va="center",
        )

        from olmo_eval.analysis.eval_power import minimum_detectable_effect

        # Use the median pair variance as a single matrix-wide summary term.
        pair_vars = sorted(p.var_paired_diff for p in result.pairs)
        median_var = pair_vars[len(pair_vars) // 2] if pair_vars else 0.0
        shared_n = result.instance_count

        mde: float | None
        if shared_n > 0 and median_var > 0:
            mde = minimum_detectable_effect(n=shared_n, omega2=median_var, alpha=0.05, power=0.80)
            mde_str = f"{mde:.1%}"
        else:
            mde = None
            mde_str = "—"

        # n_eff = n_shared * (sigma_A^2 + sigma_B^2) / Var(d)
        n_eff_values: list[int] = []
        for p in result.pairs:
            if p.var_paired_diff > 0 and p.var_marginal_sum > 0:
                n_eff_values.append(round(shared_n * p.var_marginal_sum / p.var_paired_diff))
        n_eff_values.sort()
        median_n_eff: int | None = n_eff_values[len(n_eff_values) // 2] if n_eff_values else None
        pairing_gain: float | None = (
            median_n_eff / shared_n if median_n_eff is not None and shared_n > 0 else None
        )
        pairing_gain_str = f"{pairing_gain:.1f}x" if pairing_gain is not None else "—"

        contested_values = sorted(p.wins_a + p.wins_b for p in result.pairs)
        median_contested: int | None = (
            contested_values[len(contested_values) // 2] if contested_values else None
        )
        median_contested_rate: float | None = (
            median_contested / shared_n if median_contested is not None and shared_n > 0 else None
        )
        median_contested_str = (
            f"{median_contested_rate:.0%}" if median_contested_rate is not None else "—"
        )

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

        def _tier_pairing_gain(ratio: float | None) -> str:
            if ratio is None:
                return _TIER_NEUTRAL
            if ratio >= 2.0:
                return _TIER_GOOD
            if ratio >= 1.2:
                return _TIER_OK
            return _TIER_BAD

        stats: list[tuple[str, str, str]] = [
            ("shared n", f"{shared_n:,}", _tier_shared_n(shared_n)),
            ("median contested", median_contested_str, _TEXT_LIGHT),
            ("pairing gain", pairing_gain_str, _tier_pairing_gain(pairing_gain)),
            ("MDE @ 80%", mde_str, _tier_mde(mde)),
        ]
        col_w = footer_width_frac / len(stats)
        for idx, (heading, value, value_color) in enumerate(stats):
            cx = footer_left_frac + (idx + 0.5) * col_w
            fig.text(
                cx, _y(stat_head_in), heading, fontsize=8, color=_TEXT_DIM, ha="center", va="center"
            )
            fig.text(
                cx,
                _y(stat_value_in),
                value,
                fontsize=13,
                fontweight="bold",
                color=value_color,
                ha="center",
                va="center",
            )

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=_BG)
        plt.close(fig)

    return fig

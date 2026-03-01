"""Custom Textual widgets for metrics plotting."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from math import ceil, floor, log10

from textual_hires_canvas import Canvas
from textual_plot import PlotWidget
from textual_plot.axis_formatter import AxisFormatter, CategoricalAxisFormatter
from textual_plot.plot_widget import BarPlot, ErrorBarPlot, LinePlot, ScatterPlot


class TimestampAxisFormatter(AxisFormatter):
    """Formatter that maps sample indices to timestamp labels."""

    def __init__(self, timestamps: list[datetime]) -> None:
        self._timestamps = timestamps
        self._n = len(timestamps)

    def get_ticks(self, min_: float, max_: float, max_ticks: int = 8) -> list[float]:
        """Generate tick positions at nice intervals."""
        if self._n == 0:
            return []

        # Use nice interval spacing like NumericAxisFormatter
        delta = max_ - min_
        if delta <= 0:
            return [min_]

        tick_spacing = delta / min(max_ticks, 5)
        power = floor(log10(tick_spacing)) if tick_spacing > 0 else 0
        approx_interval = tick_spacing / 10**power

        # Find nearest nice interval (1, 2, 5, 10, ...)
        nice_intervals = [1.0, 2.0, 5.0, 10.0]
        interval = nice_intervals[0] * 10**power
        for ni in nice_intervals:
            if ni >= approx_interval:
                interval = ni * 10**power
                break

        # Generate ticks
        start = ceil(min_ / interval) * interval
        ticks = []
        tick = start
        while tick <= max_:
            if 0 <= tick < self._n:
                ticks.append(float(tick))
            tick += interval

        return ticks

    def get_labels_for_ticks(self, ticks: Sequence[float]) -> list[str]:
        """Format tick positions as timestamps (HH:MM)."""
        labels = []
        for tick in ticks:
            idx = int(round(tick))
            if 0 <= idx < self._n:
                ts = self._timestamps[idx]
                labels.append(ts.strftime("%H:%M"))
            else:
                labels.append("")
        return labels


# Custom marker for braille legend entries - scattered dot pattern
# Creates a checkerboard-like texture similar to the reference UI
BRAILLE_LEGEND_MARKER = "⠑⠪⠑"


class CleanPlotWidget(PlotWidget):
    """PlotWidget with ticks only on bottom and left axes and custom legend."""

    can_focus = True
    margin_top = 1
    margin_bottom = 2
    margin_left = 6

    DEFAULT_CSS = """
    CleanPlotWidget {
        & > .plot--label {
            color: $text-disabled;
            text-style: none;
        }
    }
    """

    def _render_x_label(self) -> None:
        """Render x-axis label closer to the plot (y=1 instead of y=2)."""
        from textual_hires_canvas import Canvas, TextAlign

        canvas = self.query_one("#plot", Canvas)
        margin = self.query_one("#margin-bottom", Canvas)
        margin.write_text(
            canvas.size.width // 2 + self.margin_left,
            1,
            f"[{self.get_component_rich_style('plot--label')}]{self._x_label}",
            TextAlign.CENTER,
        )

    def _render_plot(self) -> None:
        """Render plot, showing 'No Data' message when empty."""
        from textual_hires_canvas import TextAlign

        super()._render_plot()

        # If no datasets, show "No Data" message centered in plot
        if not self._datasets:
            canvas = self.query_one("#plot", Canvas)
            if canvas._canvas_size is not None:
                cx = canvas._canvas_size.width // 2
                cy = canvas._canvas_size.height // 2
                canvas.write_text(cx, cy, "[bold dim]No Data[/]", TextAlign.CENTER)

    def _update_legend(self) -> None:
        """Update legend with custom braille marker for scatter plots."""
        from rich.text import Text
        from textual.widgets import Static
        from textual_plot import HiResMode
        from textual_plot.plot_widget import LEGEND_LINE, LEGEND_MARKER

        legend = self.query_one("#legend", Static)
        if not legend.display:
            return

        def make_entry(marker: str, style: str, label: str) -> str:
            text = Text(marker)
            text.stylize(style)
            text.append(f" {label}")
            return text.markup

        legend_lines = []
        for label, dataset in zip(self._labels, self._datasets, strict=False):
            if label is None:
                continue

            if isinstance(dataset, LinePlot):
                marker = LEGEND_LINE[dataset.hires_mode]
                style = dataset.line_style
            elif isinstance(dataset, ErrorBarPlot):
                base = (
                    dataset.marker or "┼"
                    if dataset.hires_mode is None
                    else LEGEND_MARKER[dataset.hires_mode]
                )
                marker = base.center(3)
                style = dataset.marker_style
            elif isinstance(dataset, BarPlot):
                marker = "███"
                style = (
                    dataset.bar_style[0]
                    if isinstance(dataset.bar_style, list)
                    else dataset.bar_style
                )
            elif isinstance(dataset, ScatterPlot):
                # Use custom braille marker for better visibility
                if dataset.hires_mode == HiResMode.BRAILLE:
                    marker = BRAILLE_LEGEND_MARKER
                elif dataset.hires_mode is None:
                    marker = dataset.marker.center(3)
                else:
                    marker = LEGEND_MARKER[dataset.hires_mode].center(3)
                style = dataset.marker_style
            else:
                continue

            legend_lines.append(make_entry(marker, style, label))

        for label, vline in zip(self._v_lines_labels, self._v_lines, strict=False):
            if label is not None:
                legend_lines.append(make_entry("│".center(3), vline.line_style, label))

        legend.update(Text.from_markup("\n".join(legend_lines)))
        self._position_legend()

    def _render_x_ticks(self) -> None:
        from textual_hires_canvas import TextAlign

        canvas = self.query_one("#plot", Canvas)
        bottom_margin = self.query_one("#margin-bottom", Canvas)
        bottom_margin.reset()

        if self._x_ticks is None:
            x_ticks, x_labels = self._x_formatter.get_ticks_and_labels(self._x_min, self._x_max)
        else:
            x_ticks = self._x_ticks
            x_labels = self._x_formatter.get_labels_for_ticks(x_ticks)

        for tick, label in zip(x_ticks, x_labels, strict=False):
            if tick < self._x_min or tick > self._x_max:
                continue
            align = TextAlign.CENTER
            x, _ = self.get_pixel_from_coordinate(tick, 0.0)
            if not isinstance(self._x_formatter, CategoricalAxisFormatter):
                if tick == self._x_min:
                    x -= 1
                elif tick == self._x_max:
                    align = TextAlign.RIGHT
            y = self._scale_rectangle.bottom
            new_pixel = self.combine_quad_with_pixel((0, 0, 2, 0), canvas, x, y)
            canvas.set_pixel(
                x, y, new_pixel, style=str(self.get_component_rich_style("plot--axis"))
            )
            bottom_margin.write_text(
                x + self.margin_left,
                0,
                f"[{self.get_component_rich_style('plot--tick')}]{label}",
                align,
            )

    def _render_y_ticks(self) -> None:
        from textual_hires_canvas import TextAlign

        canvas = self.query_one("#plot", Canvas)
        left_margin = self.query_one("#margin-left", Canvas)
        left_margin.reset()

        if self._y_ticks is None:
            y_ticks, y_labels = self._y_formatter.get_ticks_and_labels(self._y_min, self._y_max)
        else:
            y_ticks = self._y_ticks
            y_labels = self._y_formatter.get_labels_for_ticks(y_ticks)

        y_labels = [lbl[: self.margin_left - 1] for lbl in y_labels]

        for tick, label in zip(y_ticks, y_labels, strict=False):
            if tick < self._y_min or tick > self._y_max:
                continue
            _, y = self.get_pixel_from_coordinate(0.0, tick)
            if tick == self._y_min:
                y += 1
            new_pixel = self.combine_quad_with_pixel((0, 0, 0, 2), canvas, 0, y)
            canvas.set_pixel(
                0, y, new_pixel, style=str(self.get_component_rich_style("plot--axis"))
            )
            left_margin.write_text(
                self.margin_left - 2,
                y,
                f"[{self.get_component_rich_style('plot--tick')}]{label}",
                TextAlign.RIGHT,
            )

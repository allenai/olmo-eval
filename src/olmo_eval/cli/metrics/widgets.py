"""Custom Textual widgets for metrics plotting."""

from __future__ import annotations

from textual_hires_canvas import Canvas
from textual_plot import PlotWidget
from textual_plot.axis_formatter import CategoricalAxisFormatter


class CleanPlotWidget(PlotWidget):
    """PlotWidget with ticks only on bottom and left axes."""

    margin_top = 0
    margin_left = 6

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

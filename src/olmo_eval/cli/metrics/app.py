"""Textual application for metrics visualization."""

from __future__ import annotations

import statistics
from typing import Any

from olmo_eval.cli.metrics.config import (
    METRICS,
    METRICS_DB_NAME,
    P95_METRICS,
    SERIES_COLORS,
    DbConfig,
    QueryFilters,
)
from olmo_eval.cli.metrics.data import (
    extract_series_data,
    find_metrics_with_data,
    get_run_label,
    query_samples,
)
from olmo_eval.cli.metrics.utils import compute_p95, extract_metric_value, format_value
from olmo_eval.cli.results.options import get_database_session
from olmo_eval.cli.utils import console


def print_stats_table(samples_by_exp: dict[str, list[Any]]) -> None:
    """Print a statistics summary table to the console."""
    from rich.table import Table

    metrics = find_metrics_with_data(samples_by_exp)
    if not metrics:
        console.print("[dim]No metric data found.[/dim]")
        return

    # Print summary
    console.print()
    for exp_id, samples in samples_by_exp.items():
        if samples:
            label = get_run_label(samples, exp_id)
            console.print(f"[bold cyan]{label}[/bold cyan]: {len(samples)} samples")
    console.print()

    # Build table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Run", style="cyan", no_wrap=True)
    for m in metrics:
        table.add_column(f"Avg {m.table_name}", justify="right")

    for exp_id, samples in samples_by_exp.items():
        if not samples:
            continue
        row = [get_run_label(samples, exp_id)]
        for m in metrics:
            values = [v for s in samples if (v := extract_metric_value(s, m.path)) is not None]
            row.append(format_value(statistics.mean(values)) if values else "-")
        table.add_row(*row)

    console.print(table)
    console.print()


def run_plot_app(
    series_data: dict[str, dict[str, list[float]]],
    samples_by_exp: dict[str, list[Any]],
    metric: str | None,
    refresh_interval: int | None,
    filters: QueryFilters,
    db_config: DbConfig,
) -> None:
    """Run the Textual plotting app."""
    from olmo_eval.cli.metrics.utils import interpolate_series
    from olmo_eval.cli.metrics.widgets import CleanPlotWidget, TimestampAxisFormatter

    try:
        from textual.app import App, ComposeResult
        from textual.containers import Vertical
        from textual.coordinate import Coordinate
        from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, TabPane
        from textual.worker import Worker, WorkerState
        from textual_plot import HiResMode, LegendLocation, PlotWidget
    except ImportError:
        console.print(
            "[red]Error:[/red] Plotting requires textual-plot. "
            "Install with: pip install textual-plot"
        )
        raise SystemExit(1) from None

    # Apply braille OR patch so overlapping series combine dots instead of overwriting
    from olmo_eval.cli.metrics.braille_patch import OVERLAP_STYLE, apply_braille_or_patch

    apply_braille_or_patch()

    # Determine which metrics to plot
    if metric:
        metrics_to_plot = {metric: METRICS[metric]}
    else:
        metrics_to_plot = {
            k: v for k, v in METRICS.items() if any(k in m for m in series_data.values())
        }

    if not metrics_to_plot:
        console.print("[yellow]No metric data to plot.[/yellow]")
        return

    # Group metrics into tabs
    inference_metrics = {"throughput", "latency"}
    gpu_metrics = {"gpu_util", "gpu_mem"}

    inference_to_plot = {k: v for k, v in metrics_to_plot.items() if k in inference_metrics}
    gpu_to_plot = {k: v for k, v in metrics_to_plot.items() if k in gpu_metrics}

    sort_options = list(METRICS.keys()) + ["run", "time"]

    class MetricsApp(App[None]):
        """Textual app for displaying metrics plots."""

        CSS = """
        Screen { layout: vertical; padding: 0; }
        TabbedContent { height: 1fr; }
        ContentSwitcher { height: 1fr; }
        TabPane { height: 1fr; padding: 0; }
        .plots-grid {
            layout: grid;
            grid-size: 2;
            grid-gutter: 0 1;
            height: 1fr;
            width: 100%;
        }
        .plot-container { height: 1fr; min-height: 8; padding: 0; margin: 0; }
        .plot-title {
            text-align: center; color: $text-muted; text-style: dim;
            height: 1; padding: 0; margin: 0;
        }
        PlotWidget {
            height: 1fr;
            & > .plot--axis { color: $text-disabled; }
            & > .plot--tick { color: $text-disabled; text-style: none; }
            #legend {
                border: none;
                background: transparent;
                padding: 0 1;
                offset: 2 1;
                color: $text-muted;
            }
        }
        PlotWidget:focus {
            & > .plot--axis { color: $accent; }
        }
        #stats-table {
            height: auto; min-height: 3; max-height: 12;
            padding: 0; margin-top: 1; scrollbar-gutter: stable;
        }
        Footer { dock: bottom; }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "reset_scales", "Reset"),
            ("R", "reset_all", "Reset All"),
            ("s", "cycle_sort", "Sort"),
            ("S", "toggle_sort_direction", "Asc/Desc"),
            ("space", "toggle_solo", "Solo"),
            ("tab", "focus_next", "Focus"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._sort_index = 0
            self._sort_descending = True
            self._series_data = series_data
            self._samples_by_exp = samples_by_exp
            self._is_fetching = False
            self._last_updated: str | None = None
            self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            self._spinner_index = 0
            self._spinner_timer: Any = None
            self._spinner_min_end: float = 0
            self._hidden_series: set[int] = set()
            self._solo_mode = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False, icon="")
            with TabbedContent():
                if inference_to_plot:
                    with TabPane("Inference", id="perf-tab"), Vertical(classes="plots-grid"):
                        for key, (_, plot_name, _) in inference_to_plot.items():
                            with Vertical(classes="plot-container"):
                                yield Static(plot_name, classes="plot-title")
                                yield CleanPlotWidget(id=f"plot-{key}")
                if gpu_to_plot:
                    with TabPane("GPU", id="gpu-tab"), Vertical(classes="plots-grid"):
                        for key, (_, plot_name, _) in gpu_to_plot.items():
                            with Vertical(classes="plot-container"):
                                yield Static(plot_name, classes="plot-title")
                                yield CleanPlotWidget(id=f"plot-{key}")
            yield DataTable(id="stats-table", cursor_type="row")
            yield Footer()

        def on_mount(self) -> None:
            from datetime import datetime

            self.theme = "atom-one-dark"
            self._last_updated = datetime.now().strftime("%H:%M:%S")
            self._update_title()
            self._update_stats_table()
            self._update_plots()
            self.query_one("#stats-table", DataTable).focus()

            if refresh_interval and refresh_interval > 0:
                self.set_interval(refresh_interval, self._trigger_refresh)

        def _update_plots(self) -> None:
            from textual_plot.axis_formatter import NumericAxisFormatter

            y_labels = {
                "throughput": "tok/s",
                "latency": "sec",
                "gpu_util": "%",
                "gpu_mem": "MB",
            }

            for key in metrics_to_plot:
                plot = self.query_one(f"#plot-{key}", PlotWidget)
                plot.clear()
                plot.set_xlabel("Sample")
                plot.set_ylabel(y_labels.get(key, ""))

                # Collect visible series with their sample indices
                visible_series: list[tuple[int, list[float]]] = []
                visible_samples: list[list[Any]] = []
                for i, (exp_id, samples) in enumerate(self._samples_by_exp.items()):
                    if i in self._hidden_series or not samples:
                        continue
                    label = get_run_label(samples, exp_id)
                    if label in self._series_data and key in self._series_data[label]:
                        visible_series.append((i, self._series_data[label][key]))
                        visible_samples.append(samples)

                all_values = [v for _, vals in visible_series for v in vals]
                if not all_values:
                    plot.set_x_formatter(NumericAxisFormatter())
                    # Reset to default limits so axes don't show stale values
                    plot.set_xlimits(0, 1)
                    plot.set_ylimits(0, 1)
                    # Force refresh to trigger "No Data" message render
                    plot.refresh()
                    continue

                y_min, y_max = min(all_values), max(all_values)
                y_range = max(y_max - y_min, abs(y_max) * 0.1, 1)
                padding = y_range * 0.1

                # Set limits before plotting for proper rendering
                max_x = max(len(vals) for _, vals in visible_series)
                plot.set_xlimits(0, max_x - 1)
                plot.set_ylimits(y_min - padding, y_max + padding)

                # Use timestamp x-axis for single series, numeric for multiple
                if len(visible_series) == 1:
                    timestamps = [s.timestamp for s in visible_samples[0] if s.timestamp]
                    if timestamps:
                        plot.set_x_formatter(TimestampAxisFormatter(timestamps))
                    else:
                        plot.set_x_formatter(NumericAxisFormatter())
                else:
                    plot.set_x_formatter(NumericAxisFormatter())

                for color_idx, values in visible_series:
                    x_vals = list(range(len(values)))
                    color = SERIES_COLORS[color_idx % len(SERIES_COLORS)]

                    # Draw sparse connecting lines first (background)
                    if len(values) >= 2:
                        x_interp, y_interp = interpolate_series(
                            x_vals, values, points_per_segment=2
                        )
                        plot.plot(
                            x=x_interp,
                            y=y_interp,
                            line_style=color,
                            hires_mode=HiResMode.BRAILLE,
                        )

                    # Draw data points as scatter (visible on top)
                    plot.scatter(
                        x=x_vals,
                        y=list(values),
                        marker_style=color,
                        hires_mode=HiResMode.BRAILLE,
                    )

                # Add legend entry for overlap indicator when multiple series visible
                if len(visible_series) > 1:
                    plot.scatter(
                        x=[],
                        y=[],
                        marker_style=OVERLAP_STYLE,
                        hires_mode=HiResMode.BRAILLE,
                        label="Multiple series",
                    )
                    plot.show_legend(is_visible=True, location=LegendLocation.TOPLEFT)

        def _update_title(self) -> None:
            import time

            ts = f" · {self._last_updated}" if self._last_updated else ""
            # Check if we should still show spinner (fetching or minimum duration not met)
            show_spinner = self._is_fetching or time.time() < self._spinner_min_end
            if show_spinner:
                spinner = self._spinner_frames[self._spinner_index]
                self.title = f"Inference Metrics{ts} {spinner}"
            else:
                # Reserve space to prevent text bouncing
                self.title = f"Inference Metrics{ts}  "

        def _animate_spinner(self) -> None:
            import time

            self._spinner_index = (self._spinner_index + 1) % len(self._spinner_frames)
            self._update_title()
            # Stop timer if fetch is done AND minimum display time has passed
            if not self._is_fetching and time.time() >= self._spinner_min_end:
                self._stop_spinner()
                self._update_title()

        def _start_spinner(self) -> None:
            if self._spinner_timer is None:
                self._spinner_timer = self.set_interval(0.1, self._animate_spinner)

        def _stop_spinner(self) -> None:
            if self._spinner_timer is not None:
                self._spinner_timer.stop()
                self._spinner_timer = None

        def _trigger_refresh(self) -> None:
            import time

            if not self._is_fetching:
                self._is_fetching = True
                self._spinner_min_end = time.time() + 1.0  # Show spinner for at least 1 second
                self._update_title()
                self._start_spinner()
                self.run_worker(self._fetch_data, exclusive=True, thread=True)

        def _fetch_data(self) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[Any]]]:
            db = get_database_session(
                db_config.host, db_config.port, METRICS_DB_NAME, db_config.user, db_config.password
            )
            try:
                with db.session() as session:
                    new_samples = query_samples(session, filters)
            finally:
                db.dispose()
            return extract_series_data(new_samples), new_samples

        def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
            from datetime import datetime

            if event.state == WorkerState.SUCCESS and event.worker.result:
                new_series, new_samples = event.worker.result
                if new_series != self._series_data:
                    self._series_data = new_series
                    self._samples_by_exp = new_samples
                    self._update_plots()
                self._update_stats_table()
                self._last_updated = datetime.now().strftime("%H:%M:%S")
            # Set fetching to false; spinner will stop after minimum duration
            self._is_fetching = False

        def _update_stats_table(self) -> None:
            table = self.query_one("#stats-table", DataTable)
            cursor_row = table.cursor_row if table.row_count > 0 else 0
            table.clear(columns=True)

            metrics = find_metrics_with_data(self._samples_by_exp)
            if not metrics:
                return

            # Build columns
            sort_by = sort_options[self._sort_index]
            arrow = "▼" if self._sort_descending else "▲"
            table.add_column(f"Run {arrow}" if sort_by == "run" else "Run  ", key="run")
            table.add_column(f"Time {arrow}" if sort_by == "time" else "Time  ", key="time")
            for m in metrics:
                suffix = f" {arrow}" if sort_by == m.key else "  "
                table.add_column(f"Avg {m.table_name}{suffix}", key=f"avg_{m.key}")
                if m.key in P95_METRICS:
                    table.add_column("p95", key=f"p95_{m.key}")

            # Compute rows
            rows = []
            for idx, (exp_id, samples) in enumerate(self._samples_by_exp.items()):
                if not samples:
                    continue
                label = get_run_label(samples, exp_id)
                ts = samples[0].timestamp
                timestamp = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else ""

                metric_values: dict[str, float | None] = {}
                for m in metrics:
                    vals = [
                        v for s in samples if (v := extract_metric_value(s, m.path)) is not None
                    ]
                    metric_values[m.key] = statistics.mean(vals) if vals else None
                    if m.key in P95_METRICS and vals:
                        metric_values[f"{m.key}_p95"] = compute_p95(vals)

                rows.append((label, metric_values, idx, ts, timestamp))

            # Sort (rows without values always at bottom)
            if sort_by == "run":
                rows.sort(key=lambda r: r[0].lower(), reverse=self._sort_descending)
            elif sort_by == "time":
                with_val = [r for r in rows if r[3] is not None]
                without_val = [r for r in rows if r[3] is None]
                with_val.sort(key=lambda r: r[3], reverse=self._sort_descending)
                rows = with_val + without_val
            elif sort_by in METRICS:
                with_val = [r for r in rows if r[1].get(sort_by) is not None]
                without_val = [r for r in rows if r[1].get(sort_by) is None]
                with_val.sort(key=lambda r: r[1].get(sort_by), reverse=self._sort_descending)
                rows = with_val + without_val

            # Add rows
            for label, metric_values, idx, _ts, timestamp in rows:
                color = SERIES_COLORS[idx % len(SERIES_COLORS)]
                indicator = f"[{color}]□[/]" if idx in self._hidden_series else f"[{color}]■[/]"
                row_data = [f"{indicator} {label}", timestamp]
                for m in metrics:
                    row_data.append(format_value(metric_values.get(m.key)))
                    if m.key in P95_METRICS:
                        row_data.append(format_value(metric_values.get(f"{m.key}_p95")))
                table.add_row(*row_data, key=str(idx))

            if table.row_count > 0:
                table.move_cursor(row=min(cursor_row, table.row_count - 1))

        def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
            if event.row_key and event.row_key.value is not None:
                idx = int(event.row_key.value)
                self._hidden_series.symmetric_difference_update({idx})

                # Look up label from source data instead of parsing display text
                exp_id, samples = list(self._samples_by_exp.items())[idx]
                label = get_run_label(samples, exp_id)

                table = self.query_one("#stats-table", DataTable)
                row_idx = table.get_row_index(event.row_key)
                color = SERIES_COLORS[idx % len(SERIES_COLORS)]
                indicator = f"[{color}]□[/]" if idx in self._hidden_series else f"[{color}]■[/]"
                table.update_cell_at(Coordinate(row_idx, 0), f"{indicator} {label}")
                self._update_plots()

        def action_reset_scales(self) -> None:
            for key in metrics_to_plot:
                self.query_one(f"#plot-{key}", PlotWidget).action_reset_scales()

        def action_reset_all(self) -> None:
            """Reset all charts and settings to initial defaults."""
            self._sort_index = 0
            self._sort_descending = True
            self._hidden_series.clear()
            self._solo_mode = False
            for key in metrics_to_plot:
                self.query_one(f"#plot-{key}", PlotWidget).action_reset_scales()
            self._update_stats_table()
            self._update_plots()

        def action_cycle_sort(self) -> None:
            self._sort_index = (self._sort_index + 1) % len(sort_options)
            self._update_stats_table()

        def action_toggle_sort_direction(self) -> None:
            self._sort_descending = not self._sort_descending
            self._update_stats_table()

        def action_toggle_solo(self) -> None:
            table = self.query_one("#stats-table", DataTable)
            if table.row_count == 0:
                return

            cell_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0))
            if cell_key.row_key.value is None:
                return

            current_idx = int(cell_key.row_key.value)
            all_indices = set(range(len(self._samples_by_exp)))

            if self._solo_mode:
                # Exit solo mode: show all series
                self._hidden_series.clear()
                self._solo_mode = False
            else:
                # Enter solo mode: hide all except current
                self._hidden_series = all_indices - {current_idx}
                self._solo_mode = True

            self._update_stats_table()
            self._update_plots()

    MetricsApp().run()

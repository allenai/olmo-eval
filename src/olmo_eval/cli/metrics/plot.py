"""Plot command for visualizing inference metrics in the terminal."""

from __future__ import annotations

import statistics
from typing import Any

import click

from olmo_eval.cli.results.options import db_options, get_database_session
from olmo_eval.cli.utils import console

# Metrics organized by category
INFERENCE_METRICS = {
    "throughput": ("output_tokens_per_second", "Throughput (tok/s)"),
    "latency": ("mean_latency_s", "Latency (s)"),
}

GPU_METRICS = {
    "gpu_util": ("metadata_.gpu_summary.avg_utilization_pct", "GPU Util (%)"),
    "gpu_mem": ("metadata_.gpu_summary.avg_memory_used_mb", "GPU Mem (MB)"),
}

ALL_METRICS = {**INFERENCE_METRICS, **GPU_METRICS}

METRICS_DB_NAME = "olmo_eval_metrics"

# Colors for different series (hex codes for Rich/Textual compatibility)
SERIES_COLORS = ["#5cb8ff", "#ff7f50", "#3cb371", "#9370db", "#ffd700"]


def _extract_metric_value(sample: Any, metric_path: str) -> float | None:
    """Extract a metric value from a sample using dot-notation path."""
    parts = metric_path.split(".")
    value: Any = sample

    for part in parts:
        if part == "metadata_":
            value = getattr(value, "metadata_", None)
            if value is None:
                return None
        elif isinstance(value, dict):
            value = value.get(part)
            if value is None:
                return None
        else:
            value = getattr(value, part, None)
            if value is None:
                return None

    if isinstance(value, (int, float)):
        return float(value)
    return None


def _query_samples(
    session: Any,
    experiment_ids: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    task_hashes: tuple[str, ...],
) -> dict[str, list[Any]]:
    """Query inference samples with flexible filters."""
    from sqlalchemy import or_, select

    from olmo_eval.storage.backends.postgres.metrics_models import InferenceSample

    stmt = select(InferenceSample)

    # Each filter type: OR within, AND across
    if experiment_ids:
        stmt = stmt.where(InferenceSample.experiment_id.in_(experiment_ids))

    if experiment_groups:
        conditions = [InferenceSample.experiment_group.startswith(g) for g in experiment_groups]
        stmt = stmt.where(or_(*conditions))

    if model_names:
        conditions = [InferenceSample.model_name.startswith(n) for n in model_names]
        stmt = stmt.where(or_(*conditions))

    if model_hashes:
        conditions = [InferenceSample.model_hash.startswith(h) for h in model_hashes]
        stmt = stmt.where(or_(*conditions))

    if task_names:
        conditions = [InferenceSample.task_name.startswith(t) for t in task_names]
        stmt = stmt.where(or_(*conditions))

    if task_hashes:
        conditions = [InferenceSample.task_hash.startswith(h) for h in task_hashes]
        stmt = stmt.where(or_(*conditions))

    stmt = stmt.order_by(InferenceSample.timestamp)
    samples = list(session.execute(stmt).scalars().all())

    # Group by experiment_id
    result: dict[str, list[Any]] = {}
    for sample in samples:
        exp_id = sample.experiment_id or "unknown"
        result.setdefault(exp_id, []).append(sample)

    return result


def _compute_stats(values: list[float]) -> dict[str, float]:
    """Compute statistics for a list of values."""
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values),
    }


def _format_value(val: float, unit: str = "") -> str:
    """Format a value with appropriate precision."""
    if abs(val) >= 1000:
        return f"{val:,.0f}{unit}"
    elif abs(val) >= 10:
        return f"{val:.1f}{unit}"
    elif abs(val) >= 1:
        return f"{val:.2f}{unit}"
    else:
        return f"{val:.3f}{unit}"


def _get_run_label(samples: list[Any], exp_id: str) -> str:
    """Generate a unique label for a run, including experiment ID prefix."""
    exp_prefix = exp_id[:6]

    if not samples:
        return exp_prefix

    sample = samples[0]
    parts = []
    if sample.model_name:
        name = sample.model_name.split("/")[-1]
        parts.append(name[:20])
    if sample.provider_kind:
        parts.append(sample.provider_kind)

    if parts:
        return f"({exp_prefix}) {' / '.join(parts)}"
    return exp_prefix


def _extract_series_data(
    samples_by_exp: dict[str, list[Any]],
) -> dict[str, dict[str, list[float]]]:
    """Extract time series data for all metrics and experiments."""
    result: dict[str, dict[str, list[float]]] = {}

    for exp_id, samples in samples_by_exp.items():
        label = _get_run_label(samples, exp_id)
        result[label] = {}

        for metric_key, (path, _display_name) in ALL_METRICS.items():
            values = []
            for sample in samples:
                val = _extract_metric_value(sample, path)
                if val is not None:
                    values.append(val)
            if values:
                result[label][metric_key] = values

    return result


def _print_stats_table(
    samples_by_exp: dict[str, list[Any]],
) -> None:
    """Print a pivoted statistics summary table."""
    from rich.table import Table

    # Find which metrics have data
    metrics_with_data: list[tuple[str, str, str]] = []
    for key, (path, name) in ALL_METRICS.items():
        for samples in samples_by_exp.values():
            for sample in samples:
                if _extract_metric_value(sample, path) is not None:
                    metrics_with_data.append((key, path, name))
                    break
            else:
                continue
            break

    if not metrics_with_data:
        console.print("[dim]No metric data found.[/dim]")
        return

    # Print summary info
    console.print()
    for exp_id, samples in samples_by_exp.items():
        if not samples:
            continue
        label = _get_run_label(samples, exp_id)
        console.print(f"[bold cyan]{label}[/bold cyan]: {len(samples)} samples")
    console.print()

    # Build pivoted table: rows=experiments, cols=metrics (showing avg)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Run", style="cyan", no_wrap=True)
    for _key, _path, name in metrics_with_data:
        table.add_column(f"Avg {name}", justify="right")

    for exp_id, samples in samples_by_exp.items():
        if not samples:
            continue

        label = _get_run_label(samples, exp_id)
        row = [label]

        for _key, path, _name in metrics_with_data:
            values = [
                v for sample in samples if (v := _extract_metric_value(sample, path)) is not None
            ]
            if values:
                mean = statistics.mean(values)
                row.append(_format_value(mean))
            else:
                row.append("-")

        table.add_row(*row)

    console.print(table)
    console.print()


def _build_stats_table(samples_by_exp: dict[str, list[Any]]) -> Any:
    """Build a single pivoted stats table: rows=experiments, cols=metrics."""
    from rich.table import Table

    # Find which metrics have data
    metrics_with_data: list[tuple[str, str, str]] = []  # (key, path, name)
    for key, (path, name) in ALL_METRICS.items():
        for samples in samples_by_exp.values():
            for sample in samples:
                if _extract_metric_value(sample, path) is not None:
                    metrics_with_data.append((key, path, name))
                    break
            else:
                continue
            break

    if not metrics_with_data:
        return None

    table = Table(show_header=True, header_style="bold", expand=True)
    table.add_column("Run", style="cyan", no_wrap=True)
    for _key, _path, name in metrics_with_data:
        table.add_column(f"Avg {name}", justify="right")

    for exp_id, samples in samples_by_exp.items():
        if not samples:
            continue

        label = _get_run_label(samples, exp_id)
        row = [label]

        for _key, path, _name in metrics_with_data:
            values = [
                v for sample in samples if (v := _extract_metric_value(sample, path)) is not None
            ]
            if values:
                mean = statistics.mean(values)
                row.append(_format_value(mean))
            else:
                row.append("-")

        table.add_row(*row)

    return table


def _run_plot_app(
    series_data: dict[str, dict[str, list[float]]],
    samples_by_exp: dict[str, list[Any]],
    metric: str | None,
) -> None:
    """Run the Textual plotting app."""
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Container, Vertical, VerticalScroll
        from textual.widgets import Footer, Static
        from textual_plot import HiResMode, PlotWidget
    except ImportError:
        console.print(
            "[red]Error:[/red] Plotting requires textual-plot. "
            "Install with: pip install textual-plot"
        )
        raise SystemExit(1) from None

    # Determine which metrics to plot
    if metric:
        metrics_to_plot = {metric: ALL_METRICS[metric]}
    else:
        # Only plot metrics that have data
        metrics_to_plot = {}
        for key, (path, name) in ALL_METRICS.items():
            for label_data in series_data.values():
                if key in label_data:
                    metrics_to_plot[key] = (path, name)
                    break

    if not metrics_to_plot:
        console.print("[yellow]No metric data to plot.[/yellow]")
        return

    n_plots = len(metrics_to_plot)
    series_labels = list(series_data.keys())
    stats_table = _build_stats_table(samples_by_exp)

    class MetricsApp(App[None]):
        """Textual app for displaying metrics plots."""

        CSS = """
        Screen {
            layout: vertical;
            padding: 1 0 0 0;
        }

        #plots-area {
            layout: grid;
            grid-size: 2;
            grid-gutter: 0 1;
            height: 3fr;
        }

        .plot-container {
            height: 1fr;
            min-height: 10;
        }

        .single-plot {
            column-span: 2;
        }

        .plot-title {
            text-align: center;
            text-style: bold;
            color: $text;
            height: 1;
        }

        PlotWidget {
            height: 1fr;
        }

        #legend {
            dock: top;
            height: auto;
            max-height: 4;
            padding: 0 1 1 1;
            layout: grid;
            grid-size: 3;
            grid-gutter: 0 1;
        }

        .legend-item {
            height: 1;
            width: auto;
        }

        #stats-area {
            height: auto;
            max-height: 12;
            padding: 0 1;
        }

        Footer {
            dock: bottom;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "reset_scales", "Reset"),
            ("l", "toggle_legend", "Legend"),
        ]

        theme = "atom-one-dark"

        def compose(self) -> ComposeResult:
            # Legend showing series colors for each experiment (grid layout)
            with Container(id="legend"):
                for i, label in enumerate(series_labels):
                    color = SERIES_COLORS[i % len(SERIES_COLORS)]
                    yield Static(f"[{color}]■[/] {label}", classes="legend-item")

            # Plots area
            with Vertical(id="plots-area"):
                for key, (_path, name) in metrics_to_plot.items():
                    classes = "plot-container single-plot" if n_plots == 1 else "plot-container"
                    with Vertical(classes=classes):
                        yield Static(f"[bold]{name}[/]", classes="plot-title")
                        yield PlotWidget(id=f"plot-{key}")

            # Stats table area
            if stats_table:
                with VerticalScroll(id="stats-area"):
                    yield Static(stats_table, id="stats-table")

            yield Footer()

        def on_mount(self) -> None:
            self.title = "Inference Metrics"

            # Plot data for each metric with appropriate y-axis padding
            for key, (_path, _name) in metrics_to_plot.items():
                plot_widget = self.query_one(f"#plot-{key}", PlotWidget)

                all_values: list[float] = []
                for i, (_label, metrics) in enumerate(series_data.items()):
                    if key in metrics:
                        values = metrics[key]
                        all_values.extend(values)
                        x = list(range(len(values)))
                        color = SERIES_COLORS[i % len(SERIES_COLORS)]
                        plot_widget.plot(
                            x=x, y=values, line_style=color, hires_mode=HiResMode.BRAILLE
                        )

                # Add 10% padding to y-axis for better visibility
                if all_values:
                    y_min, y_max = min(all_values), max(all_values)
                    y_range = y_max - y_min if y_max > y_min else abs(y_max) * 0.1 or 1
                    padding = y_range * 0.1
                    plot_widget.set_ylimits(y_min - padding, y_max + padding)

        def action_reset_scales(self) -> None:
            """Reset scales on all plots."""
            for key in metrics_to_plot:
                plot_widget = self.query_one(f"#plot-{key}", PlotWidget)
                plot_widget.action_reset_scales()

        def action_toggle_legend(self) -> None:
            """Toggle legend visibility."""
            try:
                legend = self.query_one("#legend", Static)
                legend.visible = not legend.visible
            except Exception:
                pass

    app = MetricsApp()
    app.run()


@click.command()
@click.option(
    "--experiment",
    "-e",
    "experiment_ids",
    multiple=True,
    help="Experiment ID(s) to plot.",
)
@click.option(
    "--experiment-group",
    "-G",
    "experiment_groups",
    multiple=True,
    help="Experiment group prefix(es) to filter.",
)
@click.option(
    "--model",
    "-m",
    "model_names",
    multiple=True,
    help="Model name prefix(es) to filter.",
)
@click.option(
    "--model-hash",
    "-M",
    "model_hashes",
    multiple=True,
    help="Model hash prefix(es) to filter.",
)
@click.option(
    "--task",
    "-t",
    "task_names",
    multiple=True,
    help="Task name prefix(es) to filter.",
)
@click.option(
    "--task-hash",
    "-T",
    "task_hashes",
    multiple=True,
    help="Task hash prefix(es) to filter.",
)
@click.option(
    "--metric",
    "metric",
    type=click.Choice(list(ALL_METRICS.keys())),
    default=None,
    help="Focus on a single metric (default: show all).",
)
@click.option(
    "--stats-only",
    is_flag=True,
    help="Show only statistics table, no plots.",
)
@db_options
def plot(
    experiment_ids: tuple[str, ...],
    experiment_groups: tuple[str, ...],
    model_names: tuple[str, ...],
    model_hashes: tuple[str, ...],
    task_names: tuple[str, ...],
    task_hashes: tuple[str, ...],
    metric: str | None,
    stats_only: bool,
    db_host: str,
    db_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
) -> None:
    """Plot inference metrics for evaluation runs.

    Shows how provider performance metrics evolve over the course of an
    evaluation run. Filter by experiment, model, task, or experiment group.

    Interactive controls:
      - Mouse scroll to zoom
      - Mouse drag to pan
      - Press 'r' to reset zoom
      - Press 'q' to quit

    \b
    Metrics:
      throughput   Output tokens per second
      latency      Mean request latency (seconds)
      gpu_util     Average GPU utilization %
      gpu_mem      Average GPU memory used (MB)

    \b
    Examples:
        # Plot metrics for an experiment group
        olmo-eval metrics plot -G my-benchmark

        # Filter by model name
        olmo-eval metrics plot -m OLMo-3

        # Combine filters (AND logic)
        olmo-eval metrics plot -G my-benchmark -t mmlu

        # Focus on throughput only
        olmo-eval metrics plot -G my-benchmark --metric throughput

        # Just show statistics
        olmo-eval metrics plot -G my-benchmark --stats-only
    """
    # Validate at least one filter is provided
    filters = [
        experiment_ids,
        experiment_groups,
        model_names,
        model_hashes,
        task_names,
        task_hashes,
    ]
    if not any(filters):
        raise click.UsageError(
            "At least one filter is required: "
            "--experiment, --experiment-group, --model, --model-hash, --task, or --task-hash"
        )

    with console.status("[bold blue]Fetching metrics..."):
        db = get_database_session(db_host, db_port, METRICS_DB_NAME, db_user, db_password)
        try:
            with db.session() as session:
                samples_by_exp = _query_samples(
                    session,
                    experiment_ids,
                    experiment_groups,
                    model_names,
                    model_hashes,
                    task_names,
                    task_hashes,
                )
        finally:
            db.dispose()

    if not samples_by_exp:
        console.print("[dim]No metrics found for the specified filter(s).[/dim]")
        return

    if stats_only:
        _print_stats_table(samples_by_exp)
    else:
        # Extract series data and run the TUI
        series_data = _extract_series_data(samples_by_exp)
        _run_plot_app(series_data, samples_by_exp, metric)

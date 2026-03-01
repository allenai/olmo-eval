"""Console reporter for pretty-printing metrics using rich."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from ..core.schema import BatchMetrics, RequestMetrics


class ConsoleReporter:
    """Pretty-print metrics to console using rich."""

    def __init__(self) -> None:
        self._console = Console()
        self._verbose = False

    @property
    def reporter_name(self) -> str:
        return "console"

    def configure(self, verbose: bool = False, **kwargs: Any) -> None:
        """Configure the reporter.

        Args:
            verbose: If True, print per-request metrics.
        """
        self._verbose = verbose

    def report_request(self, metrics: RequestMetrics) -> None:
        """Report a single request (only in verbose mode)."""
        if not self._verbose:
            return

        self._console.print(
            f"  Request {metrics.request_id[:8]}... "
            f"prompt={metrics.prompt_tokens} tokens, "
            f"completion={metrics.completion_tokens} tokens, "
            f"latency={metrics.end_to_end_latency_s:.3f}s, "
            f"tps={metrics.tokens_per_second:.1f}"
        )

    def report_batch(self, metrics: BatchMetrics) -> None:
        """Report batch metrics using rich table."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="bold")
        table.add_column("Value", justify="right")

        # Metadata section
        if metrics.model_name:
            table.add_row("Model", metrics.model_name)
        if metrics.batch_hash:
            table.add_row("Batch", f"[dim]{metrics.batch_hash}[/dim]")
        if metrics.experiment_name:
            table.add_row("Experiment", metrics.experiment_name)

        # Requests section
        table.add_row("", "")  # Spacer
        table.add_row("Requests", "")
        table.add_row("  Total", str(metrics.total_requests))
        table.add_row("  Successful", f"[green]{metrics.successful_requests}[/green]")
        if metrics.failed_requests > 0:
            table.add_row("  Failed", f"[red]{metrics.failed_requests}[/red]")
        else:
            table.add_row("  Failed", str(metrics.failed_requests))

        # Tokens section
        table.add_row("", "")
        table.add_row("Tokens", "")
        table.add_row("  Prompt", f"{metrics.total_prompt_tokens:,}")
        table.add_row("  Completion", f"{metrics.total_completion_tokens:,}")

        # Performance section
        table.add_row("", "")
        table.add_row("Performance", "")
        table.add_row("  Wall clock", f"{metrics.wall_clock_time_s:.2f}s")
        table.add_row("  Throughput", f"[cyan]{metrics.output_tokens_per_second:,.1f}[/cyan] tok/s")
        table.add_row("  Latency (mean)", f"{metrics.mean_latency_s:.3f}s")

        # Create panel
        panel = Panel(
            table,
            title="[bold]Inference Metrics[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

        self._console.print()
        self._console.print(panel)

        # Print per-request metrics if verbose
        if self._verbose and metrics.requests:
            self._console.print("\n[bold]Per-request metrics:[/bold]")
            for req in metrics.requests:
                self.report_request(req)
            self._console.print()

    def flush(self) -> None:
        """Flush output."""
        pass

    def shutdown(self) -> None:
        """No cleanup needed."""
        pass

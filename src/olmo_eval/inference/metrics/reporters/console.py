"""Console reporter for pretty-printing metrics."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from ..core.schema import BatchMetrics, RequestMetrics


class ConsoleReporter:
    """Pretty-print metrics to stdout."""

    def __init__(self, output: TextIO | None = None) -> None:
        self._output = output or sys.stdout
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

        self._output.write(
            f"  Request {metrics.request_id[:8]}... "
            f"prompt={metrics.prompt_tokens} tokens, "
            f"completion={metrics.completion_tokens} tokens, "
            f"latency={metrics.end_to_end_latency_s:.3f}s, "
            f"tps={metrics.tokens_per_second:.1f}\n"
        )

    def report_batch(self, metrics: BatchMetrics) -> None:
        """Report batch metrics."""
        self._output.write("\n")
        self._output.write("=" * 60 + "\n")
        self._output.write("Inference Metrics Summary\n")
        self._output.write("=" * 60 + "\n")

        # Print metadata if available
        metadata_lines = []
        if metrics.experiment_name:
            metadata_lines.append(f"Experiment: {metrics.experiment_name}")
        if metrics.experiment_group:
            metadata_lines.append(f"Group: {metrics.experiment_group}")
        if metrics.model_name:
            metadata_lines.append(f"Model: {metrics.model_name}")
        if metrics.task_name:
            metadata_lines.append(f"Task: {metrics.task_name}")
        if metadata_lines:
            for line in metadata_lines:
                self._output.write(f"{line}\n")

        if metrics.tags:
            tag_str = ", ".join(f"{k}={v}" for k, v in metrics.tags.items())
            self._output.write(f"Tags: {tag_str}\n")

        self._output.write("\nRequests:\n")
        self._output.write(f"  Total:      {metrics.total_requests}\n")
        self._output.write(f"  Successful: {metrics.successful_requests}\n")
        self._output.write(f"  Failed:     {metrics.failed_requests}\n")

        self._output.write("\nTokens:\n")
        self._output.write(f"  Prompt:     {metrics.total_prompt_tokens:,}\n")
        self._output.write(f"  Completion: {metrics.total_completion_tokens:,}\n")

        self._output.write("\nPerformance:\n")
        self._output.write(f"  Wall clock:      {metrics.wall_clock_time_s:.2f}s\n")
        self._output.write(f"  Output tok/s:    {metrics.output_tokens_per_second:.1f}\n")

        self._output.write("\nLatency:\n")
        self._output.write(f"  Mean:  {metrics.mean_latency_s:.3f}s\n")

        self._output.write("=" * 60 + "\n\n")

        # Print per-request metrics if verbose
        if self._verbose and metrics.requests:
            self._output.write("Per-request metrics:\n")
            for req in metrics.requests:
                self.report_request(req)
            self._output.write("\n")

    def flush(self) -> None:
        """Flush output."""
        self._output.flush()

    def shutdown(self) -> None:
        """No cleanup needed."""
        pass

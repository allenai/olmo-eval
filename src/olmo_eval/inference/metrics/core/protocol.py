"""Protocol for metrics reporters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .schema import BatchMetrics, RequestMetrics


@runtime_checkable
class MetricsReporter(Protocol):
    """Protocol that all metrics reporters must implement."""

    @property
    def reporter_name(self) -> str:
        """Unique identifier for this reporter type."""
        ...

    def configure(self, **kwargs: Any) -> None:
        """Configure the reporter with runtime options."""
        ...

    def report_request(self, metrics: RequestMetrics) -> None:
        """Report metrics for a single request."""
        ...

    def report_batch(self, metrics: BatchMetrics) -> None:
        """Report aggregated metrics for a batch of requests."""
        ...

    def flush(self) -> None:
        """Flush any buffered metrics."""
        ...

    def shutdown(self) -> None:
        """Clean up resources."""
        ...

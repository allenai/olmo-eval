"""Background monitor for vLLM server Prometheus metrics."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLLMMetricsSnapshot:
    """A point-in-time snapshot of vLLM server metrics."""

    timestamp: str  # ISO format
    fetch_latency_ms: float  # How long the /metrics call took
    parsed_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "fetch_latency_ms": self.fetch_latency_ms,
            "error": self.error,
            "metrics": self.parsed_metrics,
        }


def parse_prometheus_metrics(text: str) -> dict[str, dict[str, Any]]:
    """Parse Prometheus text format using prometheus_client library.

    Args:
        text: Raw Prometheus metrics text.

    Returns:
        Dictionary mapping metric names to {labels: {...}, value: float}.
    """
    metrics: dict[str, dict[str, Any]] = {}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            # Skip Inf/NaN as they're not JSON-serializable
            if math.isinf(sample.value) or math.isnan(sample.value):
                continue

            metrics[sample.name] = {
                "labels": dict(sample.labels),
                "value": sample.value,
            }

    return metrics


class VLLMMetricsMonitor:
    """Background thread that polls vLLM /metrics endpoint.

    Follows the same pattern as GPUMonitor: daemon thread, thread-safe
    collection, graceful shutdown via Event.

    Usage:
        monitor = VLLMMetricsMonitor(
            base_url="http://localhost:8000",
            output_path="/path/to/vllm_metrics.jsonl",
            interval_s=5.0,
        )
        monitor.start()
        # ... run evaluation ...
        snapshots = monitor.stop()
    """

    def __init__(
        self,
        base_url: str,
        output_path: str | Path | None = None,
        interval_s: float = 10.0,
        timeout_s: float = 10.0,
    ) -> None:
        """Initialize the monitor.

        Args:
            base_url: vLLM server URL (e.g., "http://localhost:8000")
            output_path: Optional path to write JSONL metrics file
            interval_s: Polling interval in seconds
            timeout_s: HTTP request timeout
        """
        self._base_url = base_url.rstrip("/")
        # Handle /v1 suffix - metrics endpoint is at root
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[:-3]
        self._metrics_url = f"{self._base_url}/metrics"
        self._output_path = Path(output_path) if output_path else None
        self._interval_s = interval_s
        self._timeout_s = timeout_s

        self._snapshots: list[VLLMMetricsSnapshot] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None

        # Track consecutive failures for logging
        self._consecutive_failures = 0
        self._max_logged_failures = 3

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread is not None:
            return  # Already running

        self._stop_event.clear()
        self._client = httpx.Client(timeout=self._timeout_s)

        # Ensure output directory exists
        if self._output_path:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)

        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="vllm-metrics-monitor",
        )
        self._thread.start()
        logger.info(
            f"Started vLLM metrics monitor: {self._metrics_url} (interval={self._interval_s}s)"
        )

    def stop(self) -> tuple[VLLMMetricsSnapshot, ...]:
        """Stop monitoring and return collected snapshots."""
        if self._thread is None:
            with self._lock:
                return tuple(self._snapshots)

        self._stop_event.set()
        self._thread.join(timeout=self._timeout_s + 2.0)
        self._thread = None

        if self._client:
            self._client.close()
            self._client = None

        with self._lock:
            snapshots = tuple(self._snapshots)

        logger.info(f"Stopped vLLM metrics monitor: collected {len(snapshots)} snapshots")
        return snapshots

    def get_snapshots(self) -> tuple[VLLMMetricsSnapshot, ...]:
        """Get collected snapshots without stopping the monitor."""
        with self._lock:
            return tuple(self._snapshots)

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            snapshot = self._fetch_metrics()

            with self._lock:
                self._snapshots.append(snapshot)

            # Write to file immediately for real-time visibility
            if self._output_path:
                self._write_snapshot(snapshot)

            # Update failure tracking
            if snapshot.error:
                self._consecutive_failures += 1
                if self._consecutive_failures <= self._max_logged_failures:
                    logger.warning(
                        f"vLLM metrics fetch failed ({self._consecutive_failures}): "
                        f"{snapshot.error}"
                    )
                elif self._consecutive_failures == self._max_logged_failures + 1:
                    logger.warning("Suppressing further vLLM metrics fetch warnings")
            else:
                if self._consecutive_failures > self._max_logged_failures:
                    logger.info(
                        f"vLLM metrics fetch recovered after {self._consecutive_failures} failures"
                    )
                self._consecutive_failures = 0

            self._stop_event.wait(timeout=self._interval_s)

    def _fetch_metrics(self) -> VLLMMetricsSnapshot:
        """Fetch metrics from the vLLM server."""
        timestamp = datetime.now(UTC).isoformat()
        start = time.perf_counter()

        if self._client is None:
            return VLLMMetricsSnapshot(
                timestamp=timestamp,
                fetch_latency_ms=0.0,
                error="HTTP client not initialized",
            )

        try:
            response = self._client.get(self._metrics_url)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code == 200:
                raw_metrics = response.text
                parsed = parse_prometheus_metrics(raw_metrics)
                return VLLMMetricsSnapshot(
                    timestamp=timestamp,
                    fetch_latency_ms=elapsed_ms,
                    parsed_metrics=parsed,
                )
            else:
                return VLLMMetricsSnapshot(
                    timestamp=timestamp,
                    fetch_latency_ms=elapsed_ms,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

        except httpx.ConnectError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return VLLMMetricsSnapshot(
                timestamp=timestamp,
                fetch_latency_ms=elapsed_ms,
                error=f"Connection refused: {e}",
            )
        except httpx.TimeoutException as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return VLLMMetricsSnapshot(
                timestamp=timestamp,
                fetch_latency_ms=elapsed_ms,
                error=f"Timeout: {e}",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return VLLMMetricsSnapshot(
                timestamp=timestamp,
                fetch_latency_ms=elapsed_ms,
                error=str(e),
            )

    def _write_snapshot(self, snapshot: VLLMMetricsSnapshot) -> None:
        """Append snapshot to JSONL file."""
        if self._output_path is None:
            return

        try:
            with open(self._output_path, "a") as f:
                f.write(json.dumps(snapshot.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write vLLM metrics: {e}")

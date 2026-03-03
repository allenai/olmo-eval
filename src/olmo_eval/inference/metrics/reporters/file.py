"""File reporter for append-only JSONL output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    from ..core.schema import BatchMetrics, RequestMetrics


class FileReporter:
    """Append metrics as JSON lines to a file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path: Path | None = Path(path) if path else None
        self._file: TextIO | None = None
        self._include_requests = False

    @property
    def reporter_name(self) -> str:
        return "file"

    def configure(
        self,
        path: str | Path | None = None,
        include_requests: bool = False,
        **kwargs: Any,
    ) -> None:
        """Configure the reporter.

        Args:
            path: Path to the output file.
            include_requests: If True, include per-request metrics in batch output.
        """
        if path:
            self._path = Path(path)
        self._include_requests = include_requests

    def _ensure_file(self) -> TextIO:
        """Ensure file is open for writing."""
        if self._file is None:
            if self._path is None:
                raise ValueError("No path configured for file reporter")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        return self._file

    def report_request(self, metrics: RequestMetrics) -> None:
        """Write a single request as JSON line."""
        f = self._ensure_file()
        data = {
            "type": "request",
            "data": metrics.to_dict(),
        }
        f.write(json.dumps(data) + "\n")

    def report_batch(self, metrics: BatchMetrics) -> None:
        """Write batch metrics as JSON line."""
        f = self._ensure_file()
        data = metrics.to_dict(include_requests=self._include_requests)

        output = {
            "type": "batch",
            "data": data,
        }

        f.write(json.dumps(output) + "\n")

    def flush(self) -> None:
        """Flush file buffer."""
        if self._file:
            self._file.flush()

    def shutdown(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

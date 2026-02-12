"""Trace output handling for agent execution."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from agents import set_trace_processors  # type: ignore[import-not-found]
from agents.tracing import BatchTraceProcessor  # type: ignore[import-not-found]

logger = logging.getLogger(__name__)


class FileSpanExporter:
    """Exports traces to per-trace JSONL files.

    Each unique trace_id gets its own file. Since each agent run has a unique
    trace_id, concurrent agents write to separate files with no contention.
    Files are named: {output_dir}/traces/trace_{trace_id}.jsonl
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = os.path.join(output_dir, "traces")

    def _get_trace_id(self, item: Any) -> str:
        """Extract trace_id from an item."""
        if hasattr(item, "trace_id"):
            return str(item.trace_id)
        if hasattr(item, "export"):
            data = item.export()
            if isinstance(data, dict) and "trace_id" in data:
                return str(data["trace_id"])
        return "unknown"

    def export(self, items: list[Any]) -> None:
        """Export spans/traces to per-trace files."""
        # exist_ok=True handles any races safely
        os.makedirs(self._output_dir, exist_ok=True)

        # Group items by trace_id
        by_trace: dict[str, list[str]] = {}
        for item in items:
            trace_id = self._get_trace_id(item)
            if hasattr(item, "export"):
                data = item.export()
            elif hasattr(item, "model_dump"):
                data = item.model_dump()
            else:
                data = {"type": type(item).__name__, "str": str(item)}

            if trace_id not in by_trace:
                by_trace[trace_id] = []
            by_trace[trace_id].append(json.dumps(data))

        # Write each trace's items to its own file (no contention - unique trace_ids)
        for trace_id, lines in by_trace.items():
            file_path = os.path.join(self._output_dir, f"trace_{trace_id}.jsonl")
            with open(file_path, "a") as f:
                f.write("\n".join(lines) + "\n")

    def shutdown(self) -> None:
        """No-op since we don't keep file handles open."""

    def force_flush(self) -> None:
        """No-op since we write and close immediately."""


@lru_cache(maxsize=1)
def configure_trace_output(output_dir: str) -> None:
    """Configure trace output to write per-agent JSONL files.

    This sets up a BatchTraceProcessor with a FileSpanExporter that writes
    each trace to its own file: {output_dir}/traces/trace_{trace_id}.jsonl

    Called once per worker process at startup before concurrent work begins.
    Uses lru_cache to ensure idempotent configuration.

    Args:
        output_dir: Base directory for trace output.
    """
    exporter = FileSpanExporter(output_dir)
    processor = BatchTraceProcessor(exporter)
    set_trace_processors([processor])
    logger.info(f"Agent traces will be written to {output_dir}/traces/")

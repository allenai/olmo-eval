"""Parsing for DeepScholar-Bench eval outputs.

The upstream eval phase writes scores to its output folder. The exact file
layout is confirmed during validation runs; this parser is deliberately
best-effort: it pulls numeric leaves out of any JSON and metric/value rows out
of any CSV, then surfaces the primary DeepScholar metrics and their geometric
mean when present.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from typing import Any

from olmo_eval.evals.external.benchmarks.deepscholar.args import PRIMARY_METRICS

logger = logging.getLogger(__name__)


def flatten_numeric(content: Any, prefix: str = "") -> dict[str, float]:
    """Flatten a nested JSON structure into dotted-key -> float for numeric leaves."""
    metrics: dict[str, float] = {}
    if isinstance(content, dict):
        for key, value in content.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            metrics.update(flatten_numeric(value, child))
    elif isinstance(content, list):
        for idx, value in enumerate(content):
            child = f"{prefix}[{idx}]"
            metrics.update(flatten_numeric(value, child))
    elif isinstance(content, bool):
        pass  # treat bools as non-metrics
    elif isinstance(content, (int, float)):
        metrics[prefix] = float(content)
    return metrics


def parse_metrics_csv(text: str) -> dict[str, float]:
    """Pull metrics from a CSV.

    Handles two common shapes:
      - a long "metric,value" table (any two columns where one is numeric), and
      - a wide table whose numeric columns are averaged across rows.
    """
    metrics: dict[str, float] = {}
    reader = csv.reader(io.StringIO(text))
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return metrics

    header = rows[0]
    data_rows = rows[1:]

    # Long form: two columns of (label, value) where the label is non-numeric and
    # the value parses as float. The non-numeric label disambiguates this from a
    # two-column wide numeric table.
    if len(header) == 2 and data_rows:
        ok = True
        long_metrics: dict[str, float] = {}
        for row in data_rows:
            if len(row) != 2:
                ok = False
                break
            label, raw_value = row[0].strip(), row[1].strip()
            try:
                float(label)  # a numeric "label" means this is not metric,value
                ok = False
                break
            except ValueError:
                pass
            try:
                long_metrics[label] = float(raw_value)
            except ValueError:
                ok = False
                break
        if ok and long_metrics:
            return long_metrics

    # Wide form: average each numeric column.
    columns: dict[str, list[float]] = {name: [] for name in header}
    for row in data_rows:
        for name, cell in zip(header, row, strict=False):
            try:
                columns[name].append(float(cell))
            except (ValueError, TypeError):
                continue
    for name, values in columns.items():
        if values:
            metrics[name.strip()] = sum(values) / len(values)
    return metrics


def compute_geomean(
    metrics: dict[str, float], keys: tuple[str, ...] = PRIMARY_METRICS
) -> float | None:
    """Geometric mean over the named metrics, matched by exact key or dotted suffix.

    Returns None only if a required metric is missing or negative (invalid). A
    zero metric is a valid outcome (e.g. a failed generation), so any zero yields
    a geomean of 0.0.
    """
    values: list[float] = []
    for key in keys:
        match = next(
            (v for k, v in metrics.items() if k == key or k.endswith(f".{key}")),
            None,
        )
        if match is None or match < 0:
            return None
        values.append(match)
    if any(v == 0 for v in values):
        return 0.0
    return math.exp(sum(math.log(v) for v in values) / len(values))

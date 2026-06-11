"""Parse DeepScholar-Bench eval output into metrics.

EXPERIMENTAL / UNVALIDATED. Upstream `eval.main` writes a `results.csv` whose
exact columns have not been confirmed against a real run. This parser makes a
best-effort attempt: it averages numeric columns across rows and computes a
geometric mean over the metric columns (DeepScholar reports a geomean aggregate).
The column->metric mapping must be reconciled with real output before trusting
these numbers. See plans/003_deepscholar_bench.md.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Non-metric columns to exclude from aggregation (best guess; confirm against output).
_NON_METRIC_COLUMNS = {"baseline_name", "mode", "model", "model_name", "file_id", "query_id"}


def parse_results_csv(content: str) -> dict[str, Any]:
    """Parse a DeepScholar `results.csv` into metrics + a geomean aggregate.

    Args:
        content: Raw CSV text written by `eval.main`.

    Returns:
        Dict with `metrics` (dict[str, float]) and `metadata` (dict[str, Any]).
        `metrics` includes per-column means and a `geomean` over positive metrics.
    """
    reader = csv.DictReader(io.StringIO(content))
    rows = list(reader)
    if not rows:
        logger.warning("DeepScholar results.csv had no rows")
        return {"metrics": {}, "metadata": {"num_rows": 0}}

    columns = [c for c in (reader.fieldnames or []) if c not in _NON_METRIC_COLUMNS]

    metrics: dict[str, float] = {}
    for col in columns:
        values: list[float] = []
        for row in rows:
            raw = row.get(col)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                break  # non-numeric column, skip entirely
        else:
            if values:
                metrics[col] = sum(values) / len(values)

    # DeepScholar reports a geometric mean across its metrics. Compute over
    # strictly-positive metric means; zeros/negatives are excluded and noted.
    positive = [v for v in metrics.values() if v > 0]
    if positive:
        metrics["geomean"] = math.exp(sum(math.log(v) for v in positive) / len(positive))

    return {
        "metrics": metrics,
        "metadata": {"num_rows": len(rows), "metric_columns": list(metrics.keys())},
    }

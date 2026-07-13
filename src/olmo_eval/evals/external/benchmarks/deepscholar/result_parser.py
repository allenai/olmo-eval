"""Parsing for DeepScholar-Bench eval outputs.

The upstream eval phase writes one aggregate CSV per metric. This parser targets
that schema, which is pinned by ``DEEPSCHOLAR_REF``.
"""

from __future__ import annotations

import csv
import io
import math

from olmo_eval.evals.external.benchmarks.deepscholar.args import PRIMARY_METRICS


def parse_aggregate_csv(text: str, metric: str | None = None) -> float | None:
    """Read a metric's ``aggregated_results.csv`` (``baseline_name,<metric>`` header).

    Returns the value on the ``deepscholar_base`` row (or the first row), taken from
    the ``metric``-named column when present, else the first other numeric column.
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        return None
    row = next(
        (r for r in rows if (r.get("baseline_name") or "").strip() == "deepscholar_base"),
        rows[0],
    )
    candidates: list[str] = []
    if metric and metric in row:
        candidates.append(metric)
    candidates += [k for k in row if k != "baseline_name" and k not in candidates]
    for key in candidates:
        try:
            return float(row[key])
        except (TypeError, ValueError):
            continue
    return None


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

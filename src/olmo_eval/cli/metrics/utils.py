"""Generic utility functions for metrics processing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence


def extract_metric_value(sample: Any, metric_path: str) -> float | None:
    """Extract a metric value from a sample using dot-notation path."""
    value: Any = sample
    for part in metric_path.split("."):
        value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
        if value is None:
            return None
    return float(value) if isinstance(value, (int, float)) else None


def compute_p95(values: Sequence[float]) -> float:
    """Compute the 95th percentile of a sequence of values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
    return sorted_vals[idx]


def interpolate_series(
    x: Sequence[float], y: Sequence[float], points_per_segment: int = 50
) -> tuple[list[float], list[float]]:
    """Linearly interpolate a series to create denser points for smoother rendering."""
    import numpy as np

    if len(x) < 2:
        return list(x), list(y)
    x_new = np.linspace(x[0], x[-1], (len(x) - 1) * points_per_segment + 1)
    y_new = np.interp(x_new, x, y)
    return x_new.tolist(), y_new.tolist()


def format_value(val: float | None, placeholder: str = "-") -> str:
    """Format a numeric value with appropriate precision."""
    if val is None:
        return placeholder
    if abs(val) >= 1000:
        return f"{val:,.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    if abs(val) >= 1:
        return f"{val:.2f}"
    return f"{val:.3f}"

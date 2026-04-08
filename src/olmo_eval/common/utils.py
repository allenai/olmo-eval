"""Utility functions for evaluation."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any


class Serializable:
    """Mixin that provides to_dict() for dataclasses, excluding None values."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result: dict[str, Any] = {}
        for f in fields(self):  # type: ignore[ty:invalid-argument-type]
            value = getattr(self, f.name)
            if value is None:
                continue
            result[f.name] = _serialize_value(value)
        return result


def _serialize_value(value: Any) -> Any:
    """Serialize a value for JSON output."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if hasattr(value, "value"):  # Enum
        return value.value
    return value


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric (unbiased estimator).

    Pass@k measures the probability that at least one of k samples
    is correct. When n < k, uses k = min(k, n) per standard convention.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass@k

    Returns:
        pass@k probability
    """
    if n == 0:
        return 0.0
    # Clamp k to available samples (standard convention when n < k)
    k = min(k, n)
    if n - c < k:
        return 1.0
    # Use math.prod to avoid overflow for large n
    # pass@k = 1 - C(n-c, k) / C(n, k)
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def compute_pass_pow_k(n: int, c: int, k: int) -> float:
    """Compute pass^k metric (all k trials succeed).

    Pass^k measures automation readiness: the probability that
    k consecutive runs all succeed. Computed as (success_rate)^k.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k value for pass^k

    Returns:
        pass^k probability (success_rate ** k)
    """
    if n == 0:
        return 0.0
    success_rate = c / n
    return success_rate**k

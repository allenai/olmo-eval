"""Metric protocols and implementations."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .types import Response


class Metric(Protocol):
    """Protocol for aggregating scores across responses."""

    @property
    def name(self) -> str:
        """Unique identifier for this metric."""
        ...

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute aggregate metric from scored responses."""
        ...


@dataclass(frozen=True, slots=True)
class AccuracyMetric:
    """Mean accuracy across all responses for a given scorer."""

    name: str = "accuracy"
    scorer_name: str = "exact_match"

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self.scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class F1Metric:
    """Mean F1 score across all responses."""

    name: str = "f1"
    scorer_name: str = "f1"

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self.scorer_name, 0.0) for r in responses)
        return total / len(responses)

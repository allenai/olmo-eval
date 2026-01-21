"""Metric protocols and implementations."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .scorers import (
    BitsPerByteScorer,
    ExactMatchScorer,
    F1Scorer,
    PerplexityScorer,
    Scorer,
)
from .types import Response


def _get_scorer_name(scorer: type[Scorer]) -> str:
    """Extract the default name from a scorer class."""
    # Get the default value from the dataclass field
    for f in scorer.__dataclass_fields__.values():
        if f.name == "name":
            return f.default
    raise ValueError(f"Scorer {scorer} has no 'name' field")


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
    scorer: type[Scorer] = ExactMatchScorer
    _scorer_name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scorer_name", _get_scorer_name(self.scorer))

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self._scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class F1Metric:
    """Mean F1 score across all responses."""

    name: str = "f1"
    scorer: type[Scorer] = F1Scorer
    _scorer_name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scorer_name", _get_scorer_name(self.scorer))

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self._scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class BPBMetric:
    """Mean bits-per-byte of the gold/correct completion.

    For tasks with multiple continuations, this returns the BPB of the
    correct continuation. For single-continuation tasks (like perplexity),
    it returns the BPB of that continuation.
    """

    name: str = "bits_per_byte"
    scorer: type[Scorer] = BitsPerByteScorer
    _scorer_name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scorer_name", _get_scorer_name(self.scorer))

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self._scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class MeanPerplexityMetric:
    """Mean perplexity across all responses."""

    name: str = "perplexity"
    scorer: type[Scorer] = PerplexityScorer
    _scorer_name: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scorer_name", _get_scorer_name(self.scorer))

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        total = sum(r.scores.get(self._scorer_name, 0.0) for r in responses)
        return total / len(responses)

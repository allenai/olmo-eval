"""Metric protocols and implementations."""

from collections.abc import Sequence
from dataclasses import dataclass
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

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = _get_scorer_name(self.scorer)
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class F1Metric:
    """Mean F1 score across all responses."""

    name: str = "f1"
    scorer: type[Scorer] = F1Scorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = _get_scorer_name(self.scorer)
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class BPBMetric:
    """Mean bits-per-byte of the gold/correct completion.

    For tasks with multiple continuations (e.g., multiple choice), this returns
    the BPB of the correct continuation using `instance.metadata["gold_idx"]`.
    For single-continuation tasks (like perplexity), it returns the BPB of that
    continuation.
    """

    name: str = "bits_per_byte"
    scorer: type[Scorer] = BitsPerByteScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer_instance = self.scorer()
        total = 0.0

        for response in responses:
            outputs = response.outputs
            if not outputs:
                continue

            if len(outputs) > 1:
                # Multiple outputs: select the gold/correct continuation
                gold_idx = response.instance.metadata.get("gold_idx")
                if gold_idx is not None and 0 <= gold_idx < len(outputs):
                    output = outputs[gold_idx]
                else:
                    # Fallback to first output if gold_idx not available
                    output = outputs[0]
            else:
                # Single output: use it directly
                output = outputs[0]

            total += scorer_instance.score(response.instance, output)

        return total / len(responses)


@dataclass(frozen=True, slots=True)
class MeanPerplexityMetric:
    """Mean perplexity of the gold/correct completion.

    For tasks with multiple continuations (e.g., multiple choice), this returns
    the perplexity of the correct continuation using `instance.metadata["gold_idx"]`.
    For single-continuation tasks, it returns the perplexity of that continuation.
    """

    name: str = "perplexity"
    scorer: type[Scorer] = PerplexityScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer_instance = self.scorer()
        total = 0.0

        for response in responses:
            outputs = response.outputs
            if not outputs:
                continue

            if len(outputs) > 1:
                # Multiple outputs: select the gold/correct continuation
                gold_idx = response.instance.metadata.get("gold_idx")
                if gold_idx is not None and 0 <= gold_idx < len(outputs):
                    output = outputs[gold_idx]
                else:
                    # Fallback to first output if gold_idx not available
                    output = outputs[0]
            else:
                # Single output: use it directly
                output = outputs[0]

            total += scorer_instance.score(response.instance, output)

        return total / len(responses)

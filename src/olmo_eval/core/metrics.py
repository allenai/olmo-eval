"""Metric protocols and implementations."""

import math
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
    """Aggregate bits-per-byte of the gold/correct completion.

    Computes BPB by summing total logprobs and total bytes across all responses,
    then computing: -total_logprobs / (total_bytes * log(2))

    This byte-weighted approach means longer texts contribute proportionally more
    to the final metric, matching the standard aggregate BPB calculation.

    For tasks with multiple continuations (e.g., multiple choice), this uses
    the correct continuation via `instance.metadata["gold_idx"]`.
    """

    name: str = "bits_per_byte"
    scorer: type[Scorer] = BitsPerByteScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        total_logprobs = 0.0
        total_bytes = 0

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

            if output.logprobs is None:
                continue

            logprobs = [tok["logprob"] for tok in output.logprobs if "logprob" in tok]
            if not logprobs:
                continue

            num_bytes = len(output.text.encode("utf-8"))
            if num_bytes == 0:
                continue

            total_logprobs += sum(logprobs)
            total_bytes += num_bytes

        if total_bytes == 0:
            return 0.0

        return -total_logprobs / (total_bytes * math.log(2))


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

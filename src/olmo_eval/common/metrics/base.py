"""Metric base class and implementations."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

from olmo_eval.common.scorers import (
    BitsPerByteScorer,
    ExactMatchScorer,
    F1Scorer,
    LogprobScorer,
    PerplexityScorer,
    Scorer,
    ToolCallScorer,
)
from olmo_eval.common.types import Response
from olmo_eval.common.utils import compute_pass_at_k, compute_pass_pow_k


@dataclass(frozen=True)
class Metric(ABC):
    """Abstract base class for aggregating scores across responses.

    Subclasses must define:
        - name: str class attribute identifying the metric
        - scorer: type[Scorer] class attribute for the associated scorer
        - compute(): method to aggregate scores from responses
    """

    name: ClassVar[str]
    scorer: ClassVar[type[Scorer]]

    @abstractmethod
    def compute(self, responses: Sequence[Response]) -> float:
        """Compute aggregate metric from scored responses."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {"type": self.__class__.__name__, "name": self.name, "scorer": self.scorer.__name__}


@dataclass(frozen=True, slots=True)
class AccuracyMetric(Metric):
    """Mean accuracy across all responses for a given scorer."""

    name: str = "accuracy"
    scorer: type[Scorer] = ExactMatchScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class F1Metric(Metric):
    """Mean F1 score across all responses."""

    name: str = "f1"
    scorer: type[Scorer] = F1Scorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class BPBMetric(Metric):
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

        scorer = self.scorer()
        weighted_sum = 0.0
        total_bytes = 0

        for response in responses:
            outputs = response.outputs
            if not outputs:
                continue

            # Select gold output
            if len(outputs) > 1:
                gold_idx = response.instance.metadata.get("gold_idx")
                if gold_idx is not None and 0 <= gold_idx < len(outputs):
                    output = outputs[gold_idx]
                else:
                    output = outputs[0]
            else:
                output = outputs[0]

            if output.logprobs is None:
                continue

            num_bytes = len(output.text.encode("utf-8")) if output.text else 0
            if num_bytes == 0:
                continue

            # Use scorer for BPB calculation
            bpb = scorer.score(response.instance, output)
            weighted_sum += bpb * num_bytes
            total_bytes += num_bytes

        if total_bytes == 0:
            return 0.0

        return weighted_sum / total_bytes


@dataclass(frozen=True, slots=True)
class MeanPerplexityMetric(Metric):
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

        scorer = self.scorer()
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

            total += scorer.score(response.instance, output)

        return total / len(responses)


@dataclass(frozen=True, slots=True)
class PassAtKMetric(Metric):
    """Compute pass@k metric across multiple samples per task.

    The probability that at least one of k samples passes.
    """

    name: str = "pass_at_k"
    k: int = 1
    scorer: type[Scorer] = field(kw_only=True)

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute pass@k across all tasks."""
        if not responses:
            return 0.0

        scorer_name = self.scorer().name

        # Group by task ID
        task_results: dict[str, list[float]] = {}
        for r in responses:
            task_id = r.instance.metadata.get("id", "unknown")
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(r.scores.get(scorer_name, 0.0))

        # Compute pass@k for each task
        pass_at_k_values = []
        for scores in task_results.values():
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)  # Count passing
            pass_at_k_values.append(compute_pass_at_k(n, c, min(self.k, n)))

        return sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0


@dataclass(frozen=True, slots=True)
class PassPowKMetric(Metric):
    """Compute pass^k metric (all k trials succeed).

    The probability that k consecutive runs all succeed. Computed as (success_rate)^k.
    """

    name: str = "pass_pow_k"
    k: int = 1
    scorer: type[Scorer] = field(kw_only=True)

    def compute(self, responses: Sequence[Response]) -> float:
        """Compute pass^k across all tasks."""
        if not responses:
            return 0.0

        scorer_name = self.scorer().name

        # Group by task ID
        task_results: dict[str, list[float]] = {}
        for r in responses:
            task_id = r.instance.metadata.get("id", "unknown")
            if task_id not in task_results:
                task_results[task_id] = []
            task_results[task_id].append(r.scores.get(scorer_name, 0.0))

        # Compute pass^k for each task
        pass_pow_k_values = []
        for scores in task_results.values():
            n = len(scores)
            c = sum(1 for s in scores if s > 0.5)  # Count passing
            pass_pow_k_values.append(compute_pass_pow_k(n, c, self.k))

        return sum(pass_pow_k_values) / len(pass_pow_k_values) if pass_pow_k_values else 0.0


@dataclass(frozen=True, slots=True)
class ToolAccuracyMetric(Metric):
    """Mean tool call accuracy across all responses."""

    name: str = "tool_accuracy"
    scorer: type[Scorer] = ToolCallScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        total = sum(r.scores.get(scorer_name, 0.0) for r in responses)
        return total / len(responses)


@dataclass(frozen=True, slots=True)
class CorpusPerplexityMetric(Metric):
    """Corpus-level (token-weighted) perplexity.

    This is the standard token-weighted aggregation across documents.
    Documents that exceed the model's context length are truncated.
    """

    name: str = "corpus_perplexity"
    scorer: type[Scorer] = LogprobScorer

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0

        scorer = self.scorer()
        total_logprob = 0.0
        total_tokens = 0

        for response in responses:
            outputs = response.outputs
            if not outputs:
                continue
            # This should contain the output for the entire doc
            output = outputs[0]
            if output.logprobs is None:
                continue
            total_logprob += scorer.score(response.instance, output)
            total_tokens += len(output.logprobs)

        if total_tokens == 0:
            return 0.0

        avg_logprob = total_logprob / total_tokens
        return math.exp(-avg_logprob)

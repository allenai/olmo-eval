"""Scoring base class and implementations."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from olmo_eval.common.types import Instance, LMOutput


@dataclass(frozen=True)
class Scorer(ABC):
    """Abstract base class for scoring individual outputs.

    Subclasses must define:
        - name: str class attribute identifying the scorer
        - score(): method to compute score for an instance/output pair

    For scorers requiring async execution (e.g., sandboxed code execution),
    extend ExecutionScorer instead and implement ascore().
    """

    name: ClassVar[str]
    requires_async: ClassVar[bool] = False

    @abstractmethod
    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score a single output against the gold answer."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {"type": self.__class__.__name__, **asdict(self)}


@dataclass(frozen=True, slots=True)
class ExactMatchScorer(Scorer):
    """Score 1.0 if extracted answer exactly matches gold, else 0.0."""

    name: str = "exact_match"
    case_sensitive: bool = False
    strip_whitespace: bool = True

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold = instance.gold_answer
        pred = str(output.extracted_answer)
        if self.strip_whitespace:
            gold, pred = gold.strip(), pred.strip()
        if not self.case_sensitive:
            gold, pred = gold.lower(), pred.lower()
        return 1.0 if gold == pred else 0.0


@dataclass(frozen=True, slots=True)
class MultipleChoiceScorer(Scorer):
    """Score multiple choice by comparing selected index/letter."""

    name: str = "multiple_choice"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        # Normalize to uppercase letter
        gold = str(instance.gold_answer).strip().upper()
        pred = str(output.extracted_answer).strip().upper()
        return 1.0 if gold == pred else 0.0


def _normalize_text(text: str) -> str:
    """Normalize text for F1 computation by lowercasing and tokenizing."""
    import string

    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def _compute_f1(pred: str, gold: str) -> float:
    """Compute token-level F1 score between prediction and gold."""
    pred_tokens = _normalize_text(pred).split()
    gold_tokens = _normalize_text(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


@dataclass(frozen=True, slots=True)
class F1Scorer(Scorer):
    """Score using token-level F1 between prediction and gold answer."""

    name: str = "f1"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        return _compute_f1(str(output.extracted_answer), str(instance.gold_answer))


@dataclass(frozen=True, slots=True)
class BitsPerByteScorer(Scorer):
    """Compute bits per byte from logprobs.

    Bits per byte is a measure of language model performance that normalizes
    perplexity by the number of bytes in the text, making it comparable across
    different tokenizers and vocabularies.

    Formula: bits_per_byte = -sum(logprobs) / (num_bytes * log(2))
    """

    name: str = "bits_per_byte"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.logprobs is None:
            return 0.0

        total_logprob = sum(tok.get("logprob", 0.0) for tok in output.logprobs)

        num_bytes = len(output.text.encode("utf-8")) if output.text else 0

        if num_bytes == 0:
            return 0.0

        bits_per_byte = -total_logprob / (num_bytes * math.log(2))

        return bits_per_byte


@dataclass(frozen=True, slots=True)
class PerplexityScorer(Scorer):
    """Compute perplexity from logprobs.

    Perplexity measures how well a language model predicts a sequence,
    defined as exp(-average_logprob).
    """

    name: str = "perplexity"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.logprobs is None:
            return 0.0

        logprobs = [tok["logprob"] for tok in output.logprobs if "logprob" in tok]

        if not logprobs:
            return 0.0

        avg_logprob = sum(logprobs) / len(logprobs)
        perplexity = math.exp(-avg_logprob)

        return perplexity


@dataclass(frozen=True, slots=True)
class LogprobScorer(Scorer):
    """Compute total logprob for a sequence.

    This returns the sum of all token logprobs, useful for comparing
    continuation likelihoods.
    """

    name: str = "logprob"

    def score(self, instance: Instance, output: LMOutput) -> float:
        if output.logprobs is None:
            return float("-inf")

        logprobs = [tok["logprob"] for tok in output.logprobs if "logprob" in tok]

        if not logprobs:
            return float("-inf")

        return sum(logprobs)

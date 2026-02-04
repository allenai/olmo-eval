"""Scoring base class and implementations."""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from .types import Instance, LMOutput
from .utils import _execute_code_unsafe


@dataclass(frozen=True)
class Scorer(ABC):
    """Abstract base class for scoring individual outputs.

    Subclasses must define:
        - name: str class attribute identifying the scorer
        - score(): method to compute score for an instance/output pair
    """

    name: ClassVar[str]

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
    remove_whitespace: bool = False

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.remove_whitespace:
            text = "".join(text.split())
        elif self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    def score(self, instance: Instance, output: LMOutput) -> float:
        if instance.gold_answer is None or output.extracted_answer is None:
            return 0.0
        gold = self._normalize(instance.gold_answer)
        pred = self._normalize(str(output.extracted_answer))
        return 1.0 if gold == pred else 0.0


@dataclass(frozen=True, slots=True)
class ExactMatchFlexScorer(Scorer):
    """Score 1.0 if ANY extracted answer matches ANY gold answer, else 0.0.

    This scorer is more flexible than ExactMatchScorer:
    - Checks all extracted answers from the generation (via output.metadata["all_extracted_answers"])
    - Checks against all gold answers (via instance.metadata["all_gold_answers"])
    - Returns 1.0 if any combination matches

    Falls back to single gold_answer/extracted_answer if metadata is not available.
    """

    name: str = "exact_match_flex"
    case_sensitive: bool = False
    strip_whitespace: bool = True
    remove_whitespace: bool = False

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.remove_whitespace:
            text = "".join(text.split())
        elif self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    def score(self, instance: Instance, output: LMOutput) -> float:
        # Get all gold answers from metadata, or fall back to single gold_answer
        all_gold: list[str] = instance.metadata.get("all_gold_answers", [])
        if not all_gold:
            if instance.gold_answer is None:
                return 0.0
            all_gold = [instance.gold_answer]

        # Get all extracted answers from metadata, or fall back to extracted_answer
        all_extracted: list[str] = output.metadata.get("all_extracted_answers", [])
        if not all_extracted:
            if output.extracted_answer is None:
                return 0.0
            all_extracted = [str(output.extracted_answer)]

        # Check if ANY extracted answer matches ANY gold answer
        for pred in all_extracted:
            pred_norm = self._normalize(str(pred))
            for gold in all_gold:
                gold_norm = self._normalize(str(gold))
                if pred_norm == gold_norm:
                    return 1.0

        return 0.0


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

        total_logprob, num_bytes = 0.0, 0
        for tok in output.logprobs:
            total_logprob += tok.get("logprob", 0.0)
            num_bytes += len(tok.get("bytes", ()))

        if num_bytes == 0:
            return 0.0

        # Compute bits per byte
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


@dataclass(frozen=True, slots=True)
class CodeExecutionScorer(Scorer):
    """Score code by executing it against test cases.

    Note: This scorer requires the instance metadata to contain a 'test' key
    with the test code to run, and the output.extracted_answer to contain
    the complete code to execute.
    """

    name: str = "code_execution"
    timeout: float = 5.0

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score by executing code + tests."""
        if output.extracted_answer is None:
            return 0.0

        test_code = instance.metadata.get("test", "")
        if not test_code:
            return 0.0

        # Combine generated code with tests
        full_code = f"{output.extracted_answer}\n\n{test_code}"

        success, _ = _execute_code_unsafe(full_code, self.timeout)
        return 1.0 if success else 0.0


@dataclass(frozen=True)
class MathVerifyScorer(Scorer):
    """Score math answers using HuggingFace's Math-Verify library.

    This scorer uses symbolic verification to compare mathematical expressions,
    handling LaTeX notation, algebraic expressions, and numerical equivalence.

    See: https://github.com/huggingface/Math-Verify

    Note: Requires the 'math-verify' optional dependency.
    """

    name: str = "math_verify"
    use_latex: bool = True
    use_expr: bool = True
    timeout: int = 10

    def __post_init__(self) -> None:
        if not self.use_latex and not self.use_expr:
            raise ValueError("At least one extraction method must be enabled")

    def _get_extraction_configs(self) -> list:
        """Build extraction configs lazily to avoid import at class definition time."""
        try:
            from math_verify import ExprExtractionConfig, LatexExtractionConfig
        except ImportError as e:
            raise ImportError(
                "MathVerifyScorer requires the 'math-verify' package. "
                "Install it with: pip install math-verify"
            ) from e

        configs = []
        if self.use_latex:
            configs.append(LatexExtractionConfig())
        if self.use_expr:
            configs.append(ExprExtractionConfig())
        return configs

    def _parse_answer(self, text: str) -> list[Any] | None:
        """Parse a text string into a list of mathematical expressions."""
        try:
            from math_verify import parse
        except ImportError as e:
            raise ImportError(
                "MathVerifyScorer requires the 'math-verify' package. "
                "Install it with: pip install math-verify"
            ) from e

        try:
            return parse(
                text,
                extraction_config=self._get_extraction_configs(),
                parsing_timeout=-1,  # timeout handled at verification level
            )
        except Exception:
            return None

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score by comparing parsed mathematical expressions.

        Uses the generated text directly (not extracted_answer) to allow
        Math-Verify to handle its own extraction logic.

        If the instance metadata contains 'all_gold_answers', checks if the
        generated answer matches ANY of them. Otherwise falls back to the
        single gold_answer.
        """
        try:
            from math_verify import verify
        except ImportError as e:
            raise ImportError(
                "MathVerifyScorer requires the 'math-verify' package. "
                "Install it with: pip install math-verify"
            ) from e

        # Get all gold answers from metadata, or fall back to single gold_answer
        all_gold_answers: list[str] = instance.metadata.get("all_gold_answers", [])
        if not all_gold_answers:
            if instance.gold_answer is None:
                return 0.0
            all_gold_answers = [str(instance.gold_answer)]

        generated_text = output.text

        # Parse the generated answer
        parsed_generated = self._parse_answer(generated_text)
        if parsed_generated is None:
            return 0.0

        # Check if generated answer matches ANY of the gold answers
        for gold_text in all_gold_answers:
            parsed_gold = self._parse_answer(str(gold_text))
            if parsed_gold is None:
                continue

            # Use Math-Verify's verify function to compare
            try:
                if verify(parsed_gold, parsed_generated):
                    return 1.0
            except Exception:
                continue

        return 0.0

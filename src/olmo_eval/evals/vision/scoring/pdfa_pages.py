"""Scoring for the PDFA pages tasks: page numbers, page orders and page transcriptions."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.vision.scoring.common import response_text

#: ``output.metadata`` key under which the scorers store their per-example result.
RESULT_KEY = "pdfa_pages_result"
#: Words of the reference and of the answer that a transcription is compared on.
TRANSCRIPTION_WORDS = 300

_NUMBER = re.compile(r"\d+")
_MARKUP = re.compile(r"<[^>]+>|[#*_`|>]+|\$+|\\[()\[\]]")


def parse_page_number(text: str) -> int | None:
    """The first whole number in the response."""
    match = _NUMBER.search(text)
    return int(match.group()) if match else None


def parse_order(text: str) -> list[int]:
    """The whole numbers on the first line of the response that has any."""
    for line in text.splitlines():
        numbers = [int(n) for n in _NUMBER.findall(line)]
        if numbers:
            return numbers
    return []


def pair_accuracy(gold: Sequence[int], predicted: Sequence[int]) -> float:
    """Fraction of item pairs that ``predicted`` puts in the same relative order as ``gold``.

    An item's place is its first occurrence; a pair with an item ``predicted`` leaves out
    counts as wrong. A random order scores 0.5 on average.
    """
    place: dict[int, int] = {}
    for index, item in enumerate(predicted):
        place.setdefault(item, index)
    pairs = right = 0
    for i, first in enumerate(gold):
        for second in gold[i + 1 :]:
            pairs += 1
            right += first in place and second in place and place[first] < place[second]
    return right / pairs if pairs else 0.0


def plain_words(text: str) -> list[str]:
    """The text's words with Markdown and HTML markup removed and case folded."""
    return _MARKUP.sub(" ", text).casefold().split()


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two word sequences (insert, delete, substitute)."""
    previous = list(range(len(b) + 1))
    for i, word_a in enumerate(a, start=1):
        current = [i]
        for j, word_b in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (word_a != word_b))
            )
        previous = current
    return previous[-1]


def transcription_scores(
    reference: str, prediction: str, max_words: int = TRANSCRIPTION_WORDS
) -> dict[str, float]:
    """Compare a transcription with its reference on the first ``max_words`` words of each.

    :returns: ``word_similarity``, one minus the word edit distance over the longer length,
        and ``word_f1``, the F1 of the two bags of words. Both are 0-1.
    """
    ref = plain_words(reference)[:max_words]
    pred = plain_words(prediction)[:max_words]
    longest = max(len(ref), len(pred))
    similarity = 1.0 - edit_distance(ref, pred) / longest if longest else 1.0
    shared = sum((Counter(ref) & Counter(pred)).values())
    f1 = 2 * shared / (len(ref) + len(pred)) if shared else float(not ref and not pred)
    return {"word_similarity": similarity, "word_f1": f1}


def _store(output: LMOutput, result: dict[str, float]) -> None:
    if output.metadata is None:
        output.metadata = {}
    output.metadata[RESULT_KEY] = result


@dataclass(frozen=True, slots=True)
class PageNumberScorer(Scorer):
    """1.0 when the first number in the response is the page ``metadata["page_number"]``."""

    name: str = "pdfa_page_number"

    def score(self, instance: Instance, output: LMOutput) -> float:
        predicted = parse_page_number(response_text(output))
        correct = float(predicted == int(instance.metadata["page_number"]))
        _store(output, {"accuracy": correct, "chance": 1.0 / int(instance.metadata["num_images"])})
        return correct


@dataclass(frozen=True, slots=True)
class PageOrderScorer(Scorer):
    """1.0 when the response lists ``metadata["order"]`` exactly; also stores ``pairs``."""

    name: str = "pdfa_page_order"

    def score(self, instance: Instance, output: LMOutput) -> float:
        gold = [int(n) for n in instance.metadata["order"]]
        predicted = parse_order(response_text(output))
        exact = float(predicted == gold)
        _store(output, {"accuracy": exact, "pairs": pair_accuracy(gold, predicted)})
        return exact


@dataclass(frozen=True, slots=True)
class TranscriptionScorer(Scorer):
    """Word similarity of the response to the gold transcription (:func:`transcription_scores`)."""

    name: str = "pdfa_transcription"

    def score(self, instance: Instance, output: LMOutput) -> float:
        result = transcription_scores(instance.gold_answer or "", response_text(output))
        _store(output, result)
        return result["word_similarity"]


@dataclass(frozen=True)
class PdfaPagesMetric(Metric):
    """Mean of one field of the scorer's result, over all examples or those of one subset.

    The subset is the instances whose ``metadata[field]`` equals ``value``. An example of the
    subset without a result counts as 0.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    key: str = "accuracy"
    field: str | None = None
    value: str | None = None

    def _in_subset(self, response: Response) -> bool:
        return self.field is None or response.instance.metadata.get(self.field) == self.value

    def compute(self, responses: Sequence[Response]) -> float:
        values = [self.compute_instance(r) or 0.0 for r in responses if self._in_subset(r)]
        return sum(values) / len(values) if values else 0.0

    def compute_instance(self, response: Response) -> float | None:
        """The example's own value, or ``None`` outside the subset or without a result."""
        if not self._in_subset(response):
            return None
        for output in response.outputs:
            result = (output.metadata or {}).get(RESULT_KEY)
            if result is not None and self.key in result:
                return float(result[self.key])
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

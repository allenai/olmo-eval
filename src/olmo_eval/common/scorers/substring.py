"""Substring matching scorers."""

from dataclasses import dataclass

from ..types import Instance, LMOutput
from .base import Scorer


@dataclass(frozen=True, slots=True)
class SubstringRecallScorer(Scorer):
    """Substring recall scorer.

    Computes recall as the fraction of gold answer strings that appear
    (case-insensitive substring match) in the prediction.

    This scorer handles both single and multiple gold answers:
    - Single answer: Returns 1.0 if found, 0.0 otherwise
    - Multiple answers (list): Returns fraction of answers found

    Useful for tasks where the model needs to recall specific information,
    such as RULER (NIAH, VT, CWE, FWE) and similar retrieval tasks.
    """

    name: str = "substring_recall"
    case_sensitive: bool = False

    def score(self, instance: Instance, output: LMOutput) -> float:
        """Score recall of gold answers in the prediction.

        Args:
            instance: Instance with gold_answer as list of strings
            output: LMOutput with text prediction

        Returns:
            Fraction of gold answers found in prediction (0.0 to 1.0)
        """
        if instance.gold_answer is None or output.text is None:
            return 0.0

        # Handle both list and single string gold answers
        if isinstance(instance.gold_answer, list):
            gold_answers = instance.gold_answer
        else:
            gold_answers = [str(instance.gold_answer)]

        if len(gold_answers) == 0:
            return 0.0

        # Apply case sensitivity
        prediction = output.text if self.case_sensitive else output.text.lower()
        matches = sum(
            1
            for answer in gold_answers
            if (str(answer) if self.case_sensitive else str(answer).lower()) in prediction
        )

        return matches / len(gold_answers)

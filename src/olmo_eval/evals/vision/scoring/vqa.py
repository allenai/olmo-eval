"""Answer scoring for the single-image QA and counting benchmarks.

The counting scorer lands with the pointing/counting family; the remaining
QA scorers join with the image-QA family. Parsers are vendored from mm_olmo
and behavior-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass

from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput
from olmo_eval.evals.vision.scoring.count_parsing import parse_count


def _response_text(output: LMOutput) -> str:
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""


def _answers(instance: Instance) -> list[str]:
    answers = instance.metadata.get("answers")
    if answers is None:
        answer = instance.metadata.get("answer")
        answers = [] if answer is None else [answer]
    if isinstance(answers, str):
        answers = [answers]
    return list(answers)


@dataclass(frozen=True, slots=True)
class PointCountScorer(Scorer):
    """Counting accuracy for the ``point_count`` style (CountBench/PixMo Count).

    Stores ``{correct, close, valid, pred_count}`` in
    ``output.metadata["point_count_result"]`` for the per-count metrics and
    returns ``correct``.
    """

    name: str = "point_count"

    def score(self, instance: Instance, output: LMOutput) -> float:
        gt = int(instance.metadata["count"])
        pred_count = parse_count(_response_text(output))
        result = {
            "correct": float(gt == pred_count),
            "close": float(abs(gt - pred_count) <= 1),
            "valid": 1.0,
            "pred_count": pred_count,
        }
        if output.metadata is None:
            output.metadata = {}
        output.metadata["point_count_result"] = result
        return result["correct"]

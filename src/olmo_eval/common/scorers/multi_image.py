"""Scorer for the multi-image multiple-choice benchmarks.

Mirrors the per-example scoring of mm_olmo's ``MuirBenchEval`` / ``BLINKEval`` /
``MMIUEval`` evaluators: response cleanup, MMMU-style option-letter parsing, and
exact match against the gold letter (see
:mod:`olmo_eval.common.image_qa.multi_image_mc`).

Required instance metadata: ``answer`` (gold option letter), ``options``
(option contents, letter order), ``example_id`` (stable fallback seed).
"""

from __future__ import annotations

from dataclasses import dataclass

from olmo_eval.common.image_qa import multi_image_mc_score
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


def _response_text(output: LMOutput) -> str:
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""


@dataclass(frozen=True, slots=True)
class MultiImageMcScorer(Scorer):
    """Accuracy of the parsed option letter against the gold letter."""

    name: str = "multi_image_mc"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        return multi_image_mc_score(
            meta["answer"],
            _response_text(output),
            list(meta.get("options") or []),
            stable_id=str(meta.get("example_id", "")),
        )

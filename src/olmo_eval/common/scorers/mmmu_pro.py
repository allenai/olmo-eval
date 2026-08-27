"""Scorer for the MMMU-Pro benchmark (decoupled from the MMMU scorer).

Uses the official MMMU-Pro answer parser (:mod:`olmo_eval.common.image_qa.mmmu_pro`) on the **raw**
model response — no ``clean_prediction`` preprocessing, matching the reference ``evaluate.py``.
Reads ``options``/``answer``/``example_id`` from ``instance.metadata`` and returns 1.0/0.0.
"""

from __future__ import annotations

from dataclasses import dataclass

from olmo_eval.common.image_qa.mmmu_pro import mmmu_pro_score
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


@dataclass(frozen=True, slots=True)
class MmmuProScorer(Scorer):
    """Official MMMU-Pro multiple-choice scoring (1.0 if the parsed letter matches the gold)."""

    name: str = "mmmu_pro"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        return mmmu_pro_score(
            output.text or "",
            list(meta.get("options") or []),
            meta["answer"],
            example_id=str(meta.get("example_id", "")),
        )

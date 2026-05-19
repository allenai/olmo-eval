"""IFBench / IFEval metrics.

All four metrics share :class:`IFEvalScorer`, which writes per-instruction
strict and loose pass lists to ``output.metadata["ifeval"]``. Each metric
aggregates that side-band data along the prompt or instruction axis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers import IFEvalScorer, Scorer
from olmo_eval.common.types import Response


def _iter_results(responses: Sequence[Response], key: str) -> list[list[bool]]:
    out: list[list[bool]] = []
    for response in responses:
        if not response.outputs:
            out.append([])
            continue
        meta = response.outputs[0].metadata or {}
        ifeval = meta.get("ifeval") or {}
        out.append(list(ifeval.get(key, [])))
    return out


def _prompt_level(results: list[list[bool]]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r and all(r))
    return correct / len(results)


def _instruction_level(results: list[list[bool]]) -> float:
    total = sum(len(r) for r in results)
    if total == 0:
        return 0.0
    correct = sum(1 for r in results for v in r if v)
    return correct / total


@dataclass(frozen=True, slots=True)
class IFEvalPromptStrictAccuracy(Metric):
    """Fraction of prompts where every instruction passes under strict scoring."""

    name: ClassVar[str] = "prompt_level_strict_acc"
    scorer: ClassVar[type[Scorer]] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _prompt_level(_iter_results(responses, "strict"))


@dataclass(frozen=True, slots=True)
class IFEvalPromptLooseAccuracy(Metric):
    """Fraction of prompts where every instruction passes under loose scoring."""

    name: ClassVar[str] = "prompt_level_loose_acc"
    scorer: ClassVar[type[Scorer]] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _prompt_level(_iter_results(responses, "loose"))


@dataclass(frozen=True, slots=True)
class IFEvalInstStrictAccuracy(Metric):
    """Fraction of individual instructions passing under strict scoring."""

    name: ClassVar[str] = "inst_level_strict_acc"
    scorer: ClassVar[type[Scorer]] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _instruction_level(_iter_results(responses, "strict"))


@dataclass(frozen=True, slots=True)
class IFEvalInstLooseAccuracy(Metric):
    """Fraction of individual instructions passing under loose scoring."""

    name: ClassVar[str] = "inst_level_loose_acc"
    scorer: ClassVar[type[Scorer]] = IFEvalScorer

    def compute(self, responses: Sequence[Response]) -> float:
        return _instruction_level(_iter_results(responses, "loose"))

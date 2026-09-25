"""Scorer and metrics for CC-OCR's multi-scene OCR track.

:class:`CcOcrScorer` scores one response with the vendored official rules
(:mod:`olmo_eval.common.image_qa.cc_ocr`) and stores the sample's statistics on
``output.metadata["cc_ocr_result"]``. The metrics reduce them the way the official evaluator
does: a sub-dataset's score is its ``macro_f1`` (mean per-image F1), and the track score is the
unweighted mean over sub-datasets. The evaluator's secondary ``micro_f1`` (pooled over a
sub-dataset's images) is reported alongside.

Required ``instance.metadata``: ``dataset`` (the benchmark's ``split`` column) and ``answer``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from olmo_eval.common.image_qa.cc_ocr import score_ocr_sample
from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response

RESULT_KEY = "cc_ocr_result"


def _response_text(output: LMOutput) -> str:
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""


@dataclass(frozen=True, slots=True)
class CcOcrScorer(Scorer):
    """Per-image F1 of the multiset overlap between prediction and reference units."""

    name: str = "cc_ocr"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        stats = score_ocr_sample(_response_text(output), meta["answer"], meta["dataset"])
        if output.metadata is None:
            output.metadata = {}
        output.metadata[RESULT_KEY] = {"dataset": meta["dataset"], **stats}
        return float(stats["f1"])


def _results_by_dataset(responses: Sequence[Response]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for response in responses:
        if not response.outputs:
            continue
        result = (response.outputs[0].metadata or {}).get(RESULT_KEY)
        if result:
            grouped.setdefault(result["dataset"], []).append(result)
    return grouped


def _dataset_value(results: list[dict], kind: str) -> float:
    """One sub-dataset's score, with the official evaluator's smoothing constants."""
    if kind == "macro_f1":
        return sum(r["f1"] for r in results) / (len(results) + 1e-9)
    if kind == "micro_f1":
        right = sum(r["right_num"] for r in results)
        recall = right / (sum(r["gt_num"] for r in results) + 1e-9)
        precision = right / (sum(r["pred_num"] for r in results) + 1e-9)
        return 2 * recall * precision / (recall + precision + 1e-9)
    raise ValueError(f"unknown CC-OCR statistic {kind!r}")


@dataclass(frozen=True)
class CcOcrDatasetMetric(Metric):
    """Macro F1 of one sub-dataset (e.g. ``TotalText``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    dataset: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        results = _results_by_dataset(responses).get(self.dataset)
        return _dataset_value(results, "macro_f1") if results else 0.0


@dataclass(frozen=True)
class CcOcrTrackMetric(Metric):
    """Unweighted mean of the sub-dataset scores, for one statistic."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: str = "macro_f1"

    def compute(self, responses: Sequence[Response]) -> float:
        values = [_dataset_value(r, self.kind) for r in _results_by_dataset(responses).values()]
        return sum(values) / len(values) if values else 0.0

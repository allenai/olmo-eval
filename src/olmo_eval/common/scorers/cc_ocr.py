"""Scorer and metrics for CC-OCR.

:class:`CcOcrScorer` scores one response with the vendored official rules
(:mod:`olmo_eval.common.image_qa.cc_ocr`) and stores the sample's statistics on
``output.metadata["cc_ocr_result"]``. The metrics reduce those statistics the way the
official evaluator does:

* a **sub-dataset** score is ``macro_f1`` (mean per-image F1) on the two OCR tracks, the mean
  sample score on document parsing, and the field-level micro ``f1`` on KIE;
* a **track** score is the unweighted mean of its sub-dataset scores;
* **overall** is the unweighted mean of the four track scores.

The OCR tracks' ``micro_f1`` and KIE's tree-edit ``acc`` are the evaluator's secondary
outputs and are reported alongside.

Required ``instance.metadata``: ``track``, ``dataset``, ``op`` (the benchmark's
``category`` / ``split`` / ``l2-category`` columns) and ``answer``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.image_qa.cc_ocr import (
    score_doc_parsing_sample,
    score_kie_sample,
    score_ocr_sample,
)
from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response

RESULT_KEY = "cc_ocr_result"

OCR_TRACKS = ("multi_scene_ocr", "multi_lan_ocr")
TRACKS = (*OCR_TRACKS, "doc_parsing", "kie")

#: The statistic each track's sub-datasets are ranked by in the paper.
HEADLINE_KIND = {
    "multi_scene_ocr": "macro_f1",
    "multi_lan_ocr": "macro_f1",
    "doc_parsing": "score",
    "kie": "f1",
}


def _response_text(output: LMOutput) -> str:
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""


@dataclass(frozen=True, slots=True)
class CcOcrScorer(Scorer):
    """Per-sample CC-OCR score: image F1 (OCR, KIE) or similarity (document parsing)."""

    name: str = "cc_ocr"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        track, text = meta["track"], _response_text(output)
        if track in OCR_TRACKS:
            stats: dict[str, Any] = score_ocr_sample(text, meta["answer"], track, meta["dataset"])
            score = stats["f1"]
        elif track == "doc_parsing":
            score = score_doc_parsing_sample(text, meta["answer"], meta["op"])
            stats = {"score": score}
        elif track == "kie":
            stats = score_kie_sample(text, meta["answer"])
            score = stats["f1"]
        else:
            raise ValueError(f"unknown CC-OCR track: {track!r}")
        if output.metadata is None:
            output.metadata = {}
        output.metadata[RESULT_KEY] = {"track": track, "dataset": meta["dataset"], **stats}
        return float(score)


def _results_by_dataset(responses: Sequence[Response]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for response in responses:
        if not response.outputs:
            continue
        result = (response.outputs[0].metadata or {}).get(RESULT_KEY)
        if result:
            grouped.setdefault((result["track"], result["dataset"]), []).append(result)
    return grouped


def _dataset_value(results: list[dict], track: str, kind: str) -> float:
    """One sub-dataset's score, with the official evaluator's smoothing constants."""
    n = len(results)
    if kind == "macro_f1":
        return sum(r["f1"] for r in results) / (n + 1e-9)
    if kind == "micro_f1":
        right = sum(r["right_num"] for r in results)
        recall = right / (sum(r["gt_num"] for r in results) + 1e-9)
        precision = right / (sum(r["pred_num"] for r in results) + 1e-9)
        return 2 * recall * precision / (recall + precision + 1e-9)
    if kind == "score":
        return sum(r["score"] for r in results) / n if n else 0.0
    if kind == "f1":
        tp = sum(r["tp"] for r in results)
        return tp / (tp + sum(r["fn_or_fp"] for r in results) / 2 + 1e-6)
    if kind == "acc":
        return sum(r["acc"] for r in results) / (n + 1e-6)
    raise ValueError(f"unknown CC-OCR statistic {kind!r} for track {track!r}")


def _track_value(responses: Sequence[Response], track: str, kind: str) -> float | None:
    values = [
        _dataset_value(results, track, kind)
        for (result_track, _), results in _results_by_dataset(responses).items()
        if result_track == track
    ]
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class CcOcrDatasetMetric(Metric):
    """Score of one sub-dataset (e.g. ``multi_scene_ocr`` / ``TotalText``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    track: str = ""
    dataset: str = ""

    def compute(self, responses: Sequence[Response]) -> float:
        results = _results_by_dataset(responses).get((self.track, self.dataset))
        if not results:
            return 0.0
        return _dataset_value(results, self.track, HEADLINE_KIND[self.track])


@dataclass(frozen=True)
class CcOcrTrackMetric(Metric):
    """Unweighted mean of a track's sub-dataset scores, for one statistic."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    track: str = ""
    #: Statistic to average; defaults to the track's headline one.
    kind: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        return _track_value(responses, self.track, self.kind or HEADLINE_KIND[self.track]) or 0.0


@dataclass(frozen=True)
class CcOcrOverallMetric(Metric):
    """Unweighted mean of the track scores (over the tracks that were run)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        values = [_track_value(responses, track, HEADLINE_KIND[track]) for track in TRACKS]
        present = [v for v in values if v is not None]
        return sum(present) / len(present) if present else 0.0


@dataclass(frozen=True)
class CcOcrJsonParseRateMetric(Metric):
    """Share of KIE responses that parsed as JSON (unparsed ones score as empty)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        parsed = [
            r["parsed"]
            for (track, _), results in _results_by_dataset(responses).items()
            if track == "kie"
            for r in results
        ]
        return sum(parsed) / len(parsed) if parsed else 0.0

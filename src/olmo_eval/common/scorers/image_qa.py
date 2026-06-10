"""Scorers for the Molmo2 image-QA benchmarks.

Each scorer mirrors one metric of the mm_olmo reference evaluators
(``VqaEval``, ``PointCountEval``, ``MathVistaEval``) using the vendored
functions in :mod:`olmo_eval.common.image_qa`.  Scorers read what they need
from ``instance.metadata``:

============================  =================================================
Scorer                        Required instance metadata
============================  =================================================
``VqaScoreScorer``            ``answers`` (list[str])
``AnlsScorer`` / ``EmScorer``  ``answers``
``RelaxedCorrectnessScorer``  ``answers``
``ScifiRelaxedScorer``        ``answers``
``MmmuScorer``                ``answer``, ``question_type``, ``options``,
                              ``example_id``
``RealWorldQaScorer``         ``answer``, ``question_type``
``MathVistaOfflineScorer``    ``answer``, ``question_type``, ``answer_type``,
                              ``choices``, ``precision``
``MathVistaGptScorer``        same as offline, plus ``query``
``PointCountScorer``          ``count``
``Ai2dScorer``                ``answer_idx``, ``option_names``, ``abc_label``,
                              ``has_transparent_box``
============================  =================================================

VQA-family scorers apply the original ``VqaEval`` prediction cleanup
(:func:`clean_prediction`); the counting and MathVista scorers intentionally
do not, matching mm_olmo.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from olmo_eval.common.image_qa import (
    anls_metric,
    clean_prediction,
    extract_answer_quick,
    math_vista_score_from_extraction,
    math_vista_score_offline,
    mmmu_score,
    parse_count,
    real_world_qa_score,
    relaxed_correctness,
    scifi_relaxed_correctness,
    select_mc_option,
    vqa_score,
)
from olmo_eval.common.image_qa.math_vista_offline import create_test_prompt
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.execution import ContextScorer
from olmo_eval.common.types import Instance, LMOutput

if TYPE_CHECKING:
    from olmo_eval.common.execution import ScoringContext

logger = logging.getLogger(__name__)


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
class VqaScoreScorer(Scorer):
    """Official VQA v2 accuracy against the reference answer list."""

    name: ClassVar[str] = "vqa_score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        return float(vqa_score(_answers(instance), pred))


@dataclass(frozen=True, slots=True)
class AnlsScorer(Scorer):
    """ANLS (DocVQA / InfographicVQA), max over reference answers."""

    name: ClassVar[str] = "ansl"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        answers = _answers(instance)
        if not answers:
            return 0.0
        return float(max(anls_metric(ref, pred) for ref in answers))


@dataclass(frozen=True, slots=True)
class EmScorer(Scorer):
    """Case-insensitive exact match against any reference answer."""

    name: ClassVar[str] = "em"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        return float(pred.lower() in [x.lower() for x in _answers(instance)])


@dataclass(frozen=True, slots=True)
class RelaxedCorrectnessScorer(Scorer):
    """ChartQA relaxed accuracy, max over reference answers."""

    name: ClassVar[str] = "relaxed_correctness"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        answers = _answers(instance)
        if not answers:
            return 0.0
        return float(max(relaxed_correctness(ans, pred) for ans in answers))


@dataclass(frozen=True, slots=True)
class ScifiRelaxedScorer(Scorer):
    """Lenient ChartQA relaxed accuracy, max over reference answers."""

    name: ClassVar[str] = "scifi_relaxed_correctness"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        answers = _answers(instance)
        if not answers:
            return 0.0
        return float(max(scifi_relaxed_correctness(ans, pred) for ans in answers))


@dataclass(frozen=True, slots=True)
class MmmuScorer(Scorer):
    """Official MMMU scoring (multiple-choice parsing or open matching)."""

    name: ClassVar[str] = "mmmu_score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        meta = instance.metadata
        return mmmu_score(
            _answers(instance),
            pred,
            question_type=meta["question_type"],
            options=meta.get("options") or [],
            stable_id=str(meta.get("example_id", "")),
        )


@dataclass(frozen=True, slots=True)
class RealWorldQaScorer(Scorer):
    """RealWorldQA: A–D letter match for MC, normalized EM otherwise."""

    name: ClassVar[str] = "real_world_qa_score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        meta = instance.metadata
        return float(real_world_qa_score(meta["answer"], pred, meta["question_type"]))


@dataclass(frozen=True, slots=True)
class MathVistaOfflineScorer(Scorer):
    """MathVista scoring with offline (no-GPT) answer extraction."""

    name: ClassVar[str] = "score"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = _response_text(output).strip()
        meta = instance.metadata
        try:
            correct = math_vista_score_offline(
                pred,
                question_type=meta["question_type"],
                answer_type=meta["answer_type"],
                choices=list(meta.get("choices") or []),
                precision=meta.get("precision"),
                target=meta["answer"],
            )
        except Exception as exc:
            logger.warning(
                "MathVista offline scoring failed for %s: %s", meta.get("example_id"), exc
            )
            return 0.0
        return float(correct)


@dataclass(frozen=True, slots=True)
class PointCountScorer(Scorer):
    """Counting accuracy for the ``point_count`` style (CountBench/PixMo Count).

    Stores ``{correct, close, valid, pred_count}`` in
    ``output.metadata["point_count_result"]`` for the per-count metrics and
    returns ``correct``.
    """

    name: ClassVar[str] = "point_count"

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


@dataclass(frozen=True, slots=True)
class Ai2dScorer(Scorer):
    """AI2D multiple-choice scoring with opaque/transparent routing metadata.

    Stores ``{is_correct, abc_label, has_transparent_box}`` in
    ``output.metadata["ai2d_result"]`` so the two AI2D metrics can route each
    abc-label question to exactly one of the opaque/transparent variants.
    """

    name: ClassVar[str] = "mc_ai2d"

    def score(self, instance: Instance, output: LMOutput) -> float:
        pred = clean_prediction(_response_text(output))
        meta = instance.metadata
        options = list(meta["option_names"])
        pred_idx = select_mc_option(pred, options)
        is_correct = float(pred_idx == meta["answer_idx"])
        if output.metadata is None:
            output.metadata = {}
        output.metadata["ai2d_result"] = {
            "is_correct": is_correct,
            "abc_label": bool(meta["abc_label"]),
            "has_transparent_box": bool(meta["has_transparent_box"]),
        }
        return is_correct


_PROCESS_GPT_CACHE_DIR: list[str] = []


def _default_gpt_cache_dir() -> str | None:
    """Per-run GPT cache dir: env override or a fresh process-local temp dir.

    Never defaults to any pre-existing shared cache; the shared mm_olmo
    ``gpt4-cache`` must not be read or written by this scorer.
    """
    env_dir = os.environ.get("MATHVISTA_GPT_CACHE_DIR")
    if env_dir:
        return env_dir
    if not _PROCESS_GPT_CACHE_DIR:
        _PROCESS_GPT_CACHE_DIR.append(tempfile.mkdtemp(prefix="mathvista-gpt-cache-"))
    return _PROCESS_GPT_CACHE_DIR[0]


@dataclass(frozen=True)
class MathVistaGptScorer(ContextScorer):
    """MathVista scoring with the official GPT-4 answer extraction.

    Follows the official protocol: deterministic short-circuits first, then a
    ``gpt-4-0613`` extraction call (requires ``OPENAI_API_KEY``).  Responses
    are cached under ``cache_dir`` (env ``MATHVISTA_GPT_CACHE_DIR`` or a fresh
    per-process temp dir) so a user's own re-runs are cheap; existing shared
    caches are never touched.
    """

    name: ClassVar[str] = "score"

    model: str = "gpt-4-0613"
    cache_dir: str | None = field(default_factory=_default_gpt_cache_dir)
    cache_only: bool = False
    recompute: bool = False

    async def ascore_with_context(
        self,
        instance: Instance,
        output: LMOutput,
        context: ScoringContext,
    ) -> float:
        from olmo_eval.common.scorers.dense_caption_judge import _cached_gpt_call

        pred = _response_text(output).strip()
        meta = instance.metadata
        choices = list(meta.get("choices") or [])
        question_type = meta["question_type"]
        answer_type = meta["answer_type"]

        extraction = extract_answer_quick(pred, question_type, answer_type, choices)
        if extraction is None:
            try:
                extraction = await _cached_gpt_call(
                    create_test_prompt(meta["query"], pred),
                    model=self.model,
                    cache_dir=self.cache_dir or _default_gpt_cache_dir(),
                    cache_only=self.cache_only,
                    recompute=self.recompute,
                )
            except Exception as exc:
                logger.warning(
                    "MathVista GPT extraction failed for %s: %s", meta.get("example_id"), exc
                )
                return 0.0

        if output.metadata is None:
            output.metadata = {}
        output.metadata["math_vista_extraction"] = extraction

        try:
            correct = math_vista_score_from_extraction(
                extraction,
                question_type=question_type,
                answer_type=answer_type,
                choices=choices,
                precision=meta.get("precision"),
                target=meta["answer"],
            )
        except Exception as exc:
            logger.warning("MathVista GPT scoring failed for %s: %s", meta.get("example_id"), exc)
            return 0.0
        return float(correct)

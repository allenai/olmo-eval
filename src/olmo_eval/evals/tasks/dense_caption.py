"""Pixmo-cap dense-caption evaluation task.

Loads the pixmo-cap test split (2730 images), joins judge reference data from
two on-disk sources, runs a GPT-4o recall+consistency judge, and aggregates
into recall / recall_at_10 / consistency / num_statements / avg metrics.

Environment variables (optional — defaults point to Weka paths):
    DENSE_CAPTION_EVAL_DIR   root of dense_caption_eval/ (contains final-data.json,
                             mturk-eval-statements/, gpt4-cache/)
    MOLMO_DATA_DIR           parent of torch_datasets/ (contains pixmo_images/ and
                             pixmo_datasets/dense-caption-eval/test.jsonl)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.dense_caption_judge import DenseCaptionJudgeScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, SamplingParams
from olmo_eval.evals.tasks.common import Task, register, register_variant

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_DIR = "/weka/oe-training-default/mm-olmo/dense_caption_eval"
_DEFAULT_DATA_HOME = "/weka/oe-training-default/mm-olmo/torch_datasets"

# Shared scorer instance — all 5 metrics hold a reference so _get_scorers()
# deduplicates to a single GPT-judge call per example (via Scorer.__call__).
_JUDGE = DenseCaptionJudgeScorer()


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DenseCaptionRecallMetric(Metric):
    """Mean recall (×100) over valid examples."""

    name: ClassVar[str] = "recall"
    scorer: ClassVar[DenseCaptionJudgeScorer] = _JUDGE  # type: ignore[assignment]

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            o.metadata["dense_caption_result"]["recall"]
            for r in responses
            for o in r.outputs
            if o.metadata
            and "dense_caption_result" in o.metadata
            and o.metadata["dense_caption_result"].get("recall_valid")
        ]
        return (sum(vals) / len(vals) * 100) if vals else 0.0


@dataclass(frozen=True, slots=True)
class DenseCaptionConsistencyMetric(Metric):
    """Mean consistency (×100) over valid examples."""

    name: ClassVar[str] = "consistency"
    scorer: ClassVar[DenseCaptionJudgeScorer] = _JUDGE  # type: ignore[assignment]

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            o.metadata["dense_caption_result"]["consistency"]
            for r in responses
            for o in r.outputs
            if o.metadata
            and "dense_caption_result" in o.metadata
            and o.metadata["dense_caption_result"].get("consistency_valid")
        ]
        return (sum(vals) / len(vals) * 100) if vals else 0.0


@dataclass(frozen=True, slots=True)
class DenseCaptionRecallAt10Metric(Metric):
    """Mean recall-at-10 (×100) over valid examples."""

    name: ClassVar[str] = "recall_at_10"
    scorer: ClassVar[DenseCaptionJudgeScorer] = _JUDGE  # type: ignore[assignment]

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            o.metadata["dense_caption_result"]["recall_at_10"]
            for r in responses
            for o in r.outputs
            if o.metadata
            and "dense_caption_result" in o.metadata
            and o.metadata["dense_caption_result"].get("recall_valid")
        ]
        return (sum(vals) / len(vals) * 100) if vals else 0.0


@dataclass(frozen=True, slots=True)
class DenseCaptionNumStatementsMetric(Metric):
    """Mean number of mturk statements per valid example (raw, not ×100)."""

    name: ClassVar[str] = "num_statements"
    scorer: ClassVar[DenseCaptionJudgeScorer] = _JUDGE  # type: ignore[assignment]

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            o.metadata["dense_caption_result"]["num_statements"]
            for r in responses
            for o in r.outputs
            if o.metadata
            and "dense_caption_result" in o.metadata
            and o.metadata["dense_caption_result"].get("recall_valid")
        ]
        return (sum(vals) / len(vals)) if vals else 0.0


@dataclass(frozen=True, slots=True)
class DenseCaptionAvgMetric(Metric):
    """Primary metric: (mean_recall + mean_consistency) / 2 × 100."""

    name: ClassVar[str] = "avg"
    scorer: ClassVar[DenseCaptionJudgeScorer] = _JUDGE  # type: ignore[assignment]

    def compute(self, responses: Sequence[Response]) -> float:
        results = [
            o.metadata["dense_caption_result"]
            for r in responses
            for o in r.outputs
            if o.metadata and "dense_caption_result" in o.metadata
        ]
        recall_vals = [r["recall"] for r in results if r.get("recall_valid")]
        cons_vals = [r["consistency"] for r in results if r.get("consistency_valid")]
        mean_recall = sum(recall_vals) / len(recall_vals) if recall_vals else 0.0
        mean_cons = sum(cons_vals) / len(cons_vals) if cons_vals else 0.0
        return (mean_recall + mean_cons) / 2.0 * 100


_DEFAULT_METRICS: tuple[Metric, ...] = (
    DenseCaptionRecallMetric(),
    DenseCaptionConsistencyMetric(),
    DenseCaptionRecallAt10Metric(),
    DenseCaptionNumStatementsMetric(),
    DenseCaptionAvgMetric(),
)
_AVG_METRIC = DenseCaptionAvgMetric()


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@register("dense_caption")
class DenseCaptionEval(Task):
    """Pixmo-cap dense-caption GPT-judge evaluation.

    Data comes from three on-disk sources:
    * ``final-data.json`` — whisper transcripts (consistency reference)
    * ``mturk-eval-statements/{sha256(url)}.json`` — canonical statements
      (recall reference)
    * ``torch_datasets/pixmo_datasets/dense-caption-eval/test.jsonl`` — image
      paths and URLs

    The model inference request is a CHAT message "Describe this image."
    The local image path is stored in ``instance.metadata["image_path"]`` for
    an inference script to load.
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=448)
    metrics = _DEFAULT_METRICS
    primary_metric = _AVG_METRIC

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._build_instances())
        yield from self._instances_cache

    def _build_instances(self) -> Iterator[Instance]:
        eval_dir = Path(os.environ.get("DENSE_CAPTION_EVAL_DIR", _DEFAULT_EVAL_DIR))
        data_home = Path(os.environ.get("MOLMO_DATA_DIR", _DEFAULT_DATA_HOME))
        test_jsonl = data_home / "pixmo_datasets" / "dense-caption-eval" / "test.jsonl"

        with open(eval_dir / "final-data.json") as f:
            final_data = json.load(f)
        url_to_transcripts: dict[str, list[dict]] = {
            ex["image"]: ex["transcripts"] for ex in final_data
        }

        limit = self.config.limit
        count = 0
        with open(test_jsonl) as f:
            for line in f:
                if limit is not None and count >= limit:
                    break
                rec = json.loads(line)
                url: str = rec["url"]
                image_id: str = rec.get("image_id", _sha256(url))
                image_name: str = rec.get("image", image_id)
                image_path = data_home / "pixmo_images" / image_name

                transcripts = url_to_transcripts.get(url)
                if transcripts is None:
                    logger.warning("No transcripts for %s — skipping", url)
                    continue

                mturk_file = eval_dir / "mturk-eval-statements" / f"{_sha256(url)}.json"
                if not mturk_file.exists():
                    logger.warning("No mturk file for %s — skipping", url)
                    continue
                with open(mturk_file) as f2:
                    mturk_data = json.load(f2)
                mturk_statements: str = mturk_data["canonical_statements"]

                yield Instance(
                    question="Describe this image.",
                    gold_answer=None,
                    metadata={
                        "id": image_id,
                        "url": url,
                        "image_path": str(image_path),
                        "transcripts": transcripts,
                        "mturk_statements": mturk_statements,
                    },
                )
                count += 1

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )


# "pixmo_cap" is an alias for "dense_caption" with no overrides.
register_variant("dense_caption", "pixmo_cap")

"""Pixmo-cap dense-caption evaluation task.

Loads the pixmo-cap test split (2730 images), joins judge reference data from
two on-disk sources, runs a GPT-4o recall+consistency judge, and aggregates
into recall / recall_at_10 / consistency / num_statements / avg metrics.

Environment variables (optional — defaults point to Weka paths):
    DENSE_CAPTION_EVAL_DIR   root of dense_caption_eval/ (contains final-data.json,
                             mturk-eval-statements/, gpt4-cache/)
    MOLMO_DATA_DIR           parent of torch_datasets/ (same convention as the 11 image-QA
                             tasks); images live at torch_datasets/pixmo_images/ and the split
                             manifest at torch_datasets/pixmo_datasets/dense-caption-eval/test.jsonl
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from olmo_eval.common.image_qa import dense_caption_question
from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.scorers.dense_caption_judge import DenseCaptionJudgeScorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response, SamplingParams
from olmo_eval.evals.tasks.common import Task, register, register_variant
from olmo_eval.evals.tasks.common.image_qa_base import load_instance_image, torch_datasets_dir

logger = logging.getLogger(__name__)

_DEFAULT_EVAL_DIR = "/weka/oe-training-default/mm-olmo/dense_caption_eval"

# Shared scorer instance — all 5 metrics hold a reference so _get_scorers()
# deduplicates to a single GPT-judge call per example (via Scorer.__call__).
_JUDGE = DenseCaptionJudgeScorer()


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DenseCaptionRecallMetric(Metric):
    """Mean recall (×100) over valid examples."""

    name: str = "recall"
    scorer: type[Scorer] | Scorer = _JUDGE

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

    name: str = "consistency"
    scorer: type[Scorer] | Scorer = _JUDGE

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

    name: str = "recall_at_10"
    scorer: type[Scorer] | Scorer = _JUDGE

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
    """Mean number of canonical statements per valid example (raw, not ×100).

    Matches mm_olmo's reported ``num_statements``: the count of canonical
    statements GPT derives from the *model caption* during the consistency
    judge (``ConsistencyEval.num_statements``), averaged over
    consistency-valid examples — a caption-informativeness proxy, not the
    recall-side mturk statement count.
    """

    name: str = "num_statements"
    scorer: type[Scorer] | Scorer = _JUDGE

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            o.metadata["dense_caption_result"]["consistency_num_statements"]
            for r in responses
            for o in r.outputs
            if o.metadata
            and "dense_caption_result" in o.metadata
            and o.metadata["dense_caption_result"].get("consistency_valid")
        ]
        return (sum(vals) / len(vals)) if vals else 0.0


@dataclass(frozen=True, slots=True)
class DenseCaptionAvgMetric(Metric):
    """Primary metric: (mean_recall + mean_consistency) / 2 × 100."""

    name: str = "avg"
    scorer: type[Scorer] | Scorer = _JUDGE

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

    The model inference request is a CHAT message whose text reproduces mm_olmo's
    per-example seeded ``long_caption`` template (the released Molmo2-4B uses
    ``demo_or_style_v2`` + ``uber_model_v2``: no style prefix, template sampled by line
    index). The local image path is stored in ``instance.metadata["image_path"]``.
    """

    sampling_params = SamplingParams(temperature=0.0, max_tokens=448)
    metrics = _DEFAULT_METRICS
    primary_metric = _AVG_METRIC
    #: Image decoding builds the instances; the GPT judge needs an OpenAI client.
    #: A missing scorer dependency scores every instance zero rather than failing.
    dependencies = ["pillow", "openai"]
    # None => decide from the checkpoint's prompt family (see `_question`). Set to a fixed
    # string on a subclass to force one prompt regardless of family.
    caption_prompt: str | None = None
    #: Prompt family assumed when the run does not say. Matches the instruction-tuned
    #: checkpoints, mirroring `ModelPromptPointingTask`'s defaults.
    default_system_prompt_style = "demo_or_style_v2"
    #: Length number put in the tag under the `style_and_length*` families, mm_olmo's
    #: `DataFormatter.default_inference_len`. `None` sends the bare tag, which is what
    #: OLMo-core's stage 1 trains on (its caption tag carries no number); mm_olmo captioner
    #: checkpoints, and OLMo-core stage-1 checkpoints trained before that change, trained
    #: with a number and want `dense_caption_captioner_len65`.
    default_inference_len: int | None = None

    def _question(self, idx: int) -> str:
        """The caption prompt for example ``idx``, following the checkpoint's family.

        The prompt has to follow the checkpoint rather than the task, exactly as for the
        `_mp` pointing tasks: a captioner-family checkpoint (mm_olmo `train_captioner.py`,
        `system_prompt="style_and_length_v2"`, which includes OLMo-core's
        `Molmo2-Stage1.py` runs) was trained with the caption's style tag as its whole user
        turn. Handing such a checkpoint a natural-language instruction instead measures it
        out of distribution: on a 4B stage-1 run it cost 7.1 recall points (38.18 -> 45.28)
        and made 3.4% of answers come back as `<points .../>` markup rather than prose.

        So `-o system_prompt_style=style_and_length_v2` selects the tag prompt here, the same
        override that selects it for the pointing tasks. The tag is `long_caption:`, or
        `long_caption <default_inference_len>:` when a subclass sets a number.
        """
        if self.caption_prompt is not None:
            return self.caption_prompt
        style = self.config.system_prompt_style or self.default_system_prompt_style
        if style in ("style_and_length", "style_and_length_v2"):
            if self.default_inference_len is None:
                return "long_caption:"
            return f"long_caption {self.default_inference_len}:"
        return dense_caption_question(idx)

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = list(self._build_instances())
        yield from self._instances_cache

    def _build_instances(self) -> Iterator[Instance]:
        eval_dir = Path(os.environ.get("DENSE_CAPTION_EVAL_DIR", _DEFAULT_EVAL_DIR))
        # Same MOLMO_DATA_DIR convention as the 11 image-QA tasks (the mm-olmo parent;
        # `torch_datasets_dir()` appends `torch_datasets`), so one env var covers all tasks.
        data_home = torch_datasets_dir()
        test_jsonl = data_home / "pixmo_datasets" / "dense-caption-eval" / "test.jsonl"

        with open(eval_dir / "final-data.json") as f:
            final_data = json.load(f)
        url_to_transcripts: dict[str, list[dict]] = {
            ex["image"]: ex["transcripts"] for ex in final_data
        }

        limit = self.config.limit
        count = 0
        with open(test_jsonl) as f:
            # `idx` is the raw line position in test.jsonl (matches mm_olmo's
            # DeterministicDataset index over the full readlines()); it drives the seeded
            # prompt and must NOT be affected by the transcript/mturk skips below.
            for idx, line in enumerate(f):
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

                question = self._question(idx)
                yield Instance(
                    question=question,
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
        image = load_instance_image(instance)
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            images=(image,) if image is not None else None,
        )


# "pixmo_cap" is an alias for "dense_caption" with no overrides.
register_variant("dense_caption", "pixmo_cap")


@register("dense_caption_captioner")
class DenseCaptionCaptionerEval(DenseCaptionEval):
    """Dense-caption eval prompt for stage-1 (captioner-family) checkpoints.

    Checkpoints trained with ``prompt_templates="none"`` + ``system_prompt=
    "style_and_length_v2"`` see a constant eval prompt, the caption's style tag with
    nothing after it: ``"long_caption:"``. That is exactly what OLMo-core's
    ``Molmo2-Stage1.py`` trains on, so train and test share one prompt. Use
    ``dense_caption`` for released-Molmo2-family (``uber_model_v2``) checkpoints,
    which sample a seeded natural-language instruction instead, and
    ``dense_caption_captioner_len65`` for checkpoints that trained with a length
    number in the tag.
    """

    caption_prompt: str | None = "long_caption:"


@register("dense_caption_captioner_len65")
class DenseCaptionCaptionerLen65Eval(DenseCaptionEval):
    """``dense_caption_captioner`` with mm_olmo's length number in the tag.

    mm_olmo's captioner family puts the caption's length in the tag during training
    (character count // 15 with noise) and a fixed ``default_inference_len`` of 65 at
    inference, giving ``"long_caption 65:"``. Verified verbatim against an mm_olmo
    ``DenseCaptionEval-test`` prediction dump. For mm_olmo captioner checkpoints (e.g. the
    siglip2-cap-stage1 runs) and for OLMo-core stage-1 checkpoints trained before the
    number was dropped from the tag; a checkpoint trained on the bare tag saw this form
    never, so it should use ``dense_caption_captioner``.
    """

    caption_prompt: str | None = "long_caption 65:"

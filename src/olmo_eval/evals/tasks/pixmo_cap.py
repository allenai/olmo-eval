"""PixMo-Cap caption evaluation tasks.

Two task classes are provided:

- :class:`PixMoCapTask` — token-F1 / ROUGE-L / BLEU-4 scored locally, no
  external API required.  Primary metric: ``token_f1``.

- :class:`PixMoCapGPTJudgedTask` — GPT-judged recall + consistency → cap-F1,
  matching the Molmo paper's reported numbers.  Requires ``OPENAI_API_KEY``.
  Primary metric: ``cap_f1``.

Usage in an OLMo-core training script::

    from olmo_eval.evals.tasks.pixmo_cap import PixMoCapTask
    from olmo_core.train.callbacks.multimodal_generation_callback import (
        MultimodalGenerationEvaluatorCallback,
    )
    cb = MultimodalGenerationEvaluatorCallback(
        tasks=[PixMoCapTask()],
        preprocessor=preprocessor,
        generator=generator,
        tokenizer=hf_tokenizer,
    )
    trainer.with_callback("pixmo_cap_gen", cb)
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

from olmo_eval.common.types import Instance, Split
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.cap_f1_judge import CapF1Judge, DEFAULT_JUDGE_MODEL
from olmo_eval.evals.tasks.common.caption_metrics import compute_caption_metrics, tokenize
from olmo_eval.evals.tasks.common.multimodal_base import MultimodalGenerationTask
from olmo_eval.evals.tasks.common.registry import register


# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------


@dataclass
class PixMoCapTaskConfig(TaskConfig):
    """Config shared by both PixMo-Cap task variants."""

    hf_dataset: str = "allenai/pixmo-cap"
    split: Split = Split.TEST
    limit: int | None = None
    prompt: str = "Describe this image in detail."


# ---------------------------------------------------------------------------
# Lexical-metric variant
# ---------------------------------------------------------------------------


@register("pixmo_cap")
class PixMoCapTask(MultimodalGenerationTask):
    """PixMo-Cap with token-F1 / ROUGE-L / BLEU-4 (no API key needed).

    Each instance is one image paired with its human reference caption(s).
    Scoring is done locally with :func:`~olmo_eval.evals.tasks.common.caption_metrics.compute_caption_metrics`.

    Primary metric: ``token_f1`` (macro-averaged, order-insensitive).
    """

    def __init__(self, config: PixMoCapTaskConfig | None = None) -> None:
        super().__init__(config or PixMoCapTaskConfig(name="pixmo_cap"))

    # ------------------------------------------------------------------
    # Instance loading
    # ------------------------------------------------------------------

    @property
    def instances(self) -> Iterator[Instance]:  # type: ignore[override]
        from datasets import load_dataset  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        cfg: PixMoCapTaskConfig = self.config  # type: ignore[assignment]
        ds = load_dataset(cfg.hf_dataset, split=str(cfg.split))
        if cfg.limit is not None:
            ds = ds.select(range(min(cfg.limit, len(ds))))

        for row in ds:
            image = row["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            # Store all available captions in metadata for multi-ref BLEU/ROUGE.
            captions = row.get("captions") or [row["caption"]]
            yield Instance(
                question=cfg.prompt,
                gold_answer=captions[0],
                images=(image,),
                metadata={"all_captions": captions},
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, prediction: str, instance: Instance) -> dict[str, float]:
        """Per-instance token-F1 and ROUGE-L against the primary reference."""
        from olmo_eval.evals.tasks.common.caption_metrics import rouge_l, token_f1

        ref = instance.gold_answer or ""
        pred_toks = tokenize(prediction)
        ref_toks = tokenize(ref)
        return {
            "token_f1": token_f1(pred_toks, ref_toks),
            "rouge_l": rouge_l(pred_toks, ref_toks),
        }

    def score_all(
        self,
        predictions: list[str],
        instances: list[Instance],
    ) -> dict[str, float]:
        """Corpus-level aggregation including BLEU-4."""
        refs = [inst.metadata.get("all_captions") or [inst.gold_answer or ""] for inst in instances]
        report = compute_caption_metrics(predictions, refs)
        return {
            "token_f1": report.token_f1,
            "rouge_l": report.rouge_l,
            "bleu_4": report.bleu_4,
        }


# ---------------------------------------------------------------------------
# GPT-judged variant
# ---------------------------------------------------------------------------


@dataclass
class PixMoCapGPTJudgedTaskConfig(PixMoCapTaskConfig):
    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_cache_dir: Optional[str] = None
    judge_threads: int = 16


@register("pixmo_cap_gpt")
class PixMoCapGPTJudgedTask(PixMoCapTask):
    """PixMo-Cap with GPT-judged cap-F1 (recall × consistency harmonic mean).

    Matches the Molmo paper's evaluation protocol exactly.  Requires
    ``OPENAI_API_KEY`` in the environment.  Uses a disk cache keyed by prompt
    content so re-runs are free.

    Primary metric: ``cap_f1``.
    """

    def __init__(self, config: PixMoCapGPTJudgedTaskConfig | None = None) -> None:
        super().__init__(config or PixMoCapGPTJudgedTaskConfig(name="pixmo_cap_gpt"))
        self._judge: Optional[CapF1Judge] = None

    def _get_judge(self) -> CapF1Judge:
        if self._judge is None:
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY must be set to use PixMoCapGPTJudgedTask"
                )
            cfg: PixMoCapGPTJudgedTaskConfig = self.config  # type: ignore[assignment]
            self._judge = CapF1Judge(
                api_key=api_key,
                model=cfg.judge_model,
                cache_dir=cfg.judge_cache_dir,
            )
        return self._judge

    def score(self, prediction: str, instance: Instance) -> dict[str, float]:
        raise NotImplementedError(
            "PixMoCapGPTJudgedTask scores at the corpus level via score_all(); "
            "per-instance score() is not supported."
        )

    def score_all(
        self,
        predictions: list[str],
        instances: list[Instance],
    ) -> dict[str, float]:
        cfg: PixMoCapGPTJudgedTaskConfig = self.config  # type: ignore[assignment]
        judge = self._get_judge()

        # Extract gold statements for each example (recall side).
        gold_captions = [inst.gold_answer or "" for inst in instances]
        from concurrent.futures import ThreadPoolExecutor, as_completed

        gold_stmts: dict[int, list[str]] = {}
        with ThreadPoolExecutor(max_workers=cfg.judge_threads) as pool:
            futs = {pool.submit(judge.extract_statements, cap): i for i, cap in enumerate(gold_captions)}
            for fut in as_completed(futs):
                gold_stmts[futs[fut]] = fut.result()

        items = [
            {
                "image_id": str(i),
                "prediction": predictions[i],
                "gold_statements": gold_stmts[i],
                "gold_caption": gold_captions[i],
            }
            for i in range(len(predictions))
        ]
        scores = judge.score_batch(items, n_threads=cfg.judge_threads)

        recall = sum(s.recall for s in scores) / len(scores)
        consistency = sum(s.consistency for s in scores) / len(scores)
        denom = recall + consistency
        f1 = 2 * recall * consistency / denom if denom > 0 else 0.0
        return {"cap_f1": f1, "recall": recall, "consistency": consistency}

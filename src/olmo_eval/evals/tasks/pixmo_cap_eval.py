"""PixMo-Cap dense-caption eval (Weka-local, GPT-judged).

Evaluates model captions on the Molmo v1 dense-caption-eval set (2,730 images)
using the GPT-judged recall + consistency cap-F1 protocol described in the
Molmo paper.  Uses the Weka-local copy of the data with pre-extracted mturk
atomic statements — see :mod:`olmo_eval.evals.tasks.pixmo_cap` for the
HuggingFace-based variant.

Data layout (under ``$MOLMO_DATA_DIR`` or an explicit ``data_root``)::

    torch_datasets/pixmo_datasets/dense-caption-eval/
        test.jsonl                        # {image_id, image (hash), url}
        dense_caption_eval_images/{hash}  # PNG files (no extension)
    dense_caption_eval/mturk-eval-statements/
        {hash}.json   # {image, whisper_descriptions, canonical_statements}

The gold ``canonical_statements`` (pre-extracted atomic statements) are used
directly for recall scoring; ``whisper_descriptions`` are joined into the
``gold_caption`` for consistency scoring.

Usage::

    from olmo_eval.evals.tasks.pixmo_cap_eval import PixmoCapEvalTask
    task = PixmoCapEvalTask()          # uses $MOLMO_DATA_DIR
    task = PixmoCapEvalTask(data_root="/path/to/molmo_data")

    predictions = [...]            # one decoded string per instance
    instances   = list(task.instances)
    scores = task.score_all(predictions, instances)
    # → {"cap_f1": 0.72, "recall": 0.78, "consistency": 0.66}
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from os.path import join
from typing import Optional

from olmo_eval.common.types import Instance, Split
from olmo_eval.evals.tasks.common.base import TaskConfig
from olmo_eval.evals.tasks.common.cap_f1_judge import CapF1Judge, DEFAULT_JUDGE_MODEL
from olmo_eval.evals.tasks.common.multimodal_base import MultimodalGenerationTask
from olmo_eval.evals.tasks.common.registry import register

_NUM_PREFIX_RE = re.compile(r"^\s*\d+[\.\)]\s*")


def _parse_statements(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = _NUM_PREFIX_RE.sub("", line.strip())
        if line:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PixmoCapEvalTaskConfig(TaskConfig):
    """Config for :class:`PixmoCapEvalTask`."""

    data_root: Optional[str] = None
    """Root data directory. Defaults to ``$MOLMO_DATA_DIR`` or
    ``/weka/oe-training-default/mm-olmo``."""

    split: Split = Split.TEST
    limit: Optional[int] = None
    skip_missing: bool = True
    prompt: str = "Describe this image."

    judge_model: str = DEFAULT_JUDGE_MODEL
    judge_cache_dir: Optional[str] = None
    judge_threads: int = 16

    def resolve_root(self) -> str:
        if self.data_root is not None:
            return self.data_root
        return os.environ.get("MOLMO_DATA_DIR", "/weka/oe-training-default/mm-olmo")


# ---------------------------------------------------------------------------
# Base task (data loading shared by both variants)
# ---------------------------------------------------------------------------


@register("pixmo_cap_eval")
class PixmoCapEvalTask(MultimodalGenerationTask):
    """PixMo-Cap dense-caption eval with GPT-judged cap-F1 (Weka-local data).

    Each instance carries:

    - ``images``: a single PIL image
    - ``question``: the generation prompt
    - ``gold_answer``: joined whisper transcripts (for consistency scoring)
    - ``metadata["atomic_statements"]``: pre-extracted gold statements (for
      recall scoring)
    - ``metadata["image_id"]``: image hash for identification

    Primary metric: ``cap_f1``.
    """

    def __init__(self, config: PixmoCapEvalTaskConfig | None = None, *, data_root: str | None = None) -> None:
        cfg = config or PixmoCapEvalTaskConfig(name="pixmo_cap_eval")
        if data_root is not None:
            cfg = PixmoCapEvalTaskConfig(
                **{**cfg.__dict__, "data_root": data_root}  # type: ignore[arg-type]
            )
        super().__init__(cfg)
        self._judge: Optional[CapF1Judge] = None

    # ------------------------------------------------------------------
    # Instance loading
    # ------------------------------------------------------------------

    @property
    def instances(self) -> Iterator[Instance]:  # type: ignore[override]
        from PIL import Image  # type: ignore[import]

        cfg: PixmoCapEvalTaskConfig = self.config  # type: ignore[assignment]
        root = cfg.resolve_root()
        test_jsonl = join(root, "torch_datasets", "pixmo_datasets", "dense-caption-eval", "test.jsonl")
        image_dir = join(root, "torch_datasets", "pixmo_datasets", "dense-caption-eval", "dense_caption_eval_images")
        mturk_dir = join(root, "dense_caption_eval", "mturk-eval-statements")

        # Build URL → mturk-file index once.
        url_to_mturk: dict[str, str] = {}
        if os.path.isdir(mturk_dir):
            for fname in os.listdir(mturk_dir):
                if not fname.endswith(".json"):
                    continue
                path = join(mturk_dir, fname)
                try:
                    with open(path) as f:
                        rec = json.load(f)
                    url = rec.get("image")
                    if isinstance(url, str):
                        url_to_mturk[url] = path
                except (OSError, json.JSONDecodeError):
                    continue

        n = 0
        with open(test_jsonl) as f:
            for raw in f:
                if cfg.limit is not None and n >= cfg.limit:
                    return
                row = json.loads(raw)
                image_hash = row["image"]
                image_path = join(image_dir, image_hash)

                if not os.path.exists(image_path):
                    if cfg.skip_missing:
                        continue
                    raise FileNotFoundError(image_path)

                mturk_file = url_to_mturk.get(row["url"])
                if mturk_file is None or not os.path.exists(mturk_file):
                    if cfg.skip_missing:
                        continue
                    raise FileNotFoundError(f"no mturk entry for {row['url']!r}")

                try:
                    with open(mturk_file) as mf:
                        mt = json.load(mf)
                    statements = _parse_statements(mt.get("canonical_statements", ""))
                    whisper = [s for s in (mt.get("whisper_descriptions") or []) if isinstance(s, str)]
                    gold_caption = "\n\n".join(whisper)
                    image = Image.open(image_path).convert("RGB")
                except Exception:  # noqa: BLE001
                    if cfg.skip_missing:
                        continue
                    raise

                n += 1
                yield Instance(
                    question=cfg.prompt,
                    gold_answer=gold_caption,
                    images=(image,),
                    metadata={
                        "image_id": image_hash,
                        "atomic_statements": statements,
                    },
                )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, prediction: str, instance: Instance) -> dict[str, float]:
        raise NotImplementedError(
            "PixmoCapEvalTask scores at corpus level via score_all(); "
            "per-instance score() is not supported without an API key."
        )

    def score_all(
        self,
        predictions: list[str],
        instances: list[Instance],
    ) -> dict[str, float]:
        """Run GPT judge and return aggregated cap-F1, recall, consistency."""
        cfg: PixmoCapEvalTaskConfig = self.config  # type: ignore[assignment]

        if self._judge is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY must be set to score PixmoCapEvalTask")
            self._judge = CapF1Judge(
                api_key=api_key,
                model=cfg.judge_model,
                cache_dir=cfg.judge_cache_dir,
            )

        items = [
            {
                "image_id": inst.metadata.get("image_id", str(i)),
                "prediction": predictions[i],
                "gold_statements": inst.metadata.get("atomic_statements", []),
                "gold_caption": inst.gold_answer or "",
            }
            for i, inst in enumerate(instances)
        ]
        scores = self._judge.score_batch(items, n_threads=cfg.judge_threads)

        # Match mm_olmo: mean of per-example F1 (not harmonic mean of means).
        recall = sum(s.recall for s in scores) / len(scores)
        consistency = sum(s.consistency for s in scores) / len(scores)
        f1 = sum(s.f1 for s in scores) / len(scores)
        return {"cap_f1": f1, "recall": recall, "consistency": consistency}

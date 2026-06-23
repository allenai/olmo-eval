"""Scorer for the image-pointing benchmarks (PixMo-Points-Eval v3, SACoGold).

Mirrors mm_olmo's ``SegmentionPointingEval`` / ``SACoGoldPointEvaluator`` per example: parse the
predicted points from the model's text, decode the ground-truth instance masks, and compute
point-in-mask precision/recall/f1 (see :mod:`olmo_eval.common.pointing`). The result dict is
stored on ``output.metadata["pointing_result"]`` for the :class:`PointingMetric` family to
aggregate (frequency buckets + weighted primaries); the scorer returns ``f1``.

Required ``instance.metadata``:

==========================  ====================================================
``pointing_annotators``     list of annotators; each a list of COCO-RLE segmentations
``image_size``              ``(width, height)`` of the image, or ``None`` (then read
                            from ``image_path``) — used to scale predicted points
``image_path``              filesystem path (only opened when ``image_size`` is None)
``weight``                  optional per-example weight (default 1.0; SACoGold)
==========================  ====================================================
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from olmo_eval.common.pointing import (
    decode_segmentation,
    extract_image_points,
    score_pointing_example,
)
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput


def _response_text(output: LMOutput) -> str:
    answer = output.extracted_answer
    if isinstance(answer, str) and answer:
        return answer
    return output.text or ""


@dataclass(frozen=True, slots=True)
class PointingScorer(Scorer):
    """Point-in-mask precision/recall/f1 for one pointing example.

    Stores ``{precision, recall, f1, count, weight}`` in
    ``output.metadata["pointing_result"]`` and returns ``f1``.
    """

    name: str = "pointing"

    def score(self, instance: Instance, output: LMOutput) -> float:
        meta = instance.metadata
        image_size = meta.get("image_size")
        if image_size is not None:
            image_w, image_h = int(image_size[0]), int(image_size[1])
        else:
            from PIL import Image

            with Image.open(meta["image_path"]) as im:
                image_w, image_h = im.size

        # Decode the ground-truth masks for each annotator.
        annotator_masks: list[list[np.ndarray]] = []
        for rles in meta["pointing_annotators"]:
            masks: list[np.ndarray] = []
            for seg in rles:
                m = decode_segmentation(seg, image_h, image_w)
                if m is not None:
                    masks.append(np.asarray(m, dtype=bool))
            annotator_masks.append(masks)

        # Predicted points in image-pixel coordinates.
        points = np.asarray(
            extract_image_points(_response_text(output), image_w, image_h), dtype=np.float64
        ).reshape(-1, 2)

        # Masks rarely live at a higher resolution than the image; rescale the predicted points
        # into mask coordinates when so (no-op when mask resolution == image resolution).
        mask_res = next((m[0].shape for m in annotator_masks if m), None)
        if mask_res is not None and len(points) > 0:
            mask_h, mask_w = int(mask_res[0]), int(mask_res[1])
            if (mask_h, mask_w) != (image_h, image_w):
                points = points * np.array([mask_w / image_w, mask_h / image_h])[None, :]

        precision, recall, f1 = score_pointing_example(points, annotator_masks)
        count = max((len(m) for m in annotator_masks), default=0)

        if output.metadata is None:
            output.metadata = {}
        output.metadata["pointing_result"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": count,
            "weight": float(meta.get("weight", 1.0)),
        }
        return f1

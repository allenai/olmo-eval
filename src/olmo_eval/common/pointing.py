"""Pure-python primitives for the image-pointing benchmarks.

Vendored from the mm_olmo reference (no mm_olmo imports) so it stays
unit-testable without the model:

* :func:`decode_segmentation` mirrors ``molmo_prediction_evaluators._decode_segmentation``
  (COCO RLE / polygon → binary mask), tolerating the JSON-serialized RLE form used by the
  SACoGold subset (``counts`` as ``str``, ``size`` as the string ``"[h, w]"``).
* :func:`score_pointing_example` mirrors ``SegmentationPointingScorer._score_example``:
  predicted points are matched to ground-truth instance masks by maximum bipartite matching,
  yielding ``(precision, recall, f1)`` taken as the best over the available annotators.

Heavy dependencies (``pycocotools``, ``scipy``) are imported lazily inside the functions so the
module imports cleanly on CPU-only / minimal environments. The point-text parser
(:func:`extract_image_points`) is re-exported from the shared count-parsing module.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np

from olmo_eval.common.image_qa.count_parsing import extract_image_points

__all__ = ["decode_segmentation", "extract_image_points", "score_pointing_example"]


def decode_segmentation(seg: Any, height: int, width: int) -> np.ndarray | None:
    """Decode a COCO segmentation (RLE dict or polygon list) to a binary ``HxW`` array.

    Faithful to mm_olmo's ``_decode_segmentation`` but first normalizes the JSON-serialized
    RLE form (string ``counts`` / string ``size``) emitted by the SACoGold subset.
    ``height``/``width`` are only used for polygon / uncompressed-RLE inputs; a compressed RLE
    carries its own ``size`` and decodes at that resolution.
    """
    from pycocotools import mask as mask_utils

    if isinstance(seg, dict):
        rle = dict(seg)
        size = rle.get("size")
        if isinstance(size, str):
            rle["size"] = [int(v) for v in ast.literal_eval(size)]
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle["counts"] = counts.encode("utf-8")
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        m = mask_utils.decode(rle)
    elif isinstance(seg, list):
        rle = mask_utils.frPyObjects(seg, height, width)
        m = mask_utils.decode(rle)
        if m.ndim == 3:
            m = m.max(axis=2)
    else:
        return None
    return m  # uint8 0/1


def score_pointing_example(
    points: Any,
    annotators: list[list[np.ndarray]],
) -> tuple[float, float, float]:
    """Point-in-mask ``(precision, recall, f1)`` for one example, best over annotators.

    ``points`` is an ``Nx2`` (or ``Nx3`` with a leading id) array of predicted points in the
    mask's pixel coordinates. ``annotators`` is a list of annotator mask-lists (one inner list
    per annotator; each entry an ``HxW`` binary mask). For each annotator, predicted points are
    matched to masks by maximum bipartite matching (a point matches a mask iff it lies inside
    it); precision = matches / #points, recall = matches / #masks. The maximum over annotators is
    returned. Empty-mask / empty-point edge cases mirror mm_olmo exactly.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        points = points.reshape(0, 2)
    if len(points) > 0:
        points = points[:, -2:]

    max_pr = max_rec = max_f1 = 0.0
    for masks in annotators:
        if len(masks) == 0:
            pr, rec = (1.0, 1.0) if len(points) == 0 else (0.0, 0.0)
        elif len(points) == 0:
            pr, rec = 0.0, 0.0
        else:
            stacked = np.stack(masks, axis=-1)  # H x W x M
            mask_h, mask_w = stacked.shape[:2]
            rounded = np.minimum(np.round(points), np.array([mask_w, mask_h]) - 1).astype(np.int64)
            distances = np.zeros((len(points), len(masks)), dtype=bool)
            for p_idx, (x, y) in enumerate(rounded):
                distances[p_idx] = stacked[y, x].astype(bool)
            col_ind = maximum_bipartite_matching(csr_matrix(distances))
            num_matches = int((col_ind != -1).sum())
            pr = num_matches / len(points)
            rec = num_matches / stacked.shape[-1]
        f1 = 2 * pr * rec / (pr + rec) if (pr > 0 and rec > 0) else 0.0
        max_pr, max_rec, max_f1 = max(max_pr, pr), max(max_rec, rec), max(max_f1, f1)
    return max_pr, max_rec, max_f1

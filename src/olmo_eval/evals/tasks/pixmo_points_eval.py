"""PixMo-Points-Eval v3 — image pointing benchmark (test split).

Mirrors mm_olmo's ``PixMoPointsEval3Config`` (``pixmo_point_eval_v3.1``) and its
``SegmentionPointingEval``: loads the prepared arrow dataset at
``torch_datasets/pixmo_datasets/pixmo_points_eval_v3`` and asks ``"Point to <label>."`` per
example. Each example carries a single annotator's instance masks (``segmentation_rles``); scoring
is point-in-mask precision/recall/f1, with simple-mean primaries and zero/single/low/med/high-
frequency buckets keyed on the GT instance count.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

from olmo_eval.common.scorers.pointing import PointingScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.pointing_base import (
    PointingTask,
    pointing_metrics,
    rebase_data_path,
    torch_datasets_dir,
)

_SCORER = PointingScorer()
_METRICS = pointing_metrics(
    _SCORER,
    buckets=("zero", "single", "low_freq", "med_freq", "high_freq"),
    weighted_primary=False,
)


def _format_query(label: str) -> str:
    """mm_olmo ``PixMoPointsEval3Config._format_query``: lowercase, drop a leading a/the."""
    text = label.lower()
    parts = text.split()
    if parts and parts[0] in ("a", "the"):
        text = " ".join(parts[1:])
    return f"Point to {text}."


@register("pixmo_points_eval")
class PixmoPointsEvalTask(PointingTask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
    metrics = _METRICS
    primary_metric = _METRICS[2]  # f1
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(
            str(torch_datasets_dir() / "pixmo_datasets" / "pixmo_points_eval_v3")
        )
        for idx in range(len(ds)):
            ex = ds[idx]
            label = str(ex["label"])
            rles = ex["segmentation_rles"]
            image_size = None
            if rles:
                size = rles[0]["size"]
                if isinstance(size, str):
                    size = ast.literal_eval(size)
                image_size = (int(size[1]), int(size[0]))  # (width, height)
            yield Instance(
                question=_format_query(label),
                gold_answer=None,
                metadata={
                    "pointing_annotators": [rles],  # single annotator
                    "image_size": image_size,
                    "image_path": rebase_data_path(str(ex["image"])),
                    "example_id": ex["example_id"],
                    "label": label,
                },
            )

"""PixMo Count (validation by default; ``pixmo_count:test`` for the test split).

Mirrors mm_olmo's ``PixMoCountConfig(counting=True)`` (task name
``pixmo_count_counting``): loads the prepared arrow dataset at
``torch_datasets/pixmo_datasets/count`` and asks an RNG-templated counting
question per example (``point_count`` style — no style tag).

The question template is selected per example by the seeded RNG of mm_olmo's
eval data pipeline, which depends on the example's **arrow-order index** —
instances are therefore built strictly in arrow order (verified to reproduce
all 540 released validation prompts exactly).

Reference (Molmo2-4B ck2000, val): correct=0.9093.
"""

from __future__ import annotations

from collections.abc import Iterator

from olmo_eval.common.image_qa import pixmo_count_question
from olmo_eval.common.scorers.image_qa import PointCountScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register, register_variant
from olmo_eval.evals.tasks.common.image_qa_base import (
    ImageQATask,
    point_count_metrics,
    rebase_data_path,
    torch_datasets_dir,
)

_SCORER = PointCountScorer()
_METRICS = point_count_metrics(_SCORER)


@register("pixmo_count")
class PixmoCountTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=192)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # correct
    split = Split.VALIDATION

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(str(torch_datasets_dir() / "pixmo_datasets" / "count"))
        ds = ds[self.config.split.value]
        # Arrow order is load-bearing: the per-example question template is
        # picked by an RNG seeded with the arrow index.
        for idx in range(len(ds)):
            ex = ds[idx]
            yield Instance(
                question=pixmo_count_question(ex["label"], idx),
                gold_answer=str(ex["count"]),
                metadata={
                    "count": ex["count"],
                    "label": ex["label"],
                    "arrow_idx": idx,
                    "example_id": ex["image_url"],
                    "image_url": ex["image_url"],
                    "image_path": rebase_data_path(ex["image"]),
                },
            )


register_variant("pixmo_count", "test", split=Split.TEST)

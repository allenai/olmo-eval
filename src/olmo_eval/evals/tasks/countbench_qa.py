"""CountBench QA (490 examples, counts 2–10).

Mirrors mm_olmo's ``CountBenchQaConfig``: loads the prepared arrow dataset at
``torch_datasets/academic_datasets/countbench_qa`` (CountBench images/counts
merged with the PaliGemma paired questions).  The dataset has a single test
set; the natural-language counting question is used verbatim (``point_count``
style — no style tag).

Reference (Molmo2-4B ck2000): correct=0.9408.
"""

from __future__ import annotations

from collections.abc import Iterator

from olmo_eval.common.scorers.image_qa import PointCountScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.image_qa_base import (
    ImageQATask,
    lazy_hf_image,
    point_count_metrics,
    torch_datasets_dir,
)

_SCORER = PointCountScorer()
_METRICS = point_count_metrics(_SCORER)


@register("countbench_qa")
class CountBenchQaTask(ImageQATask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=192)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # correct
    split = Split.TEST  # single prepared set

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(
            str(torch_datasets_dir() / "academic_datasets" / "countbench_qa")
        )
        ds_nodecode = ds.cast_column("image", datasets.Image(decode=False))
        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            yield Instance(
                question=ex["question"],
                gold_answer=str(ex["count"]),
                metadata={
                    "count": ex["count"],
                    "example_id": ex["example_id"],
                    "image_url": ex["image_url"],
                    "image": lazy_hf_image(ds_nodecode, idx, "image"),
                },
            )

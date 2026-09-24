"""CountBench QA (490 examples, counts 2–10).

Mirrors mm_olmo's ``CountBenchQaConfig``: loads the prepared arrow dataset at
``torch_datasets/academic_datasets/countbench_qa`` (CountBench images/counts
merged with the PaliGemma paired questions).  The dataset has a single test
set and uses the ``point_count`` style.

The question follows the checkpoint's prompt family, as for ``pixmo_count``. The
default (``uber_model_v2``, instruction-tuned checkpoints) sends the dataset's
question verbatim, as the released Molmo2-4B eval did. ``-o prompt_templates=none``
(stage-1 checkpoints, with ``-o system_prompt_style=style_and_length_v2`` for the
tag) sends the object name instead, ``point_count: headsets``: the form stage 1
trains on. The benchmark ships no label, but 487 of its 490 questions read "How
many <objects> are there in the image?", so :func:`countbench_label` takes the
name from there. The other three keep their question.

Reference (Molmo2-4B ck2000): correct=0.9408.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from olmo_eval.common.pointing_prompts import POINT_COUNT_STYLE
from olmo_eval.common.scorers.image_qa import PointCountScorer
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.image_qa_base import (
    ImageQATask,
    lazy_hf_image,
    point_count_metrics,
    torch_datasets_dir,
)
from olmo_eval.evals.tasks.common.pointing_base import StylePrefixMixin

#: The question form 487 of CountBenchQA's 490 questions take; the group is the object name.
_COUNT_QUESTION = re.compile(r"^How many (.+) are there in the image\?$")


def countbench_label(question: str) -> str | None:
    """The lower-cased object name in a "How many <objects> are there in the image?"
    question (``"headsets"``), or ``None`` for a question of another form. Lower-cased
    like the stage-1 counting labels, which the formatter lower-cases."""
    m = _COUNT_QUESTION.match(question)
    return m.group(1).lower() if m else None


_SCORER = PointCountScorer()
_METRICS = point_count_metrics(_SCORER)


@register("countbench_qa")
class CountBenchQaTask(StylePrefixMixin, ImageQATask):
    style = POINT_COUNT_STYLE
    sampling_params = SamplingParams(temperature=0.0, max_tokens=192)
    metrics = _METRICS
    primary_metric = _METRICS[0]  # correct
    split = Split.TEST  # single prepared set
    #: Prompt family assumed when the run does not say; matches the instruction-tuned
    #: checkpoints, like :class:`PixmoCountTask`.
    default_prompt_templates = "uber_model_v2"

    def _question(self, question: str) -> str:
        """The question text for this example, following the prompt family.

        ``"none"`` gives the object name (:func:`countbench_label`), or the question itself
        when it has no name to take; the templated families send the question verbatim.
        """
        family = self.config.prompt_templates or self.default_prompt_templates
        if family == "none":
            label = countbench_label(question)
            return question if label is None else label
        if family in ("uber_model", "uber_model_v2"):
            return question
        raise ValueError(
            f"Unsupported prompt_templates {family!r} for countbench_qa; "
            "expected 'none', 'uber_model' or 'uber_model_v2'"
        )

    def _build_instances(self) -> Iterator[Instance]:
        import datasets

        ds = datasets.load_from_disk(
            str(torch_datasets_dir() / "academic_datasets" / "countbench_qa")
        )
        ds_nodecode = ds.cast_column("image", datasets.Image(decode=False))
        for idx in range(len(ds_nodecode)):
            ex = ds_nodecode[idx]
            yield Instance(
                question=self.apply_family_prefix(self._question(str(ex["question"]))),
                gold_answer=str(ex["count"]),
                metadata={
                    "count": ex["count"],
                    "example_id": ex["example_id"],
                    "image_url": ex["image_url"],
                    "image": lazy_hf_image(ds_nodecode, idx, "image"),
                },
            )

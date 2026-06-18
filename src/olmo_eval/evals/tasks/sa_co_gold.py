"""SACoGold — image pointing benchmark (test split).

Mirrors mm_olmo's ``SACoGoldSubset`` (``sa-co-gold-subset-v3``) + ``SACoGoldPointEvaluator``:
loads the prepared subset at ``torch_datasets/sa-co-gold/molmo-subset-v1.json`` (4994 examples) and
asks ``"Point to (the) <text_input>."`` per example. Each example has **three** annotators' instance
masks (``annotations.{a,b,c}``) and a per-example ``weight``; scoring is point-in-mask
precision/recall/f1 taken as the best over annotators, with **weighted** primaries (matching the
balanced subset) and all/single/low/med/high-frequency simple-mean buckets.
"""

from __future__ import annotations

import json
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
    buckets=("all", "single", "low_freq", "med_freq", "high_freq"),
    weighted_primary=True,
)


def _format_query(text_input: str) -> str:
    """mm_olmo ``SACoGold._format_query``: prepend "the" unless the phrase already starts a/the."""
    parts = text_input.split()
    if parts and parts[0].lower() in ("a", "the"):
        return f"Point to {text_input}."
    return f"Point to the {text_input}."


@register("sa_co_gold_subset")
class SaCoGoldSubsetTask(PointingTask):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1024)
    metrics = _METRICS
    primary_metric = _METRICS[2]  # weighted f1
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        path = torch_datasets_dir() / "sa-co-gold" / "molmo-subset-v1.json"
        with open(path) as f:
            data = json.load(f)
        for ex in data:
            md = ex["metadata"]
            ann = md["annotations"]
            annotators = [[a["segmentation"] for a in ann[k]] for k in ("a", "b", "c")]
            yield Instance(
                question=_format_query(ex["text_input"]),
                gold_answer=None,
                metadata={
                    "pointing_annotators": annotators,
                    "image_size": (int(md["width"]), int(md["height"])),
                    "image_path": rebase_data_path(ex["image"]),
                    "weight": float(md["weight"]),
                    "example_id": md["id"],
                    "subset": md["subset"],
                    "label": ex["text_input"],
                },
            )

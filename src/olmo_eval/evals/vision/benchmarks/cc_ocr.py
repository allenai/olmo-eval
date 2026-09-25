"""CC-OCR multi-scene OCR (https://arxiv.org/abs/2412.02210).

The multi-scene track of CC-OCR: 2,750 images of scene text, documents and web/UGC images in
English and Chinese, across 13 sub-datasets. The model transcribes all the text in each image;
the reference is compared as a multiset of words (characters for the Chinese sub-datasets), so
reading order does not matter.

Each row of the dataset carries its own prompt, sent verbatim in a single user turn with the
image. Scoring is the official evaluator's (:mod:`olmo_eval.evals.vision.scoring.cc_ocr`): the
primary ``macro_f1`` is the unweighted mean over
sub-datasets of each one's mean per-image F1, the track score the paper reports; ``micro_f1``
and every sub-dataset's score are reported alongside. Metrics are 0-1 (the paper reports x100).

The benchmark prescribes no decoding settings; this task decodes greedily. Data is fetched from
the Hub at a pinned revision; set ``CC_OCR_DIR`` to a local copy of the dataset repository to
read it from disk instead.
"""

from __future__ import annotations

import base64
import csv
import functools
import io
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.scoring.cc_ocr import CcOcrDatasetMetric, CcOcrScorer, CcOcrTrackMetric
from olmo_eval.evals.vision.tasks.ocr import OcrTask

HF_REPO = "wulipc/CC-OCR"
HF_REVISION = "c64517e92179991d509776064174776700cdd5a2"
TRACK = "multi_scene_ocr"

#: The track's sub-datasets (the ``split`` column), in the official index order.
SUBSETS: tuple[str, ...] = (
    "TotalText",
    "IC15",
    "InverseText",
    "Hieragent",
    "zh_scene",
    "FUNSD",
    "CORD",
    "IAM",
    "zh_doc",
    "zh_handwriting",
    "ugc_laion",
    "zh_vertical",
    "zh_dense",
)

_SCORER = CcOcrScorer()
_MACRO_F1 = CcOcrTrackMetric(name="macro_f1", scorer=_SCORER, kind="macro_f1")
_METRICS: tuple[Metric, ...] = (
    _MACRO_F1,
    CcOcrTrackMetric(name="micro_f1", scorer=_SCORER, kind="micro_f1"),
    *(CcOcrDatasetMetric(name=dataset, scorer=_SCORER, dataset=dataset) for dataset in SUBSETS),
)


@functools.lru_cache(maxsize=64)
def _load_tsv(path: str) -> list[dict[str, str]]:
    """Rows of one sub-dataset file; cached because its images are decoded row by row."""
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _decode_tsv_image(path: str, row: int):
    """Decode one base64 image cell (module-level so the owning instance stays picklable)."""
    from PIL import Image

    data = base64.b64decode(_load_tsv(path)[row]["image"])
    return Image.open(io.BytesIO(data)).convert("RGB")


def _data_dir() -> Path:
    local = os.environ.get("CC_OCR_DIR")
    if local:
        return Path(local)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            allow_patterns=[f"{TRACK}/**/*.tsv"],
        )
    )


@register("cc_ocr_multi_scene")
class CcOcrMultiSceneTask(OcrTask):
    dependencies = ["pillow", "huggingface-hub"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    metrics = _METRICS
    primary_metric = _MACRO_F1
    split = Split.TEST

    def _build_instances(self) -> Iterator[Instance]:
        instances = [
            instance
            for path in sorted((_data_dir() / TRACK).rglob("*.tsv"))
            for instance in self._tsv_instances(str(path))
        ]
        found = {instance.metadata["dataset"] for instance in instances}
        if found != set(SUBSETS):
            raise RuntimeError(
                f"CC-OCR {TRACK} has sub-datasets {sorted(found)}, expected {sorted(SUBSETS)}"
            )
        yield from instances

    def _tsv_instances(self, tsv_path: str) -> Iterator[Instance]:
        for row_index, row in enumerate(_load_tsv(tsv_path)):
            if row["category"] != TRACK:
                raise RuntimeError(f"{tsv_path}: row of track {row['category']!r} under {TRACK!r}")
            # Image names repeat across sub-datasets; only the pair is unique.
            key = f"{row['split']}/{row['image_name']}"
            yield Instance(
                question=self._question(row["question"]),
                gold_answer=row["answer"],
                metadata={
                    "id": key,
                    "example_id": key,
                    "dataset": row["split"],
                    "image_name": row["image_name"],
                    "answer": row["answer"],
                    "image": functools.partial(_decode_tsv_image, tsv_path, row_index),
                },
            )

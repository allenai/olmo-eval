"""CC-OCR — a four-track OCR benchmark for multimodal models (https://arxiv.org/abs/2412.02210).

7,058 images in 39 sub-datasets, grouped into four tracks:

* ``multi_scene_ocr`` (2,750) — scene, document and web/UGC text in English and Chinese;
* ``multi_lan_ocr`` (1,500) — ten languages, 150 images each;
* ``doc_parsing`` (800) — pages to LaTeX, tables to HTML, handwritten formulas to LaTeX and
  molecules to SMILES;
* ``kie`` (2,008) — key information extraction into a given JSON schema.

Each row of the dataset carries its own prompt, sent verbatim in a single user turn with
the image. Scoring is the official evaluator's
(:mod:`olmo_eval.common.image_qa.cc_ocr` / :mod:`olmo_eval.common.scorers.cc_ocr`): a
sub-dataset score, a track score that is the unweighted mean of its sub-datasets, and the
primary ``overall``, the unweighted mean of the four tracks. All metrics are 0-1 (the paper
reports x100).

``cc_ocr`` runs everything; ``cc_ocr_multi_scene_ocr``, ``cc_ocr_multi_lan_ocr``,
``cc_ocr_doc_parsing`` and ``cc_ocr_kie`` run one track each, with that track's score as the
primary metric.

The benchmark prescribes no decoding settings; this task decodes greedily with room for a
full page of LaTeX. Data is fetched from the Hub at a pinned revision; set ``CC_OCR_DIR`` to
a local copy of the dataset repository to read it from disk instead.
"""

from __future__ import annotations

import base64
import csv
import functools
import io
import itertools
import os
import sys
from collections.abc import Iterator
from pathlib import Path

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.cc_ocr import (
    OCR_TRACKS,
    TRACKS,
    CcOcrDatasetMetric,
    CcOcrJsonParseRateMetric,
    CcOcrOverallMetric,
    CcOcrScorer,
    CcOcrTrackMetric,
)
from olmo_eval.common.types import Instance, SamplingParams, Split
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.tasks.common.ocr_base import OcrTask

HF_REPO = "wulipc/CC-OCR"
HF_REVISION = "c64517e92179991d509776064174776700cdd5a2"

#: Track -> its sub-datasets (the ``split`` column), in the official index order.
SUBSETS: dict[str, tuple[str, ...]] = {
    "multi_scene_ocr": (
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
    ),
    "multi_lan_ocr": (
        "Arabic",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
        "Vietnamese",
    ),
    "doc_parsing": (
        "doc_photo_chn",
        "doc_photo_eng",
        "doc_scan_chn",
        "doc_scan_eng",
        "table_photo_chn",
        "table_photo_eng",
        "table_scan_chn",
        "table_scan_eng",
        "molecular_handwriting",
        "formula_handwriting",
    ),
    "kie": ("sroie2019_word", "CORD", "EPHOIE_SCUT", "POIE", "COLD_SIBR", "COLD_CELL"),
}

_SCORER = CcOcrScorer()
_OVERALL = CcOcrOverallMetric(name="overall", scorer=_SCORER)
_TRACK_METRICS = {
    track: CcOcrTrackMetric(name=track, scorer=_SCORER, track=track) for track in TRACKS
}


def _track_detail_metrics(track: str) -> tuple[Metric, ...]:
    """A track's secondary statistic(s) and its per-sub-dataset scores."""
    secondary: tuple[Metric, ...]
    if track in OCR_TRACKS:
        secondary = (
            CcOcrTrackMetric(
                name=f"{track}_micro_f1", scorer=_SCORER, track=track, kind="micro_f1"
            ),
        )
    elif track == "kie":
        secondary = (
            CcOcrTrackMetric(name="kie_acc", scorer=_SCORER, track=track, kind="acc"),
            CcOcrJsonParseRateMetric(name="kie_json_parse_rate", scorer=_SCORER),
        )
    else:
        secondary = ()
    return (
        *secondary,
        *(
            CcOcrDatasetMetric(
                name=f"{track}/{dataset}", scorer=_SCORER, track=track, dataset=dataset
            )
            for dataset in SUBSETS[track]
        ),
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
            allow_patterns=[f"{track}/**/*.tsv" for track in TRACKS],
        )
    )


@register("cc_ocr")
class CcOcrTask(OcrTask):
    dependencies = ["pillow", "huggingface-hub", "rapidfuzz", "apted", "lxml", "zss"]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
    metrics = (
        _OVERALL,
        *_TRACK_METRICS.values(),
        *(metric for track in TRACKS for metric in _track_detail_metrics(track)),
    )
    primary_metric = _OVERALL
    split = Split.TEST
    #: Tracks this task runs.
    tracks: tuple[str, ...] = TRACKS

    def _build_instances(self) -> Iterator[Instance]:
        data_dir = _data_dir()
        per_dataset: list[list[Instance]] = []
        for track in self.tracks:
            seen: set[str] = set()
            for tsv in sorted((data_dir / track).rglob("*.tsv")):
                instances = list(self._tsv_instances(str(tsv), track))
                seen.update(i.metadata["dataset"] for i in instances)
                per_dataset.append(instances)
            if seen != set(SUBSETS[track]):
                raise RuntimeError(
                    f"CC-OCR track {track!r} has sub-datasets {sorted(seen)}, "
                    f"expected {sorted(SUBSETS[track])}"
                )

        # Round-robin across sub-datasets so a small ``limit`` samples every one of them.
        for group in itertools.zip_longest(*per_dataset):
            for instance in group:
                if instance is not None:
                    yield instance

    def _tsv_instances(self, tsv_path: str, track: str) -> Iterator[Instance]:
        for row_index, row in enumerate(_load_tsv(tsv_path)):
            if row["category"] != track:
                raise RuntimeError(f"{tsv_path}: row of track {row['category']!r} under {track!r}")
            yield Instance(
                question=self._question(row["question"]),
                gold_answer=row["answer"],
                metadata={
                    # Image names repeat across sub-datasets; only the triple is unique.
                    "id": f"{track}/{row['split']}/{row['image_name']}",
                    "example_id": f"{track}/{row['split']}/{row['image_name']}",
                    "track": track,
                    "dataset": row["split"],
                    "op": row["l2-category"],
                    "image_name": row["image_name"],
                    "answer": row["answer"],
                    "image": functools.partial(_decode_tsv_image, tsv_path, row_index),
                },
            )


def _register_track_task(track: str) -> None:
    primary = _TRACK_METRICS[track]

    @register(f"cc_ocr_{track}")
    class _CcOcrTrackTask(CcOcrTask):
        metrics = (primary, *_track_detail_metrics(track))
        primary_metric = primary
        tracks = (track,)

    _CcOcrTrackTask.__name__ = f"CcOcr{track.title().replace('_', '')}Task"
    _CcOcrTrackTask.__qualname__ = _CcOcrTrackTask.__name__
    _CcOcrTrackTask.__doc__ = f"The ``{track}`` track of CC-OCR."


for _track in TRACKS:
    _register_track_task(_track)

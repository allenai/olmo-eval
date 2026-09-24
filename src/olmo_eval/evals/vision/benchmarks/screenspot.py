"""ScreenSpot GUI grounding benchmarks.

ScreenSpot (Cheng et al., 2024, https://arxiv.org/abs/2401.10935) has 1272
screenshots of mobile (iOS, Android), desktop (Windows, macOS) and web (shop,
forum, developer tool, GitLab) interfaces, each paired with an instruction that
names one UI element and that element's bounding box. ScreenSpot-v2 (Wu et al.,
2024, https://arxiv.org/abs/2410.23218) corrects mislabeled instructions and
boxes in the same benchmark. The model is asked to point at the element; the
answer is correct when the point falls inside the box.

Tasks:
    screenspot     ScreenSpot, from ``rootsautomation/ScreenSpot``.
    screenspot_v2  ScreenSpot-v2, from ``likaixin/ScreenSpot-v2-variants`` using the
                   original instructions (the repository's rephrased variants are
                   not used).

Each task reports overall accuracy plus accuracy on text and icon targets and on
each platform (mobile, desktop, web), all from the same predictions.

The prompt follows the checkpoint's pointing prompt family, as the ``_mp``
pointing tasks do: the instruction is the label handed to
:func:`build_pointing_prompt`. The defaults match the instruction-tuned Molmo2
checkpoints; ``-o prompt_templates=none -o system_prompt_style=style_and_length_v2``
selects the terse ``pointing: <instruction>`` form pretrain checkpoints were
trained on. The answer is read from Molmo2 or Molmo pointing markup (or a JSON
click action); see :mod:`olmo_eval.evals.vision.scoring.grounding`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics import AccuracyMetric, Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMOutput, Response, SamplingParams
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import register
from olmo_eval.evals.vision.data.images import lazy_hf_image
from olmo_eval.evals.vision.scoring.grounding import BBOX_KEY, PointInBoxScorer, parse_point
from olmo_eval.evals.vision.scoring.prompts import build_pointing_prompt
from olmo_eval.evals.vision.tasks.base import VisionTask

logger = logging.getLogger(__name__)

SCREENSPOT_REPO = "rootsautomation/ScreenSpot"
#: Dataset revisions, pinned so the instances behind a stored result cannot change.
SCREENSPOT_REVISION = "0be08781e2e188582f6131625ae1598d443b4d5d"
SCREENSPOT_V2_REPO = "likaixin/ScreenSpot-v2-variants"
SCREENSPOT_V2_REVISION = "aa4d83b6833d77d3bb3b7cb769951e2db80052b6"

PLATFORMS = ("mobile", "desktop", "web")
DATA_TYPES = ("text", "icon")

#: ScreenSpot names the application a screenshot came from; the paper groups them
#: into the three platforms its results are reported on.
SOURCE_TO_PLATFORM = {
    "ios": "mobile",
    "android": "mobile",
    "windows": "desktop",
    "macos": "desktop",
    "shop": "web",
    "forum": "web",
    "tool": "web",
    "gitlab": "web",
}

#: Metadata keys the per-group metrics read.
DATA_TYPE_KEY = "data_type"
PLATFORM_KEY = "platform"


@dataclass(frozen=True, slots=True)
class GroupAccuracyMetric(Metric):
    """Mean score over the instances whose metadata ``group_key`` equals ``group``.

    Lets one run of the benchmark report its per-type and per-platform numbers
    alongside the overall accuracy, from the same predictions.
    """

    name: str = "accuracy"
    scorer: type[Scorer] | Scorer = PointInBoxScorer
    group_key: str = DATA_TYPE_KEY
    group: str = "text"

    def _in_group(self, response: Response) -> bool:
        return response.instance.metadata.get(self.group_key) == self.group

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        scores = [r.scores.get(scorer_name, 0.0) for r in responses if self._in_group(r)]
        if not scores:
            logger.warning(
                "%s: no instances with %s=%r, reporting 0.0. This happens when a limited "
                "run samples none from the group; the value is not a score.",
                self.name,
                self.group_key,
                self.group,
            )
            return 0.0
        return sum(scores) / len(scores)

    def compute_instance(self, response: Response) -> float | None:
        if not self._in_group(response):
            return None
        score = response.scores.get(self.scorer().name)
        return float(score) if score is not None else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        """Serialize including the group, so differently scoped metrics hash differently."""
        # Named base call, not super(): a slots dataclass is a rebuilt class, so the
        # zero-argument form resolves against a stale __class__ cell before Python 3.13.
        return {**Metric.to_dict(self), "group_key": self.group_key, "group": self.group}


_OVERALL = AccuracyMetric(scorer=PointInBoxScorer)
_METRICS: tuple[Metric, ...] = (
    _OVERALL,
    *(
        GroupAccuracyMetric(name=f"{group}_accuracy", group_key=DATA_TYPE_KEY, group=group)
        for group in DATA_TYPES
    ),
    *(
        GroupAccuracyMetric(name=f"{group}_accuracy", group_key=PLATFORM_KEY, group=group)
        for group in PLATFORMS
    ),
)


def _percent_bbox(bbox: Sequence[float], width: float = 1.0, height: float = 1.0) -> list[float]:
    """``[x1, y1, x2, y2]`` scaled from ``width`` x ``height`` units to percent, 1 decimal."""
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / width * 100, 1),
        round(y1 / height * 100, 1),
        round(x2 / width * 100, 1),
        round(y2 / height * 100, 1),
    ]


class _ScreenSpotTask(VisionTask):
    """Shared prompt, answer extraction and metrics of the ScreenSpot tasks."""

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    metrics = _METRICS
    primary_metric = _OVERALL
    #: Prompt family assumed when the run does not say; matches the instruction-tuned
    #: Molmo2 checkpoints, as for the ``_mp`` pointing tasks.
    default_prompt_templates = "uber_model_v2"
    default_system_prompt_style = "demo_or_style_v2"

    def _question(self, instruction: str, index: int) -> str:
        return build_pointing_prompt(
            instruction,
            index,
            prompt_templates=self.config.prompt_templates or self.default_prompt_templates,
            system_prompt=self.config.system_prompt_style or self.default_system_prompt_style,
        )

    def _instance(
        self,
        *,
        index: int,
        image_id: str,
        instruction: str,
        bbox: list[float],
        data_type: str,
        data_source: str,
        platform: str,
        image: Any,
    ) -> Instance:
        return Instance(
            question=self._question(instruction, index),
            gold_answer=json.dumps(bbox),
            metadata={
                "id": image_id,
                "index": index,
                "instruction": instruction,
                BBOX_KEY: bbox,
                DATA_TYPE_KEY: data_type,
                "data_source": data_source,
                PLATFORM_KEY: platform,
                **image,
            },
        )

    def extract_answer(self, output: LMOutput) -> tuple[float, float] | None:
        return parse_point(output.text)


@register("screenspot")
class ScreenSpot(_ScreenSpotTask):
    """ScreenSpot v1.

    Each record carries the screenshot, the instruction, ``bbox`` as
    ``[x1, y1, x2, y2]`` normalized to 0-1, ``data_type`` (text / icon) and
    ``data_source`` (the application). Images decode lazily, one per request.
    """

    data_source = DataSource(path=SCREENSPOT_REPO, split="test", revision=SCREENSPOT_REVISION)

    def process_record(self, record: dict[str, Any], index: int, image: Any) -> Instance:
        source = record["data_source"]
        return self._instance(
            index=index,
            image_id=record["file_name"],
            instruction=record["instruction"],
            bbox=_percent_bbox(record["bbox"]),
            data_type=record["data_type"],
            data_source=source,
            platform=SOURCE_TO_PLATFORM[source],
            image={"image": image},
        )

    def _build_instances(self) -> Iterator[Instance]:
        from datasets import Image as HFImage
        from datasets import load_dataset

        source = self.config.get_data_source()
        dataset = load_dataset(source.path, split=source.split, revision=source.revision)
        dataset = dataset.cast_column("image", HFImage(decode=False))
        # Iterate without the image column so building instances moves no pixel bytes.
        for index, record in enumerate(dataset.remove_columns("image")):
            yield self.process_record(record, index, lazy_hf_image(dataset, index, "image"))


@register("screenspot_v2")
class ScreenSpotV2(_ScreenSpotTask):
    """ScreenSpot-v2.

    The repository holds one annotation file per platform under ``annotations/``
    and the screenshots under ``images/``. Each record carries ``bbox`` as
    ``[x1, y1, x2, y2]`` in pixels, ``img_size`` as ``[width, height]``,
    ``ui_type`` (text / icon), ``application`` and ``platform``. Platforms are
    read in :data:`PLATFORMS` order, so an instance's index (which seeds its
    prompt template) is stable.
    """

    data_source = DataSource(path=SCREENSPOT_V2_REPO, split="test", revision=SCREENSPOT_V2_REVISION)

    def process_record(self, record: dict[str, Any], index: int, image_path: str) -> Instance:
        width, height = record["img_size"]
        return self._instance(
            index=index,
            image_id=record["id"],
            instruction=record["instruction"],
            bbox=_percent_bbox(record["bbox"], width, height),
            data_type=record["ui_type"],
            data_source=record["application"],
            platform=record["platform"],
            image={"image_path": image_path},
        )

    def _build_instances(self) -> Iterator[Instance]:
        from huggingface_hub import snapshot_download

        source = self.config.get_data_source()
        root = Path(snapshot_download(source.path, repo_type="dataset", revision=source.revision))
        index = 0
        for platform in PLATFORMS:
            with open(root / "annotations" / f"{platform}.json") as f:
                records = json.load(f)
            for record in records:
                image_path = str(root / "images" / record["img_filename"])
                yield self.process_record(record, index, image_path)
                index += 1

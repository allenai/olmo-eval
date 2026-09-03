"""Shared base task and metrics for the Molmo2 image-QA benchmarks.

The 11 benchmark task modules (``chart_qa``, ``vqa2``, ``doc_qa``, ``info_qa``,
``text_vqa``, ``real_world_qa``, ``mmmu``, ``math_vista``, ``countbench_qa``,
``pixmo_count``, ``ai2d``) build on:

* :class:`ImageQATask` — caches instances, formats CHAT requests, resolves the
  mm_olmo data root.
* Generic metrics (:class:`MeanScorerMetric`, :class:`ChartQaSubsetMetric`,
  :class:`PointCountMetric`, :class:`PointCountPerCountMetric`,
  :class:`PointCountCategoryAverageMetric`, :class:`Ai2dMetric`).

Conventions:

* ``instance.question`` is the **fully formatted** prompt text — style prefix
  (e.g. ``"vqa2: "``) and multiple-choice option block already baked in, so
  inference scripts pass it through verbatim (prompt parity is the task's
  responsibility).
* The image is stored in ``instance.metadata["image_path"]`` (filesystem path)
  or ``instance.metadata["image"]`` (a PIL image or a zero-arg callable
  returning one — use :func:`load_instance_image` to resolve either form).
* Data is read from ``$MOLMO_DATA_DIR`` (default
  ``/weka/oe-training-default/mm-olmo``) and never written — loaders error
  out rather than build caches.
"""

from __future__ import annotations

import functools
import os
import re
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.base import Task

DEFAULT_MOLMO_DATA_DIR = "/weka/oe-training-default/mm-olmo"


def molmo_data_dir() -> Path:
    """Root of the mm-olmo data tree (read-only)."""
    return Path(os.environ.get("MOLMO_DATA_DIR", DEFAULT_MOLMO_DATA_DIR))


def torch_datasets_dir() -> Path:
    return molmo_data_dir() / "torch_datasets"


def rebase_data_path(path: str) -> str:
    """Rebase an absolute path recorded on another machine onto the current root.

    Cached manifests (e.g. ``vqa2/molmo_val.json``) store absolute image paths
    from the machine that built them; if the stored path does not exist locally
    but contains ``torch_datasets/``, re-anchor it under the current data root.
    """
    if os.path.exists(path):
        return path
    marker = "torch_datasets/"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        return str(torch_datasets_dir() / suffix)
    return path


def _decode_hf_image_cell(dataset: Any, index: int, column: str):
    """Decode one image cell of a no-decode HF dataset (module-level so it is picklable)."""
    import io

    from PIL import Image

    rec = dataset[index][column]
    if isinstance(rec, dict):
        if rec.get("bytes"):
            return Image.open(io.BytesIO(rec["bytes"]))
        if rec.get("path"):
            return Image.open(rec["path"])
    return rec


def lazy_hf_image(dataset, index: int, column: str = "image"):
    """A picklable zero-arg callable that decodes one image cell of a no-decode HF dataset.

    ``dataset`` should have ``column`` cast to ``datasets.Image(decode=False)`` so building
    instances never decodes pixels; the callable decodes exactly one image when called.

    Returns a ``functools.partial`` over a module-level function (not a local closure) so the
    owning ``Instance`` stays picklable across the runner's worker processes — a closure would
    raise ``AttributeError: Can't get local object 'lazy_hf_image.<locals>._load'`` on pickle.
    """
    return functools.partial(_decode_hf_image_cell, dataset, index, column)


def load_instance_image(instance: Instance):
    """Resolve an instance's image to a PIL image (or None if imageless)."""
    image = instance.metadata.get("image")
    if image is not None:
        return image() if callable(image) else image
    path = instance.metadata.get("image_path")
    if path is not None:
        from PIL import Image

        return Image.open(path)
    return None


PROMPT_STYLES = ("molmo", "neutral", "cot")

#: mm_olmo SFT style tags, prefixed to the question at instance-build time by the
#: individual tasks (``vqa2.py:92``, ``chart_qa.py:59``, ...). In-distribution for Molmo
#: checkpoints; meaningless to any other model.
_STYLE_TAG_RE = re.compile(r"^(?:vqa2|chart_qa|doc_qa|info_qa|text_vqa|ai2_diagram\w*): ")

#: The short-answer instruction published VLM numbers are measured under. Without it a
#: chat-tuned model answers "There are three bars shown in the chart." and exact-match
#: scoring against "3" gives zero -- which is what sank VQAv2/ChartQA/TextVQA here.
SHORT_ANSWER_INSTRUCTION = "\nAnswer the question using a single word or phrase."

#: MMMU embeds ``<image 1>``, ``<image 2>`` ... in the question to mark where each image
#: belongs. We attach images via the chat template instead, so under the neutral style the
#: markers are stray tokens that no model was trained to read. (``mmmu_pro`` already
#: resolves them via ``replace_images_tokens``; ``mmmu`` never did.)
_IMAGE_MARKER_RE = re.compile(r"\s*<image\s+\d+>\s*")

#: Chain-of-thought instructions. MMMU is a reasoning benchmark whose published numbers are
#: produced with CoT; "Only return the correct answer option." forbids it, which caps the
#: score at the model's direct-answer ability. Measured on Qwen3-VL-4B-Instruct: 22% of
#: outputs began reasoning anyway and truncated, and among those that did answer cleanly
#: accuracy was only 56.94% -- so eliminating truncation alone reaches ~57 against a
#: reported 67.4. Format follows MMMU-Pro's official prompt, which asks for a final
#: ``Answer:`` line precisely so a trace can be parsed.
COT_MC_INSTRUCTION = (
    "Answer the preceding multiple-choice question. Think step by step, then end your "
    "response with a line of exactly this form: 'Answer: $LETTER' (without quotes) where "
    "$LETTER is one of the option letters."
)
COT_OPEN_INSTRUCTION = (
    "Answer the preceding question. Think step by step, then end your response with a line "
    "of exactly this form: 'Answer: $ANSWER' (without quotes) where $ANSWER is your final "
    "answer, as briefly as possible."
)

#: Markers left over from a multiple-choice template; if the question already tells the
#: model how to answer, do not append a second, conflicting instruction.
_HAS_ANSWER_INSTRUCTION = (
    "Only return the correct answer option.",
    "Please answer directly with only the letter",
    "Answer with the option letter",
)


def render_question(config, instance: Instance) -> str:
    """Apply ``config.prompt_style`` to an instance's question.

    ``molmo`` returns it unchanged. ``neutral`` strips the mm_olmo style tag and, for
    short-answer tasks, appends an explicit brevity instruction. Multiple-choice questions
    already carry their own answer instruction and are left alone apart from the tag.
    """
    style = getattr(config, "prompt_style", "molmo") or "molmo"
    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt_style {style!r}; expected one of {PROMPT_STYLES}")
    question = instance.question
    if style == "molmo":
        return question

    question = _STYLE_TAG_RE.sub("", question)
    question = _IMAGE_MARKER_RE.sub(" ", question).strip()
    already_instructed = any(marker in question for marker in _HAS_ANSWER_INSTRUCTION)

    if style == "cot":
        # Strip any direct-answer instruction first: leaving it in would contradict the
        # request to reason.
        for marker in _HAS_ANSWER_INSTRUCTION:
            question = question.replace(marker, "")
        question = re.sub(r"\n{2,}", "\n", question).strip()
        instruction = COT_MC_INSTRUCTION if already_instructed else COT_OPEN_INSTRUCTION
        return f"{question}\n\n{instruction}"

    if not already_instructed:
        question = question + SHORT_ANSWER_INSTRUCTION
    return question


#: Matches the final ``Answer:`` line a CoT response is asked to end with. Deliberately the
#: LAST occurrence: a reasoning trace routinely writes "Answer:" mid-thought, and the shared
#: ``clean_prediction`` takes the FIRST (``pred.split("Answer:")[1]``), which would score the
#: model's discarded first guess.
_COT_ANSWER_RE = re.compile(r"answer\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_cot_answer(text: str) -> str:
    """Pull the final ``Answer: ...`` payload out of a chain-of-thought response.

    Falls back to the whole text when no marker is present, so a trace truncated before it
    could answer stays visible to the scorer rather than becoming an empty string.
    """
    if not text:
        return ""
    matches = _COT_ANSWER_RE.findall(text)
    if not matches:
        return text
    return matches[-1].strip().rstrip(".").strip()


IMAGE_MODES = ("real", "none", "caption")

#: Cache of loaded caption files, keyed by path. Instances are formatted one at a time, so
#: without this the JSONL would be re-read once per instance.
_CAPTION_CACHE: dict[str, dict[str, str]] = {}


def load_captions(path: str) -> dict[str, str]:
    """Load a ``{"example_id": ..., "caption": ...}`` JSONL into a dict, cached by path."""
    if path not in _CAPTION_CACHE:
        import json

        captions: dict[str, str] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                captions[str(record["example_id"])] = record["caption"]
        if not captions:
            raise ValueError(f"Caption source {path!r} is empty")
        _CAPTION_CACHE[path] = captions
    return _CAPTION_CACHE[path]


def resolve_image_mode(config, instance: Instance) -> tuple[bool, str | None]:
    """Resolve ``config.image_mode`` for one instance.

    :returns: ``(send_image, caption)``. ``caption`` is non-None only in caption mode, and
        should be prepended to the question.

    :raises ValueError: on an unknown mode, on caption mode without a ``caption_source``,
        or when a caption is missing for this instance. Missing captions must fail rather
        than fall back to text-only: the difference between "described the image" and "saw
        nothing" is the entire measurement.
    """
    mode = getattr(config, "image_mode", "real") or "real"
    if mode not in IMAGE_MODES:
        raise ValueError(f"Unknown image_mode {mode!r}; expected one of {', '.join(IMAGE_MODES)}")
    if mode == "real":
        return True, None
    if mode == "none":
        return False, None

    source = getattr(config, "caption_source", None)
    if not source:
        raise ValueError("image_mode='caption' requires caption_source to be set")
    example_id = str(instance.metadata.get("example_id", ""))
    captions = load_captions(source)
    if example_id not in captions:
        raise ValueError(
            f"No caption for example_id {example_id!r} in {source}. Regenerate the caption "
            "file for this task/split rather than running with partial coverage."
        )
    return False, captions[example_id]


def apply_caption(question: str, caption: str) -> str:
    """Prepend an oracle caption to a question, standing in for the image."""
    return f"Figure description:\n{caption}\n\n{question}"


class ImageQATask(Task):
    """Base class for the Molmo2 image-QA benchmark tasks."""

    #: Image decoding is needed to build instances, whichever provider runs them.
    dependencies = ["pillow"]

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            instances = list(self._build_instances())
            limit = self.config.limit
            if limit is not None:
                instances = instances[:limit]
            self._instances_cache = instances
        yield from self._instances_cache

    @abstractmethod
    def _build_instances(self) -> Iterator[Instance]:
        """Yield all instances for ``self.config.split`` (before ``limit``)."""
        ...

    def format_request(self, instance: Instance) -> LMRequest:
        send_image, caption = resolve_image_mode(self.config, instance)
        question = render_question(self.config, instance)
        question = apply_caption(question, caption) if caption else question
        image = load_instance_image(instance) if send_image else None
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": question},),
            images=(image,) if image is not None else None,
        )


# ---------------------------------------------------------------------------
# Generic metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MeanScorerMetric(Metric):
    """Mean of a scorer's per-response score (the mm_olmo ``global_mean``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        if not responses:
            return 0.0
        scorer_name = self.scorer().name
        return sum(r.scores.get(scorer_name, 0.0) for r in responses) / len(responses)


@dataclass(frozen=True)
class ChartQaSubsetMetric(Metric):
    """ChartQA metric over all / human / augmented examples.

    Subset membership comes from ``instance.metadata["is_human"]``, matching
    the ``_human`` / ``_aug`` breakdowns of mm_olmo's ``VqaEval``.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    subset: str = "all"  # all | human | aug

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = [r.scores.get(scorer_name, 0.0) for r in responses if self._in_subset(r)]
        return sum(vals) / len(vals) if vals else 0.0

    def _in_subset(self, response: Response) -> bool:
        if self.subset == "all":
            return True
        is_human = bool(response.instance.metadata.get("is_human"))
        return is_human if self.subset == "human" else not is_human


def _point_count_results(responses: Sequence[Response]) -> Iterator[tuple[Response, dict]]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "point_count_result" in output.metadata:
                yield response, output.metadata["point_count_result"]


@dataclass(frozen=True)
class PointCountMetric(Metric):
    """Mean of one field (``correct`` / ``close`` / ``valid``) of the count result."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: str = "correct"

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [result[self.kind] for _, result in _point_count_results(responses)]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass(frozen=True)
class PointCountPerCountMetric(Metric):
    """Counting accuracy restricted to examples with ground-truth count ``k``."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    k: int = 0

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [
            result["correct"]
            for response, result in _point_count_results(responses)
            if int(response.instance.metadata["count"]) == self.k
        ]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass(frozen=True)
class PointCountCategoryAverageMetric(Metric):
    """Macro average of per-count accuracies over the counts present."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]

    def compute(self, responses: Sequence[Response]) -> float:
        by_count: dict[int, list[float]] = {}
        for response, result in _point_count_results(responses):
            by_count.setdefault(int(response.instance.metadata["count"]), []).append(
                result["correct"]
            )
        if not by_count:
            return 0.0
        per_count = [sum(v) / len(v) for v in by_count.values()]
        return sum(per_count) / len(per_count)


@dataclass(frozen=True)
class Ai2dMetric(Metric):
    """AI2D accuracy split by box rendering.

    abc-label questions count toward exactly one variant (transparent or
    opaque, per ``has_transparent_box``); questions without abc labels count
    toward both — matching ``mc_ai2d_opaque`` / ``mc_ai2d_transparent`` in
    mm_olmo's ``VqaEval``.
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    transparent: bool = False

    def compute(self, responses: Sequence[Response]) -> float:
        vals: list[float] = []
        for response in responses:
            for output in response.outputs:
                if not output.metadata or "ai2d_result" not in output.metadata:
                    continue
                result = output.metadata["ai2d_result"]
                if result["abc_label"]:
                    if self.transparent and not result["has_transparent_box"]:
                        continue
                    if not self.transparent and result["has_transparent_box"]:
                        continue
                vals.append(result["is_correct"])
        return sum(vals) / len(vals) if vals else 0.0


# Counts present in the CountBench QA / PixMo Count eval sets.
POINT_COUNT_KS: tuple[int, ...] = tuple(range(2, 11))


def point_count_metrics(scorer: Scorer) -> tuple[Metric, ...]:
    """The full mm_olmo ``PointCountEval`` metric family for one shared scorer."""
    return (
        PointCountMetric(name="correct", scorer=scorer, kind="correct"),
        PointCountMetric(name="close", scorer=scorer, kind="close"),
        PointCountMetric(name="valid", scorer=scorer, kind="valid"),
        *(
            PointCountPerCountMetric(name=f"correct_{k}", scorer=scorer, k=k)
            for k in POINT_COUNT_KS
        ),
        PointCountCategoryAverageMetric(name="per_category_average", scorer=scorer),
    )

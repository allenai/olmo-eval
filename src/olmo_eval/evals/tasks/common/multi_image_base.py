"""Shared base task and metrics for the multi-image QA benchmarks.

The multi-image benchmarks (``muir_bench``, ``mmiu``, ``blink``) are a separate
family from the single-image QA tasks: each instance carries a *list* of images,
and scoring is the mm_olmo ``MuirBenchEval``-family protocol (response cleanup +
MMMU-style letter parsing; see
:class:`olmo_eval.common.scorers.multi_image.MultiImageMcScorer`).  This module
provides:

* :class:`MultiImageQATask` — caches instances and formats the CHAT request with
  the instance's image list, capped at ``max_images`` (mm_olmo's
  ``eval_molmo2.py`` sets ``mm_preprocessor.image.max_images = 20`` and the
  formatter truncates ``images[:max_images]``).
* :class:`MultiImageCategoryMetric` / :class:`MultiImageCountBucketMetric` —
  per-category and image-count-bucket accuracies reproducing the mm_olmo
  evaluator metric keys (``all``, per-task/sub-task/relationship, MMIU's
  ``num_images<=10`` / ``<=20`` / ``>20``).
* :func:`replace_images` — mm_olmo's ``<image>`` → ``"Image n"`` prompt rewrite
  (``academic_datasets.replace_images``).

It reuses only generic, task-agnostic helpers imported from the image-QA base;
nothing single-image-specific is shared.
"""

from __future__ import annotations

import functools
import re
from abc import abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.base import Task

# Generic (not single-image-QA-specific) data utilities.
from olmo_eval.evals.tasks.common.image_qa_base import torch_datasets_dir

__all__ = [
    "MultiImageCategoryMetric",
    "MultiImageCountBucketMetric",
    "MultiImageQATask",
    "lazy_hf_image_list",
    "load_instance_images",
    "multi_image_mc_metrics",
    "replace_images",
    "torch_datasets_dir",
]


def replace_images(question: str, options: list[str], max_images: int | None = None):
    """mm_olmo ``academic_datasets.replace_images``: rewrite ``<image>`` placeholders
    to ``"Image n"``, numbering sequentially through the question then the options."""
    all_strings = [question] + options
    image_counter = 1

    total_images = sum(s.count("<image>") for s in all_strings)
    if max_images is not None:
        total_images = min(total_images, max_images)

    replaced = []

    for s in all_strings:

        def repl(match):
            nonlocal image_counter
            if image_counter > total_images:
                return match.group(0)
            replacement = f"Image {image_counter}"
            image_counter += 1
            return replacement

        replaced.append(re.sub(r"<image>", repl, s))

    return replaced[0], replaced[1:]


def _decode_hf_image_list_cell(dataset: Any, index: int, column: str) -> list:
    """Decode one list-of-images cell of a no-decode HF dataset (module-level so it
    is picklable)."""
    import io

    from PIL import Image

    out = []
    for rec in dataset[index][column]:
        if isinstance(rec, dict):
            if rec.get("bytes"):
                rec = Image.open(io.BytesIO(rec["bytes"]))
            elif rec.get("path"):
                rec = Image.open(rec["path"])
        out.append(rec)
    return out


def lazy_hf_image_list(dataset, index: int, column: str):
    """A picklable zero-arg callable that decodes one list-of-images cell.

    ``dataset`` should have ``column`` cast to a sequence of
    ``datasets.Image(decode=False)`` so building instances never decodes pixels.
    """
    return functools.partial(_decode_hf_image_list_cell, dataset, index, column)


def load_instance_images(instance: Instance) -> list:
    """Resolve an instance's image list to PIL images.

    ``instance.metadata["images"]`` may be a zero-arg callable returning the whole
    list, or a sequence whose items are PIL images, zero-arg callables, or
    filesystem paths.
    """
    images = instance.metadata.get("images")
    if images is None:
        return []
    if callable(images):
        images = images()
    resolved = []
    for image in images:
        if callable(image):
            image = image()
        elif isinstance(image, str):
            from PIL import Image

            image = Image.open(image)
        resolved.append(image)
    return resolved


class MultiImageQATask(Task):
    """Base class for the multi-image QA benchmark tasks."""

    #: mm_olmo eval-time image cap (``eval_molmo2.py`` sets ``max_images = 20``;
    #: the data formatter truncates the image list to this length).
    max_images: int = 20

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
        images = load_instance_images(instance)[: self.max_images]
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            images=tuple(images) if images else None,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiImageCategoryMetric(Metric):
    """Mean scorer value over instances whose ``field`` metadata equals ``category``
    (``category=None`` means all instances — mm_olmo's ``all`` key)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    field: str = "task"
    category: str | None = None

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = [
            r.scores.get(scorer_name, 0.0)
            for r in responses
            if self.category is None or r.instance.metadata.get(self.field) == self.category
        ]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass(frozen=True)
class MultiImageCountBucketMetric(Metric):
    """Mean scorer value over instances bucketed by ``metadata["num_images"]``
    (mm_olmo ``MMIUEval``: ``<=10``, ``11-20``, ``>20``)."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    min_images: int = 0  # exclusive lower bound
    max_images: int | None = None  # inclusive upper bound, None = unbounded

    def compute(self, responses: Sequence[Response]) -> float:
        scorer_name = self.scorer().name
        vals = []
        for r in responses:
            num_images = int(r.instance.metadata["num_images"])
            if num_images <= self.min_images:
                continue
            if self.max_images is not None and num_images > self.max_images:
                continue
            vals.append(r.scores.get(scorer_name, 0.0))
        return sum(vals) / len(vals) if vals else 0.0


def multi_image_mc_metrics(
    scorer: Scorer,
    categories: Sequence[str],
    field: str,
) -> tuple[Metric, ...]:
    """``all`` plus one per-category accuracy, using mm_olmo's metric key strings.

    The first metric (``all``) is the task's primary metric.
    """
    return (
        MultiImageCategoryMetric(name="all", scorer=scorer, field=field, category=None),
        *(
            MultiImageCategoryMetric(name=category, scorer=scorer, field=field, category=category)
            for category in categories
        ),
    )

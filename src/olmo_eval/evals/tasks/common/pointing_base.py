"""Shared base task and metrics for the image-pointing benchmarks.

The pointing benchmarks (``pixmo_points_eval``, ``sa_co_gold_subset``) are a separate family from
the image-QA tasks: the model is asked ``"Point to <thing>."`` and scored by point-in-mask
precision/recall/f1 (see :class:`olmo_eval.common.scorers.pointing.PointingScorer`).  This module
provides:

* :class:`PointingTask` — caches instances and formats the CHAT+image request (its own base, not a
  subclass of the image-QA task).
* :class:`PointingMetric` + :func:`pointing_metrics` — frequency-bucketed and (optionally) weighted
  means over the per-example results, reproducing mm_olmo's ``SegmentionPointingEval`` /
  ``SACoGoldPointEvaluator`` metric key sets.

It reuses only generic, task-agnostic helpers (image loading + data-root resolution) imported from
the image-QA base; nothing image-QA-specific is shared.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Instance, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common.base import Task

# Generic (not image-QA-specific) data/image utilities.
from olmo_eval.evals.tasks.common.image_qa_base import (
    load_instance_image,
    rebase_data_path,
    torch_datasets_dir,
)

__all__ = [
    "PointingMetric",
    "PointingTask",
    "load_instance_image",
    "pointing_metrics",
    "rebase_data_path",
    "torch_datasets_dir",
]


class PointingTask(Task):
    """Base class for the image-pointing benchmark tasks."""

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
        image = load_instance_image(instance)
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
            images=(image,) if image is not None else None,
        )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

# Frequency buckets keyed on the ground-truth instance count, matching mm_olmo
# (low/med_freq overlap intentionally, as in ``SegmentionPointingEval``).
_BUCKETS: dict[str, Callable[[int], bool]] = {
    "all": lambda c: True,
    "zero": lambda c: c == 0,
    "single": lambda c: c == 1,
    "low_freq": lambda c: c <= 3,
    "med_freq": lambda c: 3 <= c <= 10,
    "high_freq": lambda c: c >= 10,
}

_KINDS: tuple[str, ...] = ("precision", "recall", "f1")


def _pointing_results(responses: Sequence[Response]) -> Iterator[dict]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "pointing_result" in output.metadata:
                yield output.metadata["pointing_result"]


@dataclass(frozen=True)
class PointingMetric(Metric):
    """Mean of one field (``precision`` / ``recall`` / ``f1``) over a count bucket.

    With ``weighted=True`` the mean is weighted by each example's ``weight`` (the SACoGold
    primary metrics); otherwise it is a simple mean (mm_olmo's ``global_mean``).
    """

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: str = "f1"
    bucket: str = "all"
    weighted: bool = False

    def compute(self, responses: Sequence[Response]) -> float:
        in_bucket = _BUCKETS[self.bucket]
        num = 0.0
        den = 0.0
        for result in _pointing_results(responses):
            if not in_bucket(int(result["count"])):
                continue
            weight = float(result["weight"]) if self.weighted else 1.0
            num += float(result[self.kind]) * weight
            den += weight
        return num / den if den else 0.0


def pointing_metrics(
    scorer: Scorer,
    *,
    buckets: Sequence[str],
    weighted_primary: bool = False,
) -> tuple[Metric, ...]:
    """Build the pointing metric family for one shared scorer.

    Returns the primary ``precision``/``recall``/``f1`` (over all examples, weighted iff
    ``weighted_primary``) followed by simple-mean ``{bucket}_{kind}`` metrics for each requested
    bucket — reproducing the exact key set of the corresponding mm_olmo evaluator.
    """
    metrics: list[Metric] = [
        PointingMetric(name=kind, scorer=scorer, kind=kind, bucket="all", weighted=weighted_primary)
        for kind in _KINDS
    ]
    for bucket in buckets:
        metrics.extend(
            PointingMetric(name=f"{bucket}_{kind}", scorer=scorer, kind=kind, bucket=bucket)
            for kind in _KINDS
        )
    return tuple(metrics)

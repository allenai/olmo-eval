"""Single-image QA task base and its metric families."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from olmo_eval.common.metrics.base import Metric
from olmo_eval.common.scorers.base import Scorer
from olmo_eval.common.types import Response
from olmo_eval.evals.vision.tasks.base import VisionTask


class ImageQATask(VisionTask):
    """Base class for the single-image QA benchmarks."""


def _point_count_results(responses: Sequence[Response]) -> Iterator[tuple[Response, dict]]:
    for response in responses:
        for output in response.outputs:
            if output.metadata and "point_count_result" in output.metadata:
                yield response, output.metadata["point_count_result"]


def _point_count_result(response: Response) -> dict | None:
    for output in response.outputs:
        if output.metadata and "point_count_result" in output.metadata:
            return output.metadata["point_count_result"]
    return None


@dataclass(frozen=True)
class PointCountMetric(Metric):
    """Mean of one field (``correct`` / ``close`` / ``valid``) of the count result."""

    name: str  # type: ignore[misc]
    scorer: Scorer  # type: ignore[misc]
    kind: str = "correct"

    def compute(self, responses: Sequence[Response]) -> float:
        vals = [result[self.kind] for _, result in _point_count_results(responses)]
        return sum(vals) / len(vals) if vals else 0.0

    def compute_instance(self, response: Response) -> float | None:
        result = _point_count_result(response)
        return float(result[self.kind]) if result is not None else None

    def supports_pairwise_scorer_fallback(self) -> bool:
        # The scorer channel is `correct`; close/valid are their own fields.
        return False


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

    def compute_instance(self, response: Response) -> float | None:
        result = _point_count_result(response)
        if result is None or int(response.instance.metadata["count"]) != self.k:
            return None
        return float(result["correct"])

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


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

    def compute_instance(self, response: Response) -> float | None:
        # A macro average over counts has no exact per-instance decomposition.
        return None

    def supports_pairwise_scorer_fallback(self) -> bool:
        return False


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

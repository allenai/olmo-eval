"""Unit tests for the image-pointing family (geometry, metrics, prompts).

Pure CPU — no GPU, no model, no dataset.  ``decode_segmentation`` needs pycocotools and is
skipped when it is not installed; everything else uses scipy (a declared dep) only.
"""

from __future__ import annotations

import numpy as np
import pytest

from olmo_eval.common.pointing import score_pointing_example
from olmo_eval.common.scorers.pointing import PointingScorer
from olmo_eval.common.types import Instance, LMOutput, Response
from olmo_eval.evals.tasks.common.pointing_base import PointingMetric
from olmo_eval.evals.tasks.pixmo_points_eval import _format_query as pixmo_query
from olmo_eval.evals.tasks.sa_co_gold import _format_query as saco_query


def _mask(region: tuple[slice, slice], shape: tuple[int, int] = (10, 10)) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[region] = True
    return m


# ---------------------------------------------------------------------------
# score_pointing_example
# ---------------------------------------------------------------------------


class TestScorePointingExample:
    # one 3x3 mask block at rows/cols 2..4 -> mask[y, x] True for x,y in [2,4]
    BLOCK = (slice(2, 5), slice(2, 5))

    def test_point_inside(self) -> None:
        pr, rec, f1 = score_pointing_example([[3, 3]], [[_mask(self.BLOCK)]])
        assert (pr, rec, f1) == (1.0, 1.0, 1.0)

    def test_point_outside(self) -> None:
        assert score_pointing_example([[8, 8]], [[_mask(self.BLOCK)]]) == (0.0, 0.0, 0.0)

    def test_two_points_one_mask(self) -> None:
        pr, rec, f1 = score_pointing_example([[3, 3], [8, 8]], [[_mask(self.BLOCK)]])
        assert pr == pytest.approx(0.5)
        assert rec == pytest.approx(1.0)
        assert f1 == pytest.approx(2 / 3)

    def test_two_points_two_masks(self) -> None:
        masks = [_mask((slice(2, 5), slice(2, 5))), _mask((slice(6, 9), slice(6, 9)))]
        pr, rec, f1 = score_pointing_example([[3, 3], [7, 7]], [masks])
        assert (pr, rec, f1) == (1.0, 1.0, 1.0)

    def test_empty_masks_empty_points(self) -> None:
        assert score_pointing_example([], [[]]) == (1.0, 1.0, 1.0)

    def test_empty_masks_with_points(self) -> None:
        assert score_pointing_example([[3, 3]], [[]]) == (0.0, 0.0, 0.0)

    def test_max_over_annotators(self) -> None:
        miss = [_mask((slice(6, 9), slice(6, 9)))]  # (3,3) not inside
        hit = [_mask(self.BLOCK)]  # (3,3) inside
        assert score_pointing_example([[3, 3]], [miss, hit]) == (1.0, 1.0, 1.0)

    def test_strips_leading_id_column(self) -> None:
        # Nx3 input (leading object id) -> only last two cols are coordinates.
        assert score_pointing_example([[0, 3, 3]], [[_mask(self.BLOCK)]]) == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# PointingMetric
# ---------------------------------------------------------------------------


def _pointing_response(count: int, f1: float, weight: float = 1.0) -> Response:
    instance = Instance(question="q", metadata={})
    result = {"precision": f1, "recall": f1, "f1": f1, "count": count, "weight": weight}
    output = LMOutput(text="", metadata={"pointing_result": result})
    return Response(instance=instance, request=None, outputs=[output])


class TestPointingMetric:
    SCORER = PointingScorer()

    def _responses(self) -> list[Response]:
        # (count, f1, weight)
        rows = [(1, 1.0, 1.0), (2, 0.0, 3.0), (11, 0.5, 1.0)]
        return [_pointing_response(c, f, w) for c, f, w in rows]

    def test_simple_mean_all(self) -> None:
        m = PointingMetric(name="f1", scorer=self.SCORER, kind="f1", bucket="all")
        assert m.compute(self._responses()) == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    def test_weighted_mean_all(self) -> None:
        m = PointingMetric(name="f1", scorer=self.SCORER, kind="f1", bucket="all", weighted=True)
        # (1*1 + 0*3 + 0.5*1) / (1 + 3 + 1)
        assert m.compute(self._responses()) == pytest.approx(1.5 / 5)

    def test_bucket_single(self) -> None:
        m = PointingMetric(name="single_f1", scorer=self.SCORER, kind="f1", bucket="single")
        assert m.compute(self._responses()) == pytest.approx(1.0)

    def test_bucket_high_freq(self) -> None:
        m = PointingMetric(name="high_freq_f1", scorer=self.SCORER, kind="f1", bucket="high_freq")
        assert m.compute(self._responses()) == pytest.approx(0.5)

    def test_empty(self) -> None:
        assert PointingMetric(name="f1", scorer=self.SCORER).compute([]) == 0.0


# ---------------------------------------------------------------------------
# prompt builders
# ---------------------------------------------------------------------------


class TestFormatQuery:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Earring", "Point to earring."),
            ("a cat", "Point to cat."),
            ("The Dogs", "Point to dogs."),
        ],
    )
    def test_pixmo(self, label: str, expected: str) -> None:
        assert pixmo_query(label) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("cotton t-shirt", "Point to the cotton t-shirt."),
            ("a cat", "Point to a cat."),
            ("the dog", "Point to the dog."),
        ],
    )
    def test_saco(self, text: str, expected: str) -> None:
        assert saco_query(text) == expected


# ---------------------------------------------------------------------------
# decode_segmentation (needs pycocotools)
# ---------------------------------------------------------------------------


class TestDecodeSegmentation:
    def test_roundtrip_and_str_normalization(self) -> None:
        mask_utils = pytest.importorskip("pycocotools.mask")
        from olmo_eval.common.pointing import decode_segmentation

        mask = np.zeros((4, 5), dtype=np.uint8)
        mask[1:3, 2:4] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))  # {counts: bytes, size: [4, 5]}

        decoded = decode_segmentation(rle, 4, 5)
        assert decoded is not None
        np.testing.assert_array_equal(decoded.astype(bool), mask.astype(bool))

        # SACoGold JSON form: counts as str, size as the string "[h, w]".
        str_rle = {"counts": rle["counts"].decode("ascii"), "size": str(list(rle["size"]))}
        decoded_str = decode_segmentation(str_rle, 4, 5)
        assert decoded_str is not None
        np.testing.assert_array_equal(decoded_str.astype(bool), mask.astype(bool))

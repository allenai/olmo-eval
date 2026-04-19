"""Tests for pairwise matrix helpers."""

from __future__ import annotations

import numpy as np
import pytest

from olmo_eval.analysis.pairwise import ModelMeta, PairStats, PairwiseResult
from olmo_eval.analysis.pairwise_plot import build_se_matrix, build_win_rate_matrix


def _build_result() -> tuple[PairwiseResult, PairStats, PairStats]:
    pair_01 = PairStats(index_a=0, index_b=1, wins_a=8, wins_b=2, ties=0)
    pair_02 = PairStats(index_a=0, index_b=2, wins_a=3, wins_b=7, ties=0)
    result = PairwiseResult(
        task_name="mmlu",
        metric="accuracy:exact_match",
        margin=0.0,
        instance_count=100,
        models=[
            ModelMeta(label="model_a"),
            ModelMeta(label="model_b"),
            ModelMeta(label="model_c"),
        ],
        pairs=[pair_01, pair_02],
    )
    return result, pair_01, pair_02


def test_build_win_rate_matrix_uses_pair_stats_and_defaults_missing_pairs() -> None:
    result, _, _ = _build_result()

    matrix = build_win_rate_matrix(result)

    assert np.isnan(matrix[0, 0])
    assert np.isnan(matrix[1, 1])
    assert np.isnan(matrix[2, 2])
    assert matrix[0, 1] == pytest.approx(0.8)
    assert matrix[1, 0] == pytest.approx(0.2)
    assert matrix[0, 2] == pytest.approx(0.3)
    assert matrix[2, 0] == pytest.approx(0.7)
    assert matrix[1, 2] == pytest.approx(0.5)
    assert matrix[2, 1] == pytest.approx(0.5)


def test_build_se_matrix_uses_pair_stats_and_defaults_missing_pairs() -> None:
    result, pair_01, pair_02 = _build_result()

    matrix = build_se_matrix(result)

    assert np.isnan(matrix[0, 0])
    assert np.isnan(matrix[1, 1])
    assert np.isnan(matrix[2, 2])
    assert matrix[0, 1] == pytest.approx(pair_01.se)
    assert matrix[1, 0] == pytest.approx(pair_01.se)
    assert matrix[0, 2] == pytest.approx(pair_02.se)
    assert matrix[2, 0] == pytest.approx(pair_02.se)
    assert matrix[1, 2] == 0.0
    assert matrix[2, 1] == 0.0

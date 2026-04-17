"""Tests for pairwise comparison logic."""

from __future__ import annotations

import pytest

from olmo_eval.analysis.pairwise import (
    ModelMeta,
    PairStats,
    PairwiseResult,
    _compute_pairs,
    get_win_rate,
)


class TestPairStats:
    def test_win_rate_a_basic(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=7, wins_b=3, ties=0)
        assert p.win_rate_a == pytest.approx(0.7)

    def test_win_rate_b_is_complement(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=7, wins_b=3, ties=0)
        assert p.win_rate_b == pytest.approx(0.3)

    def test_win_rate_excludes_ties(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=6, wins_b=4, ties=90)
        # win_rate is computed from contested instances only
        assert p.win_rate_a == pytest.approx(0.6)

    def test_all_ties_returns_half(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=0, wins_b=0, ties=100)
        assert p.win_rate_a == 0.5
        assert p.win_rate_b == 0.5

    def test_no_instances_returns_half(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=0, wins_b=0, ties=0)
        assert p.win_rate_a == 0.5


class TestGetWinRate:
    def setup_method(self) -> None:
        self.pairs = [
            PairStats(index_a=0, index_b=1, wins_a=8, wins_b=2, ties=0),
            PairStats(index_a=0, index_b=2, wins_a=3, wins_b=7, ties=0),
            PairStats(index_a=1, index_b=2, wins_a=5, wins_b=5, ties=0),
        ]

    def test_forward_lookup(self) -> None:
        assert get_win_rate(self.pairs, 0, 1) == pytest.approx(0.8)

    def test_reverse_lookup(self) -> None:
        assert get_win_rate(self.pairs, 1, 0) == pytest.approx(0.2)

    def test_missing_pair_returns_half(self) -> None:
        assert get_win_rate(self.pairs, 99, 100) == 0.5

    def test_symmetric(self) -> None:
        wr_ab = get_win_rate(self.pairs, 0, 2)
        wr_ba = get_win_rate(self.pairs, 2, 0)
        assert wr_ab + wr_ba == pytest.approx(1.0)


class TestComputePairs:
    def test_one_strictly_better(self) -> None:
        scores = {
            0: {"a": 1.0, "b": 1.0, "c": 1.0},
            1: {"a": 0.0, "b": 0.0, "c": 0.0},
        }
        shared = {"a", "b", "c"}
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        assert len(pairs) == 1
        assert pairs[0].wins_a == 3
        assert pairs[0].wins_b == 0
        assert pairs[0].ties == 0

    def test_identical_scores_all_ties(self) -> None:
        scores = {
            0: {"a": 0.5, "b": 0.5},
            1: {"a": 0.5, "b": 0.5},
        }
        shared = {"a", "b"}
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        assert pairs[0].ties == 2
        assert pairs[0].wins_a == 0
        assert pairs[0].wins_b == 0

    def test_margin_converts_close_scores_to_ties(self) -> None:
        scores = {
            0: {"a": 0.51, "b": 0.49, "c": 0.80},
            1: {"a": 0.50, "b": 0.50, "c": 0.10},
        }
        shared = {"a", "b", "c"}
        # margin=0.05 means a-b differences of 0.01 are ties
        pairs = _compute_pairs(scores, 2, shared, margin=0.05)
        assert pairs[0].ties == 2  # a (diff=0.01) and b (diff=0.01)
        assert pairs[0].wins_a == 1  # c (diff=0.70)

    def test_three_models(self) -> None:
        scores = {
            0: {"x": 1.0, "y": 0.0},
            1: {"x": 0.0, "y": 1.0},
            2: {"x": 0.5, "y": 0.5},
        }
        shared = {"x", "y"}
        pairs = _compute_pairs(scores, 3, shared, margin=0.0)
        # 3 pairs: (0,1), (0,2), (1,2)
        assert len(pairs) == 3

    def test_empty_shared_set(self) -> None:
        scores = {
            0: {"a": 1.0},
            1: {"b": 1.0},
        }
        pairs = _compute_pairs(scores, 2, set(), margin=0.0)
        assert pairs[0].wins_a == 0
        assert pairs[0].wins_b == 0
        assert pairs[0].ties == 0

    def test_preserves_index_order(self) -> None:
        scores = {
            0: {"a": 1.0},
            1: {"a": 0.0},
        }
        shared = {"a"}
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        assert pairs[0].index_a == 0
        assert pairs[0].index_b == 1
        assert pairs[0].wins_a == 1

    def test_skips_none_scores(self) -> None:
        # native_id "b" is in shared set but missing from scores_by_idx[0]
        scores: dict[int, dict[str, float]] = {
            0: {"a": 1.0},
            1: {"a": 0.0, "b": 0.5},
        }
        shared = {"a", "b"}
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        # "b" skipped because idx=0 doesn't have it
        assert pairs[0].wins_a == 1
        assert pairs[0].wins_b == 0
        assert pairs[0].ties == 0


class TestPairwiseResult:
    def test_result_fields(self) -> None:
        result = PairwiseResult(
            task_name="mmlu",
            metric="accuracy:exact_match",
            margin=0.0,
            instance_count=100,
            models=[
                ModelMeta(label="model_a"),
                ModelMeta(label="model_b"),
            ],
            pairs=[
                PairStats(index_a=0, index_b=1, wins_a=60, wins_b=40, ties=0),
            ],
        )
        assert result.instance_count == 100
        assert len(result.models) == 2
        assert len(result.pairs) == 1

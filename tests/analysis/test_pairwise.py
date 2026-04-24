"""Tests for pairwise comparison logic."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql

from olmo_eval.analysis.eval_power import minimum_detectable_effect
from olmo_eval.analysis.pairwise import (
    ModelMeta,
    PairStats,
    PairwiseEligibilityError,
    PairwiseResult,
    _build_experiment_refetch_stmt,
    _compute_pairs,
    _extract_pairwise_instance_score,
    get_task_metric_profile,
    get_win_rate,
)
from olmo_eval.analysis.pairwise_metrics import build_row_metrics, build_task_metrics
from olmo_eval.analysis.pairwise_viewer_payload import build_pairwise_viewer_payload
from olmo_eval.common.metrics import PassPowKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types.base import EvalResult


class TestPairStats:
    def test_n_contested_excludes_ties(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=7, wins_b=3, ties=90)
        assert p.n_contested == 10

    def test_win_rate_a_basic(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=7, wins_b=3, ties=0)
        assert p.win_rate_a == pytest.approx(0.7)

    def test_win_rate_excludes_ties(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=6, wins_b=4, ties=90)
        assert p.win_rate_a == pytest.approx(0.6)

    def test_all_ties_returns_half(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=0, wins_b=0, ties=100)
        assert p.win_rate_a == 0.5
        assert p.win_rate_b == 0.5

    def test_no_instances_returns_half(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=0, wins_b=0, ties=0)
        assert p.win_rate_a == 0.5

    def test_se_even_split(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=50, wins_b=50, ties=0)
        assert p.se == pytest.approx(math.sqrt(0.25 / 99), abs=1e-9)

    def test_se_skewed(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=7, wins_b=3, ties=0)
        assert p.se == pytest.approx(math.sqrt(0.21 / 9), abs=1e-9)

    def test_se_unanimous_is_zero(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=10, wins_b=0, ties=0)
        assert p.se == 0.0

    def test_se_degenerate_n_zero(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=0, wins_b=0, ties=0)
        assert p.se == 0.0

    def test_se_degenerate_n_one(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=1, wins_b=0, ties=0)
        assert p.se == 0.0

    def test_prob_a_gt_b_uses_same_normal_approximation_as_se(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=14, wins_b=6, ties=0)
        z = (p.win_rate_a - 0.5) / p.se
        expected = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        assert p.prob_a_gt_b == pytest.approx(expected)

    def test_prob_a_gt_b_degenerate_cases_fall_back_to_extremes_or_half(self) -> None:
        assert PairStats(index_a=0, index_b=1, wins_a=0, wins_b=0, ties=0).prob_a_gt_b == 0.5
        assert PairStats(index_a=0, index_b=1, wins_a=1, wins_b=0, ties=0).prob_a_gt_b == 1.0
        assert PairStats(index_a=0, index_b=1, wins_a=0, wins_b=1, ties=0).prob_a_gt_b == 0.0

    def test_p_value_keeps_tail_precision_for_strong_pairs(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=90, wins_b=10, ties=0)
        assert p.prob_a_gt_b == 1.0
        assert 0.0 < p.p_value < 1e-20


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
        pairs = _compute_pairs(scores, 2, shared, margin=0.05)
        assert pairs[0].ties == 2
        assert pairs[0].wins_a == 1

    def test_empty_shared_set(self) -> None:
        scores = {
            0: {"a": 1.0},
            1: {"b": 1.0},
        }
        pairs = _compute_pairs(scores, 2, set(), margin=0.0)
        assert pairs[0].wins_a == 0
        assert pairs[0].wins_b == 0
        assert pairs[0].ties == 0

    def test_skips_none_scores(self) -> None:
        scores: dict[int, dict[str, float]] = {
            0: {"a": 1.0},
            1: {"a": 0.0, "b": 0.5},
        }
        shared = {"a", "b"}
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        assert pairs[0].wins_a == 1
        assert pairs[0].wins_b == 0
        assert pairs[0].ties == 0


class TestPairwiseInstanceScoreExtraction:
    def test_pass_at_1_metrics_can_fall_back_to_underlying_scorer_channel(self) -> None:
        profile = get_task_metric_profile("humaneval:pass_at_1", "pass_at_1:code_exec")
        assert profile is not None
        assert profile.supports_scorer_fallback is True
        assert profile.display_format == "percentage"
        assert profile.higher_is_better is True
        assert _extract_pairwise_instance_score(
            task_name="humaneval:pass_at_1",
            metric_key="pass_at_1:code_exec",
            instance_metrics={"code_exec": {"code_exec": 1.0}},
        ) == pytest.approx(1.0)

    def test_pass_at_k_metrics_above_one_stay_blocked_without_exact_metric_key(self) -> None:
        profile = get_task_metric_profile("humaneval:pass_at_10", "pass_at_10:code_exec")
        assert profile is not None
        assert profile.supports_scorer_fallback is False
        assert (
            _extract_pairwise_instance_score(
                task_name="humaneval:pass_at_10",
                metric_key="pass_at_10:code_exec",
                instance_metrics={"code_exec": {"code_exec": 1.0}},
            )
            is None
        )

    def test_mc_accuracy_metrics_do_not_fall_back_to_raw_logprob_scores(self) -> None:
        profile = get_task_metric_profile("basic_skills_coding:rc::olmo3base", "accuracy:logprob")
        assert profile is not None
        assert profile.supports_scorer_fallback is False
        assert (
            _extract_pairwise_instance_score(
                task_name="basic_skills_coding:rc::olmo3base",
                metric_key="accuracy:logprob",
                instance_metrics={"logprob": {"logprob": -196.8}},
            )
            is None
        )

    def test_mc_accuracy_metrics_still_use_exact_metric_key_when_present(self) -> None:
        assert _extract_pairwise_instance_score(
            task_name="basic_skills_coding:rc::olmo3base",
            metric_key="accuracy:logprob",
            instance_metrics={
                "accuracy": {"logprob": 1.0},
                "logprob": {"logprob": -196.8},
            },
        ) == pytest.approx(1.0)

    def test_simple_mean_metrics_can_still_use_scorer_channel(self) -> None:
        profile = get_task_metric_profile("gsm8k:olmo3base", "accuracy:exact_match")
        assert profile is not None
        assert profile.supports_scorer_fallback is True
        assert profile.display_format == "percentage"
        assert profile.higher_is_better is True
        assert _extract_pairwise_instance_score(
            task_name="gsm8k:olmo3base",
            metric_key="accuracy:exact_match",
            instance_metrics={"exact_match": {"exact_match": 1.0}},
        ) == pytest.approx(1.0)

    def test_lab_bench_refusal_metrics_stay_percentage_without_scorer_fallback(self) -> None:
        precision_profile = get_task_metric_profile(
            "lab_bench_litqa2",
            "precision:multiple_choice",
        )
        coverage_profile = get_task_metric_profile(
            "lab_bench_litqa2",
            "coverage:multiple_choice",
        )

        assert precision_profile is not None
        assert precision_profile.supports_scorer_fallback is False
        assert precision_profile.display_format == "percentage"
        assert precision_profile.unit == "proportion"

        assert coverage_profile is not None
        assert coverage_profile.supports_scorer_fallback is False
        assert coverage_profile.display_format == "percentage"
        assert coverage_profile.unit == "proportion"

        assert (
            _extract_pairwise_instance_score(
                task_name="lab_bench_litqa2",
                metric_key="precision:multiple_choice",
                instance_metrics={"multiple_choice": {"multiple_choice": 1.0}},
            )
            is None
        )
        assert (
            _extract_pairwise_instance_score(
                task_name="lab_bench_litqa2",
                metric_key="coverage:multiple_choice",
                instance_metrics={"multiple_choice": {"multiple_choice": 1.0}},
            )
            is None
        )

    def test_bpb_metrics_resolve_as_raw_lower_is_better(self) -> None:
        profile = get_task_metric_profile("humaneval:bpb", "bits_per_byte:bits_per_byte")
        assert profile is not None
        assert profile.supports_scorer_fallback is False
        assert profile.display_format == "raw"
        assert profile.higher_is_better is False
        assert profile.unit == "bits_per_byte"

    def test_byte_weighted_bpb_metrics_resolve_without_crashing(self) -> None:
        profile = get_task_metric_profile("ds1000:bpb", "bits_per_byte:bits_per_byte")
        assert profile is not None
        assert profile.supports_scorer_fallback is False
        assert profile.display_format == "raw"
        assert profile.higher_is_better is False
        assert profile.unit == "bits_per_byte"

    def test_pass_pow_k_only_allows_scorer_fallback_for_k_equals_one(self) -> None:
        k1_metric = PassPowKMetric(scorer=CodeExecutionScorer, k=1)
        k2_metric = PassPowKMetric(scorer=CodeExecutionScorer, k=2)
        assert k1_metric.supports_pairwise_scorer_fallback() is True
        assert k2_metric.supports_pairwise_scorer_fallback() is False


class TestPairwiseEligibilityError:
    def test_to_payload_preserves_structured_diagnostics(self) -> None:
        error = PairwiseEligibilityError(
            code="insufficient_matched_experiments",
            summary="Only 1 run matched the paired-test requirements for this scope.",
            filter_summary="groups=['olmo-3-parity-apr5'], suite='mmlu:humanities:mc:olmo3base'",
            counts=[{"label": "matched runs", "value": 1}],
            matched_runs=[{"label": "allenai/OLMo-3-1025-7B (abc12345)"}],
            suggestions=["Broaden the filters."],
        )

        payload = error.to_payload()

        assert str(error) == "Only 1 run matched the paired-test requirements for this scope."
        assert payload["code"] == "insufficient_matched_experiments"
        assert payload["counts"] == [{"label": "matched runs", "value": 1}]
        assert payload["matched_runs"] == [{"label": "allenai/OLMo-3-1025-7B (abc12345)"}]
        assert payload["suggestions"] == ["Broaden the filters."]


class TestPairwiseResult:
    def test_html_payload_uses_direct_p_values_instead_of_rounded_probabilities(self) -> None:
        result = PairwiseResult(
            task_name="olmobase:math",
            metric="accuracy:exact_match",
            margin=0.0,
            instance_count=100,
            models=[
                ModelMeta(label="model-a\n(abc12345)", model_name="model-a", model_hash="abc12345"),
                ModelMeta(label="model-b\n(def67890)", model_name="model-b", model_hash="def67890"),
            ],
            pairs=[PairStats(index_a=0, index_b=1, wins_a=90, wins_b=10, ties=0)],
            model_shared_scores=(0.9, 0.1),
        )

        payload = build_pairwise_viewer_payload(result)
        probability = payload["matrix"]["probability"][0][1]
        p_value = payload["matrix"]["p_value"][0][1]

        assert probability == 1.0
        assert p_value is not None
        assert 0.0 < p_value < 1e-20

    def test_html_payload_includes_matrix_mde80_reference(self) -> None:
        result = PairwiseResult(
            task_name="olmobase:math",
            metric="accuracy:exact_match",
            margin=0.0,
            instance_count=100,
            models=[
                ModelMeta(label="model-a\n(abc12345)", model_name="model-a", model_hash="abc12345"),
                ModelMeta(label="model-b\n(def67890)", model_name="model-b", model_hash="def67890"),
            ],
            pairs=[
                PairStats(
                    index_a=0,
                    index_b=1,
                    wins_a=60,
                    wins_b=40,
                    ties=0,
                    var_paired_diff=0.04,
                )
            ],
            model_shared_scores=(0.6, 0.4),
        )

        payload = build_pairwise_viewer_payload(result)

        expected = minimum_detectable_effect(
            n=100,
            omega2=0.04,
            alpha=0.05,
            power=0.80,
        )
        assert payload["meta"]["mde80"] == pytest.approx(expected)
        assert payload["meta"]["mde80_by_alpha"]["0.05"] == pytest.approx(expected)

    def test_html_payload_marks_raw_lower_is_better_metrics(self) -> None:
        result = PairwiseResult(
            task_name="humaneval:bpb",
            metric="bits_per_byte:bits_per_byte",
            margin=0.0,
            instance_count=100,
            models=[
                ModelMeta(label="model-a\n(abc12345)", model_name="model-a", model_hash="abc12345"),
                ModelMeta(label="model-b\n(def67890)", model_name="model-b", model_hash="def67890"),
            ],
            pairs=[PairStats(index_a=0, index_b=1, wins_a=60, wins_b=40, ties=0)],
            model_shared_scores=(0.41, 0.57),
            score_display_format="raw",
            score_unit="bits_per_byte",
            higher_is_better=False,
        )

        payload = build_pairwise_viewer_payload(result)

        assert payload["meta"]["score_display_format"] == "raw"
        assert payload["meta"]["score_unit"] == "bits_per_byte"
        assert payload["meta"]["higher_is_better"] is False
        assert payload["matrix"]["score_diff"][0][1] == pytest.approx(-0.16)

    def test_lower_is_better_summaries_pick_smallest_scores_as_best(self) -> None:
        result = PairwiseResult(
            task_name="code:bpb",
            metric="bits_per_byte:bits_per_byte",
            margin=0.0,
            instance_count=50,
            models=[
                ModelMeta(label="model-a\n(aaa11111)", model_name="model-a", model_hash="aaa11111"),
                ModelMeta(label="model-b\n(bbb22222)", model_name="model-b", model_hash="bbb22222"),
            ],
            pairs=[],
            task_names=("ds1000:bpb", "bigcodebench:bpb"),
            task_metric_keys=("bits_per_byte:bits_per_byte", "bits_per_byte:bits_per_byte"),
            model_task_scores=((0.90, 0.20), (0.30, 0.40)),
            score_display_format="raw",
            score_unit="bits_per_byte",
            higher_is_better=False,
        )

        row_metrics = build_row_metrics(result)
        task_metrics = build_task_metrics(result)
        payload = build_pairwise_viewer_payload(result)

        assert row_metrics[0].best_task_label == "bigcodebench"
        assert row_metrics[0].best_task_score == pytest.approx(0.20)
        assert row_metrics[0].worst_task_label == "ds1000"
        assert row_metrics[0].worst_task_score == pytest.approx(0.90)

        assert task_metrics[0].best_model_label == "model-b (bbb22222)"
        assert task_metrics[0].best_model_score == pytest.approx(0.30)
        assert task_metrics[0].worst_model_label == "model-a (aaa11111)"
        assert task_metrics[0].worst_model_score == pytest.approx(0.90)

        assert payload["models"][0]["best_task_score"] == pytest.approx(0.20)
        assert payload["models"][0]["worst_task_score"] == pytest.approx(0.90)
        assert payload["task_stats"][0]["best_model_score"] == pytest.approx(0.30)
        assert payload["task_stats"][0]["worst_model_score"] == pytest.approx(0.90)

    def test_html_payload_keeps_same_name_task_hashes_distinct(self) -> None:
        result = PairwiseResult(
            task_name="gsm8k:olmo3base",
            metric="accuracy:exact_match",
            margin=0.0,
            instance_count=20,
            models=[
                ModelMeta(label="model-a\n(abc12345)", model_name="model-a", model_hash="abc12345"),
                ModelMeta(label="model-b\n(def67890)", model_name="model-b", model_hash="def67890"),
            ],
            pairs=[PairStats(index_a=0, index_b=1, wins_a=12, wins_b=8, ties=0)],
            task_names=("gsm8k:olmo3base", "gsm8k:olmo3base"),
            task_hashes=("task-hash-alpha", "task-hash-beta"),
            task_metric_keys=("accuracy:exact_match", "accuracy:exact_match"),
            model_task_scores=((0.70, 0.55), (0.60, 0.45)),
            model_shared_scores=(0.625, 0.525),
        )

        payload = build_pairwise_viewer_payload(result)

        assert [column["id"] for column in payload["task_columns"]] == [
            "task-hash-alpha",
            "task-hash-beta",
        ]
        assert all("[" in column["full_label"] for column in payload["task_columns"])
        assert payload["models"][0]["task_scores"]["task-hash-alpha"] == pytest.approx(0.70)
        assert payload["models"][0]["task_scores"]["task-hash-beta"] == pytest.approx(0.55)
        assert [task_stat["id"] for task_stat in payload["task_stats"]] == [
            "task-hash-alpha",
            "task-hash-beta",
        ]


class TestComputePairsCompoundKeys:
    """Compound keys keep identical native IDs in different tasks distinct."""

    def test_same_native_id_different_tasks_not_collapsed(self) -> None:
        scores = {
            0: {("task_1", "q1"): 1.0, ("task_2", "q1"): 1.0},
            1: {("task_1", "q1"): 0.0, ("task_2", "q1"): 0.0},
        }
        shared = {("task_1", "q1"), ("task_2", "q1")}
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        assert pairs[0].wins_a == 2
        assert pairs[0].wins_b == 0


class TestBuildExperimentRefetchStmt:
    @staticmethod
    def _eval_result(experiment_id: str, model_hash: str | None) -> EvalResult:
        return EvalResult(
            experiment_id=experiment_id,
            model_name=f"model-{experiment_id}",
            backend_name="backend",
            timestamp=datetime(2026, 4, 19, tzinfo=UTC),
            model_hash=model_hash,
        )

    def test_uses_exact_deduped_experiment_hash_pairs(self) -> None:
        stmt = _build_experiment_refetch_stmt(
            [
                self._eval_result("exp2", "hashB"),
                self._eval_result("exp1", "hashA"),
                self._eval_result("exp2", "hashB"),
                self._eval_result("exp0", None),
            ]
        )

        assert stmt is not None
        compiled = stmt.compile(dialect=postgresql.dialect())
        sql = str(compiled)

        assert "(experiments.experiment_id, experiments.model_hash) IN" in sql
        assert "experiments.experiment_id IN" not in sql
        assert "experiments.model_hash IN" not in sql
        assert len(compiled.params) == 1
        assert list(compiled.params.values()) == [[("exp1", "hashA"), ("exp2", "hashB")]]

    def test_returns_none_when_no_non_null_model_hashes_exist(self) -> None:
        stmt = _build_experiment_refetch_stmt(
            [
                self._eval_result("exp0", None),
                self._eval_result("exp1", None),
            ]
        )

        assert stmt is None

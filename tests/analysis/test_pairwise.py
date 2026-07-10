"""Tests for pairwise comparison logic."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from olmo_eval.analysis.eval_power import minimum_detectable_effect
from olmo_eval.analysis.pairwise import (
    ModelMeta,
    PairStats,
    PairwiseEligibilityError,
    PairwiseResult,
    _build_experiment_refetch_stmt,
    _build_scope_task_alias_map,
    _canonicalize_scope_task_rows,
    _compute_pairs,
    _equivalent_scope_task_names,
    _extract_pairwise_instance_score,
    _merge_latest_instance_key_rows,
    _merge_latest_instance_score_rows,
    _merge_latest_task_count_rows,
    _merge_latest_task_rows,
    _update_task_rows_num_instances,
    _update_task_rows_num_instances_from_counts,
    compute_pairwise,
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


class TestScopeTaskAliases:
    def test_noop_variant_aliases_expand_to_same_scope_leaf(self) -> None:
        aliases = _equivalent_scope_task_names("lambada:olmo3base")

        assert "lambada:olmo3base" in aliases
        assert "lambada" in aliases

    def test_non_equivalent_variant_does_not_alias_to_base_task(self) -> None:
        aliases = _equivalent_scope_task_names("medmcqa:olmo3base")

        assert "medmcqa:olmo3base" in aliases
        assert "medmcqa" not in aliases

    def test_canonicalize_scope_task_rows_maps_alias_rows_back_to_suite_leaf(self) -> None:
        _expanded, alias_map = _build_scope_task_alias_map(("lambada:olmo3base",))

        rows = _canonicalize_scope_task_rows(
            [
                SimpleNamespace(
                    experiment_pk=17,
                    task_name="lambada",
                    task_hash="054bdcd6",
                    num_instances=5153,
                    metrics={"greedy_accuracy": {"greedy_accuracy": 0.5}},
                    primary_metric="greedy_accuracy:greedy_accuracy",
                )
            ],
            alias_to_canonical=alias_map,
        )

        assert len(rows) == 1
        assert rows[0].task_name == "lambada:olmo3base"
        assert rows[0].task_hash == "054bdcd6"


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
        profile = get_task_metric_profile("basic_skills_coding:rc:olmo3base", "accuracy:logprob")
        assert profile is not None
        assert profile.supports_scorer_fallback is False
        assert (
            _extract_pairwise_instance_score(
                task_name="basic_skills_coding:rc:olmo3base",
                metric_key="accuracy:logprob",
                instance_metrics={"logprob": {"logprob": -196.8}},
            )
            is None
        )

    def test_mc_accuracy_metrics_still_use_exact_metric_key_when_present(self) -> None:
        assert _extract_pairwise_instance_score(
            task_name="basic_skills_coding:rc:olmo3base",
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

    def test_html_payload_scope_score_uses_suite_aggregation_strategy(self) -> None:
        from olmo_eval.evals.suites.registry import _REGISTRY, AggregationStrategy, Suite

        nested_suite = Suite(
            name="_test_nested_payload",
            tasks=("task_a", "task_b", "task_c"),
            aggregation=AggregationStrategy.AVERAGE,
        )
        aoa_suite = Suite(
            name="_test_aoa_payload",
            tasks=("task_single", nested_suite),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )
        _REGISTRY["_test_aoa_payload"] = aoa_suite

        try:
            result = PairwiseResult(
                task_name="_test_aoa_payload",
                suite_name="_test_aoa_payload",
                metric="accuracy:exact_match",
                margin=0.0,
                instance_count=20,
                models=[
                    ModelMeta(
                        label="model-a\n(abc12345)",
                        model_name="model-a",
                        model_hash="abc12345",
                    ),
                    ModelMeta(
                        label="model-b\n(def67890)",
                        model_name="model-b",
                        model_hash="def67890",
                    ),
                ],
                pairs=[PairStats(index_a=0, index_b=1, wins_a=12, wins_b=8, ties=0)],
                task_names=("task_single", "task_a", "task_b", "task_c"),
                task_hashes=("task-single", "task-a", "task-b", "task-c"),
                task_metric_keys=("accuracy:exact_match",) * 4,
                model_task_scores=((1.0, 0.4, 0.5, 0.6), (0.8, 0.2, 0.3, 0.4)),
                score_display_format="percentage",
                score_unit="proportion",
                higher_is_better=True,
            )

            payload = build_pairwise_viewer_payload(result)

            assert payload["models"][0]["scope_score"] == pytest.approx(0.75)
            assert payload["models"][1]["scope_score"] == pytest.approx(0.55)
            assert payload["models"][0]["display_score"] == pytest.approx(0.75)
            assert "avg_task_score" not in payload["models"][0]
        finally:
            del _REGISTRY["_test_aoa_payload"]


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


class TestLatestRunMergeHelpers:
    @staticmethod
    def _experiment(experiment_id: int, *, hour: int) -> SimpleNamespace:
        return SimpleNamespace(
            id=experiment_id,
            model_name="model-a",
            model_hash="abc12345deadbeef",
            timestamp=datetime(2026, 4, 21, hour, 0, tzinfo=UTC),
        )

    def test_merge_latest_task_rows_backfills_missing_tasks_and_prefers_newest_metric_values(
        self,
    ) -> None:
        older = self._experiment(10, hour=8)
        latest = self._experiment(11, hour=12)

        task_rows = [
            SimpleNamespace(
                experiment_pk=10,
                task_name="task-a",
                task_hash="task-hash-a",
                num_instances=2,
                metrics={
                    "accuracy": {"exact_match": 0.40},
                    "f1": {"exact_match": 0.70},
                },
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-a",
                task_hash="task-hash-a",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.60}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_name="task-b",
                task_hash="task-hash-b",
                num_instances=3,
                metrics={"accuracy": {"exact_match": 0.80}},
                primary_metric="accuracy:exact_match",
            ),
        ]

        merged_rows = _merge_latest_task_rows(
            task_rows=task_rows,
            source_experiments=[older, latest],
            display_experiments=[latest],
        )

        assert {row.task_hash for row in merged_rows} == {"task-hash-a", "task-hash-b"}
        row_by_hash = {row.task_hash: row for row in merged_rows}

        assert row_by_hash["task-hash-a"].experiment_pk == 11
        assert row_by_hash["task-hash-a"].metrics["accuracy"]["exact_match"] == pytest.approx(0.60)
        assert row_by_hash["task-hash-a"].metrics["f1"]["exact_match"] == pytest.approx(0.70)
        assert row_by_hash["task-hash-b"].experiment_pk == 11
        assert row_by_hash["task-hash-b"].num_instances == 3

    def test_merge_latest_instance_rows_backfills_keys_and_prefers_latest_non_null_scores(
        self,
    ) -> None:
        older = self._experiment(10, hour=8)
        latest = self._experiment(11, hour=12)

        instance_rows = [
            SimpleNamespace(experiment_pk=10, task_hash="task-hash-a", native_id="q1"),
            SimpleNamespace(experiment_pk=10, task_hash="task-hash-a", native_id="q3"),
            SimpleNamespace(experiment_pk=11, task_hash="task-hash-a", native_id="q1"),
            SimpleNamespace(experiment_pk=11, task_hash="task-hash-a", native_id="q2"),
        ]

        merged_keys = _merge_latest_instance_key_rows(
            instance_rows=instance_rows,
            source_experiments=[older, latest],
            display_experiments=[latest],
        )

        assert {(row.experiment_pk, row.task_hash, row.native_id) for row in merged_keys} == {
            (11, "task-hash-a", "q1"),
            (11, "task-hash-a", "q2"),
            (11, "task-hash-a", "q3"),
        }

        score_rows = [
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-hash-a",
                native_id="q1",
                raw_score=1.0,
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_hash="task-hash-a",
                native_id="q1",
                raw_score=None,
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-hash-a",
                native_id="q2",
                raw_score=0.2,
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_hash="task-hash-a",
                native_id="q2",
                raw_score=0.9,
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-hash-a",
                native_id="q3",
                raw_score=0.4,
            ),
        ]

        merged_scores = _merge_latest_instance_score_rows(
            instance_rows=score_rows,
            source_experiments=[older, latest],
            display_experiments=[latest],
        )

        score_by_native = {row.native_id: row.raw_score for row in merged_scores}
        assert score_by_native["q1"] == pytest.approx(1.0)
        assert score_by_native["q2"] == pytest.approx(0.9)
        assert score_by_native["q3"] == pytest.approx(0.4)

    def test_merge_latest_instance_rows_accepts_sql_score_alias(self) -> None:
        older = self._experiment(10, hour=8)
        latest = self._experiment(11, hour=12)

        score_rows = [
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-hash-a",
                native_id="q1",
                score=0.4,
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_hash="task-hash-a",
                native_id="q1",
                score=0.9,
            ),
        ]

        merged_scores = _merge_latest_instance_score_rows(
            instance_rows=score_rows,
            source_experiments=[older, latest],
            display_experiments=[latest],
        )

        assert len(merged_scores) == 1
        assert merged_scores[0].experiment_pk == 11
        assert merged_scores[0].native_id == "q1"
        assert merged_scores[0].raw_score == pytest.approx(0.9)

    def test_update_task_rows_num_instances_uses_merged_instance_counts(self) -> None:
        task_rows = [
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-a",
                task_hash="task-hash-a",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.60}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-b",
                task_hash="task-hash-b",
                num_instances=5,
                metrics={"accuracy": {"exact_match": 0.80}},
                primary_metric="accuracy:exact_match",
            ),
        ]
        instance_rows = [
            SimpleNamespace(experiment_pk=11, task_hash="task-hash-a", native_id="q1"),
            SimpleNamespace(experiment_pk=11, task_hash="task-hash-a", native_id="q2"),
            SimpleNamespace(experiment_pk=11, task_hash="task-hash-a", native_id="q3"),
        ]

        updated_rows = _update_task_rows_num_instances(
            task_rows=task_rows,
            instance_rows=instance_rows,
        )

        row_by_hash = {row.task_hash: row for row in updated_rows}
        assert row_by_hash["task-hash-a"].num_instances == 3
        assert row_by_hash["task-hash-b"].num_instances == 5

    def test_merge_latest_task_count_rows_maps_model_hash_to_latest_display_run(self) -> None:
        latest = self._experiment(11, hour=12)

        count_rows = [
            SimpleNamespace(
                model_hash="abc12345deadbeef",
                task_hash="task-hash-a",
                num_instances=7,
            )
        ]

        merged_rows = _merge_latest_task_count_rows(
            count_rows=count_rows,
            display_experiments=[latest],
        )

        assert len(merged_rows) == 1
        assert merged_rows[0].experiment_pk == 11
        assert merged_rows[0].task_hash == "task-hash-a"
        assert merged_rows[0].num_instances == 7

    def test_update_task_rows_num_instances_from_counts_uses_aggregated_counts(self) -> None:
        task_rows = [
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-a",
                task_hash="task-hash-a",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.60}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-b",
                task_hash="task-hash-b",
                num_instances=5,
                metrics={"accuracy": {"exact_match": 0.80}},
                primary_metric="accuracy:exact_match",
            ),
        ]
        count_rows = [
            SimpleNamespace(
                experiment_pk=11,
                task_hash="task-hash-a",
                num_instances=3,
            )
        ]

        updated_rows = _update_task_rows_num_instances_from_counts(
            task_rows=task_rows,
            count_rows=count_rows,
        )

        row_by_hash = {row.task_hash: row for row in updated_rows}
        assert row_by_hash["task-hash-a"].num_instances == 3
        assert row_by_hash["task-hash-b"].num_instances == 5


class SequenceExecuteResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)

    def scalar_one(self):
        if not self._rows:
            raise AssertionError("Expected one scalar result")
        return self._rows[0]

    def scalar_one_or_none(self):
        if not self._rows:
            return None
        return self._rows[0]


class SequenceSession:
    def __init__(self, responses, *, compile_queries: bool = False):
        self._responses = list(responses)
        self._compile_queries = compile_queries

    def execute(self, _query):
        if not self._responses:
            raise AssertionError("Unexpected session.execute() call")
        if self._compile_queries:
            _query.compile(dialect=postgresql.dialect())
        return SequenceExecuteResult(self._responses.pop(0))


class TestComputePairwiseLatestRunMerge:
    @staticmethod
    def _experiment(
        experiment_id: int,
        *,
        model_name: str,
        model_hash: str,
        hour: int,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=experiment_id,
            model_name=model_name,
            model_hash=model_hash,
            timestamp=datetime(2026, 4, 21, hour, 0, tzinfo=UTC),
        )

    def test_latest_mode_merges_compatible_repeated_runs(self, monkeypatch) -> None:
        from olmo_eval.evals.suites import registry as suite_registry
        from olmo_eval.storage.backends.postgres import repository as repository_mod

        older_a = self._experiment(10, model_name="model-a", model_hash="hash-a", hour=8)
        latest_a = self._experiment(11, model_name="model-a", model_hash="hash-a", hour=12)
        older_b = self._experiment(20, model_name="model-b", model_hash="hash-b", hour=7)
        latest_b = self._experiment(21, model_name="model-b", model_hash="hash-b", hour=11)
        all_experiments = [older_a, latest_a, older_b, latest_b]

        monkeypatch.setattr(
            repository_mod.ExperimentRepository,
            "query_rows",
            lambda self, **kwargs: list(all_experiments),
        )
        monkeypatch.setattr(suite_registry, "suite_exists", lambda name: name == "demo:suite")
        monkeypatch.setattr(
            suite_registry,
            "get_suite",
            lambda name: SimpleNamespace(expand=lambda: ("task-a", "task-b")),
        )

        task_rows = [
            SimpleNamespace(
                experiment_pk=10,
                task_name="task-a",
                task_hash="task-a-hash",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.50}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-a",
                task_hash="task-a-hash",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.90}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_name="task-b",
                task_hash="task-b-hash",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.40}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=20,
                task_name="task-a",
                task_hash="task-a-hash",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.30}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=21,
                task_name="task-a",
                task_hash="task-a-hash",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.70}},
                primary_metric="accuracy:exact_match",
            ),
            SimpleNamespace(
                experiment_pk=20,
                task_name="task-b",
                task_hash="task-b-hash",
                num_instances=2,
                metrics={"accuracy": {"exact_match": 0.60}},
                primary_metric="accuracy:exact_match",
            ),
        ]
        instance_count_rows = [
            SimpleNamespace(model_hash="hash-a", task_hash="task-a-hash", num_instances=2),
            SimpleNamespace(model_hash="hash-a", task_hash="task-b-hash", num_instances=2),
            SimpleNamespace(model_hash="hash-b", task_hash="task-a-hash", num_instances=2),
            SimpleNamespace(model_hash="hash-b", task_hash="task-b-hash", num_instances=2),
        ]
        preflight_rows = [
            SimpleNamespace(model_hash="hash-a"),
            SimpleNamespace(model_hash="hash-b"),
        ]
        score_rows = [
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-a-hash",
                native_id="q1",
                raw_score=0.2,
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-a-hash",
                native_id="q2",
                raw_score=0.2,
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_hash="task-a-hash",
                native_id="q1",
                raw_score=1.0,
            ),
            SimpleNamespace(
                experiment_pk=11,
                task_hash="task-a-hash",
                native_id="q2",
                raw_score=0.0,
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-b-hash",
                native_id="q3",
                raw_score=1.0,
            ),
            SimpleNamespace(
                experiment_pk=10,
                task_hash="task-b-hash",
                native_id="q4",
                raw_score=1.0,
            ),
            SimpleNamespace(
                experiment_pk=20,
                task_hash="task-a-hash",
                native_id="q1",
                raw_score=0.1,
            ),
            SimpleNamespace(
                experiment_pk=20,
                task_hash="task-a-hash",
                native_id="q2",
                raw_score=0.1,
            ),
            SimpleNamespace(
                experiment_pk=21,
                task_hash="task-a-hash",
                native_id="q1",
                raw_score=0.0,
            ),
            SimpleNamespace(
                experiment_pk=21,
                task_hash="task-a-hash",
                native_id="q2",
                raw_score=1.0,
            ),
            SimpleNamespace(
                experiment_pk=20,
                task_hash="task-b-hash",
                native_id="q3",
                raw_score=0.0,
            ),
            SimpleNamespace(
                experiment_pk=20,
                task_hash="task-b-hash",
                native_id="q4",
                raw_score=0.0,
            ),
        ]
        session = SequenceSession(
            [task_rows, instance_count_rows, preflight_rows, score_rows],
            compile_queries=True,
        )

        result = compute_pairwise(
            session=session,
            suite_name="demo:suite",
            keep_all=False,
            require_full_coverage=True,
        )

        assert [model.model_name for model in result.models] == ["model-a", "model-b"]
        assert [model.model_hash for model in result.models] == ["hash-a", "hash-b"]
        assert result.n_experiments_matched == 4
        assert result.n_experiments_dropped == 2
        assert result.instance_count == 4
        assert result.task_names == ("task-a", "task-b")
        assert result.task_hashes == ("task-a-hash", "task-b-hash")
        assert result.model_task_scores[0] == pytest.approx((0.90, 0.40))
        assert result.model_task_scores[1] == pytest.approx((0.70, 0.60))
        assert len(result.pairs) == 1
        assert result.pairs[0].wins_a == 3
        assert result.pairs[0].wins_b == 1
        assert session._responses == []

    def test_latest_mode_keeps_sample_metric_keys_when_sql_filters_out_null_scores(
        self,
        monkeypatch,
    ) -> None:
        from olmo_eval.evals.suites import registry as suite_registry
        from olmo_eval.storage.backends.postgres import repository as repository_mod

        latest_a = self._experiment(11, model_name="model-a", model_hash="hash-a", hour=12)
        latest_b = self._experiment(21, model_name="model-b", model_hash="hash-b", hour=11)
        all_experiments = [latest_a, latest_b]

        monkeypatch.setattr(
            repository_mod.ExperimentRepository,
            "query_rows",
            lambda self, **kwargs: list(all_experiments),
        )
        monkeypatch.setattr(suite_registry, "suite_exists", lambda name: name == "demo:suite")
        monkeypatch.setattr(
            suite_registry,
            "get_suite",
            lambda name: SimpleNamespace(expand=lambda: ("task-a",)),
        )

        task_rows = [
            SimpleNamespace(
                experiment_pk=11,
                task_name="task-a",
                task_hash="task-a-hash",
                num_instances=2,
                metrics={"accuracy": {"logprob": 0.90}},
                primary_metric="accuracy:logprob",
            ),
            SimpleNamespace(
                experiment_pk=21,
                task_name="task-a",
                task_hash="task-a-hash",
                num_instances=2,
                metrics={"accuracy": {"logprob": 0.70}},
                primary_metric="accuracy:logprob",
            ),
        ]
        instance_count_rows = [
            SimpleNamespace(model_hash="hash-a", task_hash="task-a-hash", num_instances=2),
            SimpleNamespace(model_hash="hash-b", task_hash="task-a-hash", num_instances=2),
        ]
        session = SequenceSession(
            [
                task_rows,
                instance_count_rows,
                [],
                [0],
                [{"exact_match": 1.0}],
            ],
            compile_queries=True,
        )

        with pytest.raises(PairwiseEligibilityError) as exc_info:
            compute_pairwise(
                session=session,
                suite_name="demo:suite",
                keep_all=False,
                require_full_coverage=True,
            )

        assert "Sample stored instance metric keys: exact_match" in exc_info.value.notes
        assert session._responses == []


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

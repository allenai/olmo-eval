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
    _filter_suite_task_names,
    _is_excluded_experiment,
    _is_excluded_task,
    _matches_exact,
    _matches_prefix,
    get_se,
    get_task_metric_profile,
    get_win_rate,
)
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

    def test_win_rate_b_is_complement(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=7, wins_b=3, ties=0)
        assert p.win_rate_b == pytest.approx(0.3)

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

    def test_se_ignores_ties(self) -> None:
        without_ties = PairStats(index_a=0, index_b=1, wins_a=50, wins_b=50, ties=0)
        with_ties = PairStats(index_a=0, index_b=1, wins_a=50, wins_b=50, ties=50)
        assert with_ties.se == pytest.approx(without_ties.se, abs=1e-12)

    def test_se_symmetric_in_p(self) -> None:
        hi = PairStats(index_a=0, index_b=1, wins_a=80, wins_b=20, ties=0)
        lo = PairStats(index_a=0, index_b=1, wins_a=20, wins_b=80, ties=0)
        assert hi.se == pytest.approx(lo.se, abs=1e-12)

    def test_prob_a_gt_b_even_split_is_half(self) -> None:
        p = PairStats(index_a=0, index_b=1, wins_a=50, wins_b=50, ties=0)
        assert p.prob_a_gt_b == pytest.approx(0.5)

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


class TestGetSe:
    def setup_method(self) -> None:
        self.pairs = [
            PairStats(index_a=0, index_b=1, wins_a=50, wins_b=50, ties=0),
            PairStats(index_a=0, index_b=2, wins_a=70, wins_b=30, ties=0),
        ]

    def test_forward_lookup(self) -> None:
        assert get_se(self.pairs, 0, 1) == pytest.approx(math.sqrt(0.25 / 99))

    def test_reverse_lookup_same(self) -> None:
        assert get_se(self.pairs, 1, 0) == get_se(self.pairs, 0, 1)

    def test_missing_pair_returns_zero(self) -> None:
        assert get_se(self.pairs, 99, 100) == 0.0


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


class TestExcludeHelpers:
    def test_matches_prefix(self) -> None:
        assert _matches_prefix("llama3.1-8b", ["llama", "qwen"]) is True
        assert _matches_prefix("llama3.1-8b", ["qwen"]) is False
        assert _matches_prefix(None, ["llama"]) is False

    def test_matches_exact(self) -> None:
        assert _matches_exact("gsm8k:olmo3base", ["gsm8k:olmo3base"]) is True
        assert _matches_exact("gsm8k:olmo3base", ["gsm8k"]) is False
        assert _matches_exact(None, ["gsm8k:olmo3base"]) is False

    def test_is_excluded_experiment_uses_prefix_matching(self) -> None:
        assert _is_excluded_experiment(
            model_name="llama3.1-8b",
            model_hash="abc12345deadbeef",
            exclude_model_names=["llama"],
        )
        assert _is_excluded_experiment(
            model_name="qwen2.5-7b",
            model_hash="abc12345deadbeef",
            exclude_model_hashes=["abc12345"],
        )
        assert not _is_excluded_experiment(
            model_name="qwen2.5-7b",
            model_hash="fff99999deadbeef",
            exclude_model_names=["llama"],
            exclude_model_hashes=["abc12345"],
        )

    def test_is_excluded_task_uses_exact_name_and_hash_prefix(self) -> None:
        assert _is_excluded_task(
            task_name="humaneval:3shot:pass_at_1",
            task_hash="abc12345deadbeef",
            exclude_task_names=["humaneval:3shot:pass_at_1"],
        )
        assert _is_excluded_task(
            task_name="humaneval:3shot:pass_at_1",
            task_hash="abc12345deadbeef",
            exclude_task_hashes=["abc12345"],
        )
        assert not _is_excluded_task(
            task_name="humaneval:3shot:pass_at_1",
            task_hash="abc12345deadbeef",
            exclude_task_names=["humaneval"],
            exclude_task_hashes=["fff99999"],
        )

    def test_filter_suite_task_names_preserves_order(self) -> None:
        assert _filter_suite_task_names(
            ("task_a", "task_b", "task_c"),
            exclude_task_names=["task_b"],
        ) == ("task_a", "task_c")


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

    def test_three_models(self) -> None:
        scores = {
            0: {"x": 1.0, "y": 0.0},
            1: {"x": 0.0, "y": 1.0},
            2: {"x": 0.5, "y": 0.5},
        }
        shared = {"x", "y"}
        pairs = _compute_pairs(scores, 3, shared, margin=0.0)
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

    def test_bpb_metrics_resolve_as_raw_lower_is_better(self) -> None:
        profile = get_task_metric_profile("humaneval:bpb", "bits_per_byte:bits_per_byte")
        assert profile is not None
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

    def test_suite_fields_default_empty(self) -> None:
        result = PairwiseResult(
            task_name="mmlu_abstract_algebra",
            metric="acc_raw:acc_raw",
            margin=0.0,
            instance_count=100,
            models=[],
            pairs=[],
        )
        assert result.suite_name is None
        assert result.task_names == ()
        assert result.model_task_scores == ()

    def test_suite_fields_populated(self) -> None:
        result = PairwiseResult(
            task_name="olmobase:math",
            metric="per-task primary",
            margin=0.0,
            instance_count=8000,
            models=[],
            pairs=[],
            suite_name="olmobase:math",
            task_names=("gsm8k:olmo3base", "minerva_math_algebra:olmo3base"),
        )
        assert result.suite_name == "olmobase:math"
        assert len(result.task_names) == 2

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


class TestComputePairsCompoundKeys:
    """Compound keys keep identical native IDs in different tasks distinct."""

    def test_pools_across_tasks(self) -> None:
        scores = {
            0: {
                ("task_1", "q1"): 1.0,
                ("task_1", "q2"): 1.0,
                ("task_1", "q3"): 1.0,
                ("task_2", "q1"): 0.0,
                ("task_2", "q2"): 0.0,
                ("task_2", "q3"): 0.0,
            },
            1: {
                ("task_1", "q1"): 0.0,
                ("task_1", "q2"): 0.0,
                ("task_1", "q3"): 0.0,
                ("task_2", "q1"): 1.0,
                ("task_2", "q2"): 1.0,
                ("task_2", "q3"): 1.0,
            },
        }
        shared = set(scores[0].keys())
        pairs = _compute_pairs(scores, 2, shared, margin=0.0)
        assert pairs[0].wins_a == 3
        assert pairs[0].wins_b == 3
        assert pairs[0].ties == 0

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

"""Tests for olmo_eval.runner module."""

import pytest

# Import to ensure tasks and suites are registered
import olmo_eval.evals  # noqa: F401
import olmo_eval.evals.tasks  # noqa: F401
from olmo_eval.runners import EvalRunner, ValidationError


class TestEvalRunnerValidation:
    """Tests for EvalRunner.validate method."""

    def test_validate_valid_task(self):
        """Test validation passes for valid task."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["humaneval"],
        )
        # Should not raise
        runner.validate()

    def test_validate_valid_suite(self):
        """Test validation passes for valid suite."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["mt_mbpp_v2fix"],
        )
        # Should not raise
        runner.validate()

    def test_validate_multiple_valid_specs(self):
        """Test validation passes for multiple valid specs."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["humaneval", "mbpp", "mt_mbpp_v2fix"],
        )
        # Should not raise
        runner.validate()

    def test_validate_invalid_task_raises(self):
        """Test validation fails for unknown task."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["nonexistent_task"],
        )
        with pytest.raises(ValidationError, match="Unknown task or suite"):
            runner.validate()

    def test_validate_invalid_suite_raises(self):
        """Test validation fails for unknown suite."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["nonexistent:suite"],
        )
        with pytest.raises(ValidationError, match="Unknown task or suite"):
            runner.validate()

    def test_validate_invalid_regime_raises(self):
        """Test validation fails for unknown variant/regime.

        Note: Regimes are now accessed as variants using single colon syntax.
        """
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["humaneval:nonexistent_regime"],
        )
        with pytest.raises(ValidationError, match="Unknown variant/regime"):
            runner.validate()

    def test_validate_collects_multiple_errors(self):
        """Test validation collects all errors."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["bad_task1", "bad_task2", "humaneval"],
        )
        with pytest.raises(ValidationError) as exc_info:
            runner.validate()

        error_msg = str(exc_info.value)
        assert "bad_task1" in error_msg
        assert "bad_task2" in error_msg

    def test_validate_mixed_valid_and_invalid(self):
        """Test validation fails if any spec is invalid."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["humaneval", "nonexistent", "mt_mbpp_v2fix"],
        )
        with pytest.raises(ValidationError, match="nonexistent"):
            runner.validate()

    def test_validate_empty_task_specs(self):
        """Test validation passes with empty task specs."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=[],
        )
        # Should not raise (though running would be pointless)
        runner.validate()


class TestSuiteAggregations:
    """Tests for compute_suite_aggregations function."""

    def test_suite_aggregation_basic(self):
        """Test basic suite aggregation without overrides."""
        # Create mock task results that match expanded mt_mbpp_v2fix tasks
        # Get actual tasks in mt_mbpp_v2fix suite
        from olmo_eval.evals.suites import get_suite
        from olmo_eval.runners.utils import compute_suite_aggregations

        suite = get_suite("mt_mbpp_v2fix")
        expanded_tasks = suite.expand()

        # Create mock results for each task (nested format)
        task_results = {}
        for task in expanded_tasks:
            task_results[task] = {"metrics": {"accuracy": {"exact_match": 0.75}}}

        result = compute_suite_aggregations(["mt_mbpp_v2fix"], task_results)

        assert "mt_mbpp_v2fix" in result
        assert result["mt_mbpp_v2fix"]["metrics"]["accuracy"]["exact_match"] == 0.75
        assert result["mt_mbpp_v2fix"]["num_tasks"] == len(expanded_tasks)

    def test_suite_aggregation_with_overrides(self):
        """Test suite aggregation with inline overrides."""
        from olmo_eval.evals.suites import get_suite
        from olmo_eval.runners.utils import compute_suite_aggregations

        suite = get_suite("mt_mbpp_v2fix")
        expanded_tasks = suite.expand()

        # Create mock results with override suffix (as would happen in real run)
        task_results = {}
        for task in expanded_tasks:
            task_results[f"{task}::temperature=0.6"] = {
                "metrics": {"accuracy": {"exact_match": 0.80}}
            }

        result = compute_suite_aggregations(["mt_mbpp_v2fix::temperature=0.6"], task_results)

        assert "mt_mbpp_v2fix::temperature=0.6" in result
        assert result["mt_mbpp_v2fix::temperature=0.6"]["metrics"]["accuracy"][
            "exact_match"
        ] == pytest.approx(0.80)
        assert result["mt_mbpp_v2fix::temperature=0.6"]["num_tasks"] == len(expanded_tasks)

    def test_suite_aggregation_with_priority(self):
        """Test suite aggregation with priority suffix."""
        from olmo_eval.evals.suites import get_suite
        from olmo_eval.runners.utils import compute_suite_aggregations

        suite = get_suite("mt_mbpp_v2fix")
        expanded_tasks = suite.expand()

        # Create mock results with priority suffix
        task_results = {}
        for task in expanded_tasks:
            task_results[f"{task}@high"] = {"metrics": {"accuracy": {"exact_match": 0.85}}}

        result = compute_suite_aggregations(["mt_mbpp_v2fix@high"], task_results)

        assert "mt_mbpp_v2fix@high" in result
        assert result["mt_mbpp_v2fix@high"]["metrics"]["accuracy"]["exact_match"] == pytest.approx(
            0.85
        )

    def test_suite_aggregation_with_overrides_and_priority(self):
        """Test suite aggregation with both overrides and priority."""
        from olmo_eval.evals.suites import get_suite
        from olmo_eval.runners.utils import compute_suite_aggregations

        suite = get_suite("mt_mbpp_v2fix")
        expanded_tasks = suite.expand()

        # Create mock results with both suffixes (order: ::overrides@priority)
        task_results = {}
        for task in expanded_tasks:
            task_results[f"{task}::temperature=0@urgent"] = {
                "metrics": {"accuracy": {"exact_match": 0.90}}
            }

        result = compute_suite_aggregations(["mt_mbpp_v2fix::temperature=0@urgent"], task_results)

        assert "mt_mbpp_v2fix::temperature=0@urgent" in result
        assert result["mt_mbpp_v2fix::temperature=0@urgent"]["metrics"]["accuracy"][
            "exact_match"
        ] == pytest.approx(0.90)

    def test_suite_aggregation_non_suite_ignored(self):
        """Test that non-suite specs are ignored."""
        from olmo_eval.runners.utils import compute_suite_aggregations

        task_results = {"humaneval": {"metrics": {"accuracy": {"exact_match": 0.75}}}}

        result = compute_suite_aggregations(["humaneval"], task_results)

        # humaneval is a task, not a suite
        assert result == {}

    def test_suite_aggregation_average_of_averages(self):
        """Test AVERAGE_OF_AVERAGES aggregation weights children equally."""
        from olmo_eval.evals.suites.registry import (
            _REGISTRY,
            AggregationStrategy,
            Suite,
        )
        from olmo_eval.runners.utils import compute_suite_aggregations

        # Create a nested suite for testing
        nested_suite = Suite(
            name="_test_nested",
            tasks=("task_a", "task_b", "task_c"),  # 3 tasks
            aggregation=AggregationStrategy.AVERAGE,
        )

        # Create an average-of-averages suite
        aoa_suite = Suite(
            name="_test_aoa",
            tasks=(
                "task_single",  # Individual task
                nested_suite,  # Nested suite with 3 tasks
            ),
            aggregation=AggregationStrategy.AVERAGE_OF_AVERAGES,
        )

        # Register temporarily
        _REGISTRY["_test_aoa"] = aoa_suite

        try:
            # Create task results (nested format):
            # - task_single: 1.0
            # - task_a, task_b, task_c: 0.4, 0.5, 0.6 (average = 0.5)
            # Expected AVERAGE_OF_AVERAGES: (1.0 + 0.5) / 2 = 0.75
            # (NOT simple average: (1.0 + 0.4 + 0.5 + 0.6) / 4 = 0.625)
            task_results = {
                "task_single": {"metrics": {"bits_per_byte": {"bpb_scorer": 1.0}}},
                "task_a": {"metrics": {"bits_per_byte": {"bpb_scorer": 0.4}}},
                "task_b": {"metrics": {"bits_per_byte": {"bpb_scorer": 0.5}}},
                "task_c": {"metrics": {"bits_per_byte": {"bpb_scorer": 0.6}}},
            }

            result = compute_suite_aggregations(["_test_aoa"], task_results)

            # Check top-level suite aggregation
            assert "_test_aoa" in result
            # Average of averages: (1.0 + 0.5) / 2 = 0.75
            assert result["_test_aoa"]["metrics"]["bits_per_byte"]["bpb_scorer"] == pytest.approx(
                0.75
            )
            assert result["_test_aoa"]["num_tasks"] == 4  # All tasks included
            assert result["_test_aoa"]["num_children"] == 2  # 2 children
            assert result["_test_aoa"]["aggregation"] == "average_of_averages"
            assert result["_test_aoa"]["nested_suites"] == ["_test_nested"]

            # Check nested suite aggregation is also reported
            assert "_test_nested" in result
            # Nested suite average: (0.4 + 0.5 + 0.6) / 3 = 0.5
            assert result["_test_nested"]["metrics"]["bits_per_byte"][
                "bpb_scorer"
            ] == pytest.approx(0.5)
            assert result["_test_nested"]["num_tasks"] == 3
            assert result["_test_nested"]["aggregation"] == "average"
            assert result["_test_nested"]["parent_suite"] == "_test_aoa"
        finally:
            # Clean up
            del _REGISTRY["_test_aoa"]


class TestGetPrimaryMetric:
    """Tests for get_primary_metric function.

    Note: Metrics are now nested: {metric_name: {scorer_name: value}}.
    The preferred parameter uses "metric:scorer" format, and the result
    is ("metric:scorer", value).
    """

    def test_preferred_metric_used_when_present(self):
        """Test that preferred metric is used when specified and present."""
        from olmo_eval.runners.utils import get_primary_metric

        metrics = {
            "accuracy": {"exact_match": 0.75},
            "bits_per_byte": {"bpb_scorer": 0.5},
            "f1": {"f1_scorer": 0.8},
        }
        result = get_primary_metric(metrics, preferred="bits_per_byte:bpb_scorer")

        assert result == ("bits_per_byte:bpb_scorer", 0.5)

    def test_preferred_metric_ignored_when_not_present(self):
        """Test fallback when preferred metric is not in metrics dict."""
        from olmo_eval.runners.utils import get_primary_metric

        metrics = {"accuracy": {"exact_match": 0.75}, "f1": {"f1_scorer": 0.8}}
        result = get_primary_metric(metrics, preferred="bits_per_byte:bpb_scorer")

        # Falls back to accuracy (first scorer alphabetically)
        assert result == ("accuracy:exact_match", 0.75)

    def test_accuracy_fallback_when_no_preferred(self):
        """Test that accuracy is used when no preferred metric specified."""
        from olmo_eval.runners.utils import get_primary_metric

        metrics = {"accuracy": {"exact_match": 0.75}, "bits_per_byte": {"bpb_scorer": 0.5}}
        result = get_primary_metric(metrics)

        assert result == ("accuracy:exact_match", 0.75)

    def test_alphabetical_fallback_when_no_accuracy(self):
        """Test alphabetical fallback when no accuracy and no preferred."""
        from olmo_eval.runners.utils import get_primary_metric

        metrics = {"f1": {"f1_scorer": 0.8}, "bits_per_byte": {"bpb_scorer": 0.5}}
        result = get_primary_metric(metrics)

        # bits_per_byte comes before f1 alphabetically, bpb_scorer is first (only) scorer
        assert result == ("bits_per_byte:bpb_scorer", 0.5)

    def test_empty_metrics_returns_none(self):
        """Test that empty metrics returns None."""
        from olmo_eval.runners.utils import get_primary_metric

        result = get_primary_metric({})
        assert result is None

    def test_preferred_none_same_as_not_specified(self):
        """Test that preferred=None behaves same as not specifying."""
        from olmo_eval.runners.utils import get_primary_metric

        metrics = {"accuracy": {"exact_match": 0.75}, "bits_per_byte": {"bpb_scorer": 0.5}}

        result_none = get_primary_metric(metrics, preferred=None)
        result_default = get_primary_metric(metrics)

        assert result_none == result_default

"""Tests for olmo_eval.runner module."""

import pytest

# Import to ensure tasks and suites are registered
import olmo_eval.evals  # noqa: F401
import olmo_eval.evals.tasks  # noqa: F401
from olmo_eval.runners import EvalRunner
from olmo_eval.runners.sequential import ValidationError


class TestEvalRunnerValidation:
    """Tests for EvalRunner.validate method."""

    def test_validate_valid_task(self):
        """Test validation passes for valid task."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["arc_challenge"],
        )
        # Should not raise
        runner.validate()

    def test_validate_valid_suite(self):
        """Test validation passes for valid suite."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["core:mc"],
        )
        # Should not raise
        runner.validate()

    def test_validate_multiple_valid_specs(self):
        """Test validation passes for multiple valid specs."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["arc_challenge", "arc_easy", "core:mc"],
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
            task_specs=["arc_challenge:nonexistent_regime"],
        )
        with pytest.raises(ValidationError, match="Unknown variant/regime"):
            runner.validate()

    def test_validate_collects_multiple_errors(self):
        """Test validation collects all errors."""
        runner = EvalRunner(
            model_name="llama3.1-8b",
            task_specs=["bad_task1", "bad_task2", "arc_challenge"],
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
            task_specs=["arc_challenge", "nonexistent", "core:mc"],
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

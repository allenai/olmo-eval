"""Tests for olmo_eval.core.metrics module."""

import pytest

from olmo_eval.core.metrics import AccuracyMetric
from olmo_eval.core.scorers import ExactMatchScorer, MultipleChoiceScorer
from olmo_eval.core.types import Instance, LMOutput, LMRequest, RequestType, Response


class TestAccuracyMetric:
    """Tests for AccuracyMetric."""

    def _make_response(self, score: float, scorer_name: str = "exact_match") -> Response:
        """Helper to create a response with a score."""
        return Response(
            instance=Instance(question="Q", gold_answer="A"),
            request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
            outputs=[LMOutput(text="A")],
            scores={scorer_name: score},
        )

    def test_accuracy_all_correct(self):
        """Test accuracy with all correct answers."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(1.0),
            self._make_response(1.0),
            self._make_response(1.0),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 1.0

    def test_accuracy_all_incorrect(self):
        """Test accuracy with all incorrect answers."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(0.0),
            self._make_response(0.0),
            self._make_response(0.0),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 0.0

    def test_accuracy_mixed(self):
        """Test accuracy with mixed results."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(1.0),
            self._make_response(0.0),
            self._make_response(1.0),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == pytest.approx(2 / 3)

    def test_accuracy_empty_responses(self):
        """Test accuracy with empty response list."""
        metric = AccuracyMetric()

        accuracy = metric.compute([])

        assert accuracy == 0.0

    def test_accuracy_single_response(self):
        """Test accuracy with single response."""
        metric = AccuracyMetric()
        responses = [self._make_response(1.0)]

        accuracy = metric.compute(responses)

        assert accuracy == 1.0

    def test_accuracy_custom_scorer(self):
        """Test accuracy with custom scorer type."""
        metric = AccuracyMetric(scorer=MultipleChoiceScorer)
        responses = [
            self._make_response(1.0, "multiple_choice"),
            self._make_response(0.0, "multiple_choice"),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 0.5

    def test_accuracy_missing_scorer(self):
        """Test accuracy when scorer name not in scores dict."""
        metric = AccuracyMetric(scorer=MultipleChoiceScorer)
        responses = [
            self._make_response(1.0, "exact_match"),
            self._make_response(1.0, "exact_match"),
        ]

        accuracy = metric.compute(responses)

        # Missing scorer defaults to 0.0
        assert accuracy == 0.0

    def test_accuracy_name(self):
        """Test metric name."""
        metric = AccuracyMetric()
        assert metric.name == "accuracy"

        custom = AccuracyMetric(name="custom_accuracy")
        assert custom.name == "custom_accuracy"

    def test_accuracy_partial_scores(self):
        """Test accuracy with partial scores (not just 0 or 1)."""
        metric = AccuracyMetric()
        responses = [
            self._make_response(0.5),
            self._make_response(0.5),
        ]

        accuracy = metric.compute(responses)

        assert accuracy == 0.5

    def test_accuracy_default_scorer(self):
        """Test that default scorer is ExactMatchScorer."""
        metric = AccuracyMetric()
        assert metric.scorer == ExactMatchScorer

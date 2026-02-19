"""Tests for olmo_eval.core.metrics module."""

import math

import pytest

from olmo_eval.common.metrics import AccuracyMetric, BPBMetric
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response


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


class TestBPBMetric:
    """Tests for BPBMetric with gold selection."""

    def _make_output_with_logprobs(self, text: str, logprobs: list[float]) -> LMOutput:
        """Helper to create an LMOutput with logprobs.

        Each logprob entry corresponds to one character of the text for simplicity.
        """
        # Create logprob entries with bytes (one char per token for test simplicity)
        entries = []
        for i, lp in enumerate(logprobs):
            char = text[i] if i < len(text) else ""
            entries.append(
                {
                    "token": char,
                    "logprob": lp,
                    "bytes": list(char.encode("utf-8")),
                }
            )
        return LMOutput(text=text, logprobs=entries)

    def _make_response(
        self,
        outputs: list[LMOutput],
        gold_idx: int | None = None,
    ) -> Response:
        """Helper to create a response with multiple outputs."""
        metadata = {"gold_idx": gold_idx} if gold_idx is not None else {}
        return Response(
            instance=Instance(question="Q", gold_answer="A", metadata=metadata),
            request=LMRequest(request_type=RequestType.COMPLETION, prompt="Q"),
            outputs=outputs,
        )

    def test_bpb_single_output(self):
        """Test BPB with single output."""
        metric = BPBMetric()
        # Text "ab" = 2 bytes, logprobs sum to -2.0
        # BPB = -(-2.0) / (2 * log(2)) = 2.0 / 1.386 ≈ 1.443
        output = self._make_output_with_logprobs("ab", [-1.0, -1.0])
        responses = [self._make_response([output])]

        bpb = metric.compute(responses)

        expected = 2.0 / (2 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_multiple_outputs_selects_gold(self):
        """Test that BPB selects the gold/correct output."""
        metric = BPBMetric()

        # Create 3 outputs with different BPB values
        # Output 0: "a" (1 byte), logprob -1.0 -> BPB = 1/(1*log2) ≈ 1.443
        # Output 1: "ab" (2 bytes), logprob -4.0 -> BPB = 4/(2*log2) ≈ 2.885 (gold)
        # Output 2: "abc" (3 bytes), logprob -3.0 -> BPB = 3/(3*log2) ≈ 1.443
        outputs = [
            self._make_output_with_logprobs("a", [-1.0]),
            self._make_output_with_logprobs("ab", [-2.0, -2.0]),
            self._make_output_with_logprobs("abc", [-1.0, -1.0, -1.0]),
        ]
        responses = [self._make_response(outputs, gold_idx=1)]

        bpb = metric.compute(responses)

        # Should use output 1 (gold_idx=1): 4.0 / (2 * log(2))
        expected = 4.0 / (2 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_multiple_outputs_without_gold_idx_uses_first(self):
        """Test that BPB falls back to first output without gold_idx."""
        metric = BPBMetric()

        outputs = [
            self._make_output_with_logprobs("a", [-1.0]),
            self._make_output_with_logprobs("ab", [-2.0, -2.0]),
        ]
        # No gold_idx specified
        responses = [self._make_response(outputs, gold_idx=None)]

        bpb = metric.compute(responses)

        # Should use output 0 (first): 1.0 / (1 * log(2))
        expected = 1.0 / (1 * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_empty_responses(self):
        """Test BPB with empty response list."""
        metric = BPBMetric()

        bpb = metric.compute([])

        assert bpb == 0.0

    def test_bpb_aggregates_across_responses(self):
        """Test that BPB uses byte-weighted aggregation across responses."""
        metric = BPBMetric()

        # Response 1: "a" (1 byte), logprob -1.0
        # Response 2: "ab" (2 bytes), logprob -4.0 (sum of -2.0, -2.0)
        # Byte-weighted: total_logprobs = 5.0, total_bytes = 3
        # BPB = 5.0 / (3 * log(2))
        responses = [
            self._make_response([self._make_output_with_logprobs("a", [-1.0])]),
            self._make_response([self._make_output_with_logprobs("ab", [-2.0, -2.0])]),
        ]

        bpb = metric.compute(responses)

        total_logprobs = 1.0 + 4.0  # negated from -1.0 and -4.0
        total_bytes = 1 + 2
        expected = total_logprobs / (total_bytes * math.log(2))
        assert bpb == pytest.approx(expected)

    def test_bpb_no_logprobs_returns_zero(self):
        """Test that output without logprobs returns 0."""
        metric = BPBMetric()

        output = LMOutput(text="test", logprobs=None)
        responses = [self._make_response([output])]

        bpb = metric.compute(responses)

        assert bpb == 0.0

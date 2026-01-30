"""Tests for pass@k and pass^k reliability metrics."""

from olmo_eval.core.metrics import PassAtKMetric, PassPowKMetric
from olmo_eval.core.scorers import CodeExecutionScorer
from olmo_eval.core.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.core.utils import compute_pass_at_k, compute_pass_pow_k


def make_response(task_id: str, score: float, scorer_name: str = "code_execution") -> Response:
    """Helper to create Response with given score."""
    return Response(
        instance=Instance(question="test", metadata={"id": task_id}),
        request=LMRequest(request_type=RequestType.CHAT),
        outputs=[LMOutput(text="")],
        scores={scorer_name: score},
    )


class TestComputePassAtK:
    """Tests for compute_pass_at_k utility function."""

    def test_all_correct(self):
        """Test when all samples are correct."""
        result = compute_pass_at_k(n=5, c=5, k=1)
        assert result == 1.0

    def test_none_correct(self):
        """Test when no samples are correct."""
        result = compute_pass_at_k(n=5, c=0, k=1)
        assert result == 0.0

    def test_one_of_many_correct(self):
        """Test when one of many samples is correct."""
        result = compute_pass_at_k(n=10, c=1, k=1)
        assert 0.0 < result < 1.0

    def test_k_equals_n(self):
        """Test when k equals n."""
        result = compute_pass_at_k(n=5, c=3, k=5)
        assert result == 1.0  # At least one correct in 5 draws from 3 correct

    def test_k_greater_than_n_minus_c(self):
        """Test when k > n - c (guaranteed success)."""
        result = compute_pass_at_k(n=5, c=4, k=2)
        assert result == 1.0

    def test_pass_at_1_with_half_correct(self):
        """Test pass@1 with 50% success rate."""
        result = compute_pass_at_k(n=10, c=5, k=1)
        assert result == 0.5


class TestComputePassPowK:
    """Tests for compute_pass_pow_k utility function."""

    def test_all_correct(self):
        """Test when all samples are correct."""
        result = compute_pass_pow_k(n=5, c=5, k=1)
        assert result == 1.0

    def test_none_correct(self):
        """Test when no samples are correct."""
        result = compute_pass_pow_k(n=5, c=0, k=1)
        assert result == 0.0

    def test_half_correct_k1(self):
        """Test 50% success rate with k=1."""
        result = compute_pass_pow_k(n=10, c=5, k=1)
        assert result == 0.5

    def test_half_correct_k2(self):
        """Test 50% success rate with k=2."""
        result = compute_pass_pow_k(n=10, c=5, k=2)
        assert result == 0.25  # 0.5^2

    def test_half_correct_k3(self):
        """Test 50% success rate with k=3."""
        result = compute_pass_pow_k(n=10, c=5, k=3)
        assert result == 0.125  # 0.5^3

    def test_zero_samples(self):
        """Test with zero samples."""
        result = compute_pass_pow_k(n=0, c=0, k=1)
        assert result == 0.0


class TestPassAtKMetric:
    """Tests for PassAtKMetric class."""

    def test_all_correct(self):
        """Test when all responses are correct."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=1)
        responses = [
            make_response("task1", 1.0),
            make_response("task2", 1.0),
        ]

        result = metric.compute(responses)
        assert result == 1.0

    def test_all_incorrect(self):
        """Test when all responses are incorrect."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=1)
        responses = [
            make_response("task1", 0.0),
            make_response("task2", 0.0),
        ]

        result = metric.compute(responses)
        assert result == 0.0

    def test_mixed_results(self):
        """Test with mixed results."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=1)
        responses = [
            make_response("task1", 1.0),
            make_response("task2", 0.0),
        ]

        result = metric.compute(responses)
        assert result == 0.5

    def test_multiple_samples_per_task(self):
        """Test with multiple samples per task."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=1)
        # Same task, multiple samples
        responses = [
            make_response("task1", 0.0),  # Sample 1 fails
            make_response("task1", 1.0),  # Sample 2 succeeds
            make_response("task1", 0.0),  # Sample 3 fails
        ]

        result = metric.compute(responses)
        # 1 of 3 correct, pass@1 should be 1/3
        assert 0.3 < result < 0.4

    def test_k_greater_than_1(self):
        """Test with k > 1."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=2)
        responses = [
            make_response("task1", 0.0),
            make_response("task1", 1.0),
            make_response("task1", 1.0),
        ]

        result = metric.compute(responses)
        assert result == 1.0  # 2 of 3 correct, pass@2 should be 1.0

    def test_empty_responses(self):
        """Test with empty responses."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=1)
        result = metric.compute([])
        assert result == 0.0

    def test_to_dict(self):
        """Test serialization."""
        metric = PassAtKMetric(scorer=CodeExecutionScorer, k=5)
        d = metric.to_dict()
        assert d["type"] == "PassAtKMetric"
        assert d["name"] == "pass_at_k"


class TestPassPowKMetric:
    """Tests for PassPowKMetric class."""

    def test_all_correct(self):
        """Test when all responses are correct."""
        metric = PassPowKMetric(scorer=CodeExecutionScorer, k=1)
        responses = [
            make_response("task1", 1.0),
            make_response("task2", 1.0),
        ]

        result = metric.compute(responses)
        assert result == 1.0

    def test_all_incorrect(self):
        """Test when all responses are incorrect."""
        metric = PassPowKMetric(scorer=CodeExecutionScorer, k=1)
        responses = [
            make_response("task1", 0.0),
            make_response("task2", 0.0),
        ]

        result = metric.compute(responses)
        assert result == 0.0

    def test_k_exponential_decay(self):
        """Test exponential decay with k."""
        # 50% success rate
        responses = [
            make_response("task1", 1.0),
            make_response("task1", 0.0),
        ]

        k1_metric = PassPowKMetric(scorer=CodeExecutionScorer, k=1)
        k2_metric = PassPowKMetric(scorer=CodeExecutionScorer, k=2)
        k3_metric = PassPowKMetric(scorer=CodeExecutionScorer, k=3)

        k1_result = k1_metric.compute(responses)
        k2_result = k2_metric.compute(responses)
        k3_result = k3_metric.compute(responses)

        assert k1_result == 0.5
        assert k2_result == 0.25
        assert k3_result == 0.125

    def test_empty_responses(self):
        """Test with empty responses."""
        metric = PassPowKMetric(scorer=CodeExecutionScorer, k=1)
        result = metric.compute([])
        assert result == 0.0

    def test_to_dict(self):
        """Test serialization."""
        metric = PassPowKMetric(scorer=CodeExecutionScorer, k=3)
        d = metric.to_dict()
        assert d["type"] == "PassPowKMetric"
        assert d["name"] == "pass_pow_k"


class TestMetricsComparison:
    """Tests comparing pass@k and pass^k metrics."""

    def test_pass_at_k_greater_than_pow_k(self):
        """Test that pass@k >= pass^k always."""
        responses = [
            make_response("task1", 1.0),
            make_response("task1", 0.0),
            make_response("task1", 1.0),
            make_response("task1", 0.0),
        ]

        for k in [1, 2, 3, 4]:
            at_k = PassAtKMetric(scorer=CodeExecutionScorer, k=k).compute(responses)
            pow_k = PassPowKMetric(scorer=CodeExecutionScorer, k=k).compute(responses)
            assert at_k >= pow_k

    def test_both_equal_at_perfect_score(self):
        """Test both metrics equal when all correct."""
        responses = [
            make_response("task1", 1.0),
            make_response("task1", 1.0),
            make_response("task1", 1.0),
        ]

        for k in [1, 2, 3]:
            at_k = PassAtKMetric(scorer=CodeExecutionScorer, k=k).compute(responses)
            pow_k = PassPowKMetric(scorer=CodeExecutionScorer, k=k).compute(responses)
            assert at_k == 1.0
            assert pow_k == 1.0

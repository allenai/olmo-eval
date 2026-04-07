"""Tests for pass@k and pass^k reliability metrics."""

from collections.abc import Iterator

from olmo_eval.common.metrics import PassAtKMetric, PassPowKMetric
from olmo_eval.common.scorers import CodeExecutionScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.common.utils import compute_pass_at_k, compute_pass_pow_k
from olmo_eval.evals.tasks.common import Task, TaskConfig


class MinimalTask(Task):
    """Minimal task implementation for testing."""

    @property
    def instances(self) -> Iterator[Instance]:
        yield Instance(question="test")

    def format_request(self, instance: Instance) -> LMRequest:
        return LMRequest(request_type=RequestType.CHAT)


def make_response(task_id: str, score: float, scorer_name: str = "code_exec") -> Response:
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

    def test_n_less_than_k_no_correct(self):
        """Test when n < k with no correct samples."""
        # n=3, c=0, k=5: effectively pass@3 with 0 correct = 0.0
        result = compute_pass_at_k(n=3, c=0, k=5)
        assert result == 0.0

    def test_n_less_than_k_some_correct(self):
        """Test when n < k with some correct samples."""
        # n=3, c=1, k=5: effectively pass@3 with 1 correct
        # Drawing all 3 samples guarantees getting the 1 correct one
        result = compute_pass_at_k(n=3, c=1, k=5)
        assert result == 1.0

    def test_n_less_than_k_all_correct(self):
        """Test when n < k with all correct samples."""
        # n=3, c=3, k=5: effectively pass@3 with 3 correct = 1.0
        result = compute_pass_at_k(n=3, c=3, k=5)
        assert result == 1.0

    def test_zero_samples(self):
        """Test with zero samples."""
        result = compute_pass_at_k(n=0, c=0, k=1)
        assert result == 0.0


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
        assert d["name"] == "pass_at_5"


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
        assert d["name"] == "pass_pow_3"


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


class TestExpandMultiOutputResponses:
    """Tests for _expand_multi_output_responses method."""

    def test_single_output_unchanged(self):
        """Test that single-output responses are unchanged."""
        task = MinimalTask(TaskConfig(name="test"))
        responses = [
            Response(
                instance=Instance(question="q1", metadata={"id": "task1"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=[LMOutput(text="output1")],
            )
        ]

        expanded = task._expand_multi_output_responses(responses)

        assert len(expanded) == 1
        assert expanded[0] is responses[0]

    def test_multi_output_expanded(self):
        """Test that multi-output responses are expanded into separate responses."""
        task = MinimalTask(TaskConfig(name="test"))
        outputs = [
            LMOutput(text="output1", metadata={"score:code_exec": 1.0}),
            LMOutput(text="output2", metadata={"score:code_exec": 0.0}),
            LMOutput(text="output3", metadata={"score:code_exec": 1.0}),
        ]
        responses = [
            Response(
                instance=Instance(question="q1", metadata={"id": "task1"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=outputs,
            )
        ]

        expanded = task._expand_multi_output_responses(responses)

        assert len(expanded) == 3
        # Each expanded response should have one output
        for i, resp in enumerate(expanded):
            assert len(resp.outputs) == 1
            assert resp.outputs[0] is outputs[i]
            assert resp.instance is responses[0].instance
            assert resp.request is responses[0].request

    def test_scores_copied_from_metadata(self):
        """Test that scores are copied from output metadata to response scores."""
        task = MinimalTask(TaskConfig(name="test"))
        outputs = [
            LMOutput(text="output1", metadata={"score:code_exec": 1.0}),
            LMOutput(text="output2", metadata={"score:code_exec": 0.0}),
        ]
        responses = [
            Response(
                instance=Instance(question="q1", metadata={"id": "task1"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=outputs,
            )
        ]

        expanded = task._expand_multi_output_responses(responses)

        assert expanded[0].scores["code_exec"] == 1.0
        assert expanded[1].scores["code_exec"] == 0.0

    def test_multiple_scorer_scores(self):
        """Test that multiple scorer scores are all copied."""
        task = MinimalTask(TaskConfig(name="test"))
        outputs = [
            LMOutput(
                text="output1",
                metadata={"score:code_exec": 1.0, "score:exact_match": 0.5},
            ),
            LMOutput(
                text="output2",
                metadata={"score:code_exec": 0.0, "score:exact_match": 1.0},
            ),
        ]
        responses = [
            Response(
                instance=Instance(question="q1", metadata={"id": "task1"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=outputs,
            )
        ]

        expanded = task._expand_multi_output_responses(responses)

        assert expanded[0].scores["code_exec"] == 1.0
        assert expanded[0].scores["exact_match"] == 0.5
        assert expanded[1].scores["code_exec"] == 0.0
        assert expanded[1].scores["exact_match"] == 1.0

    def test_empty_metadata_handled(self):
        """Test that outputs with no metadata are handled gracefully."""
        task = MinimalTask(TaskConfig(name="test"))
        outputs = [
            LMOutput(text="output1"),  # No metadata
            LMOutput(text="output2", metadata={}),  # Empty metadata
        ]
        responses = [
            Response(
                instance=Instance(question="q1", metadata={"id": "task1"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=outputs,
            )
        ]

        expanded = task._expand_multi_output_responses(responses)

        assert len(expanded) == 2
        assert expanded[0].scores == {}
        assert expanded[1].scores == {}

    def test_mixed_single_and_multi_output(self):
        """Test handling of mixed single and multi-output responses."""
        task = MinimalTask(TaskConfig(name="test"))
        responses = [
            Response(
                instance=Instance(question="q1", metadata={"id": "task1"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=[LMOutput(text="single", metadata={"score:code_exec": 1.0})],
            ),
            Response(
                instance=Instance(question="q2", metadata={"id": "task2"}),
                request=LMRequest(request_type=RequestType.CHAT),
                outputs=[
                    LMOutput(text="multi1", metadata={"score:code_exec": 0.0}),
                    LMOutput(text="multi2", metadata={"score:code_exec": 1.0}),
                ],
            ),
        ]

        expanded = task._expand_multi_output_responses(responses)

        assert len(expanded) == 3
        # First response unchanged
        assert expanded[0] is responses[0]
        # Second response expanded into two
        assert expanded[1].outputs[0].text == "multi1"
        assert expanded[2].outputs[0].text == "multi2"

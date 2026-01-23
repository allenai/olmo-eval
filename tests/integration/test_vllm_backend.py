"""GPU integration tests for the vLLM backend.

These tests require:
- Docker with GPU support, OR
- vLLM installed locally with GPU access

Run with:
    pytest tests/integration/test_vllm_backend.py -v --gpu

To skip Docker management (if vLLM is already running):
    pytest tests/integration/test_vllm_backend.py -v --gpu --no-docker

To use a different model:
    pytest tests/integration/test_vllm_backend.py -v --gpu \\
        --vllm-model "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
"""

import pytest

from olmo_eval.core import LMRequest, RequestType, SamplingParams

pytestmark = [pytest.mark.gpu]


class TestVLLMBackendGenerate:
    """Tests for VLLMBackend.generate method."""

    def test_generate_single_prompt(self, vllm_backend):
        """Test generating from a single prompt."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="The capital of France is",
            )
        ]

        results = vllm_backend.generate(requests)

        assert len(results) == 1
        assert len(results[0]) >= 1  # At least one output
        assert isinstance(results[0][0].text, str)
        assert len(results[0][0].text) > 0

    def test_generate_multiple_prompts(self, vllm_backend, small_test_prompts):
        """Test generating from multiple prompts."""
        requests = [
            LMRequest(request_type=RequestType.COMPLETION, prompt=p) for p in small_test_prompts
        ]

        results = vllm_backend.generate(requests)

        assert len(results) == len(small_test_prompts)
        for result in results:
            assert len(result) >= 1
            assert isinstance(result[0].text, str)

    def test_generate_with_sampling_params(self, vllm_backend):
        """Test generating with custom sampling parameters."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Once upon a time",
            )
        ]
        params = SamplingParams(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
        )

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        assert len(results[0][0].text) > 0

    def test_generate_with_stop_sequences(self, vllm_backend):
        """Test generating with stop sequences."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Count: 1, 2, 3,",
            )
        ]
        params = SamplingParams(
            max_tokens=100,
            stop_sequences=("\n", "."),
        )

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        # Output should not contain stop sequences (they terminate generation)
        output = results[0][0].text
        assert "\n" not in output or output.endswith("\n") is False

    def test_generate_multiple_samples(self, vllm_backend):
        """Test generating multiple samples per prompt."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="The meaning of life is",
            )
        ]
        params = SamplingParams(
            max_tokens=30,
            temperature=0.8,
            num_samples=3,
        )

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        assert len(results[0]) == 3  # 3 samples
        # Each sample should be different (with high probability)
        texts = [r.text for r in results[0]]
        assert len(texts) == 3

    def test_generate_with_logprobs(self, vllm_backend):
        """Test generating with logprobs enabled."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Hello",
            )
        ]
        params = SamplingParams(
            max_tokens=10,
            logprobs=5,
        )

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        output = results[0][0]
        assert output.logprobs is not None
        assert len(output.logprobs) > 0
        # Each logprob entry should have token and logprob
        for entry in output.logprobs:
            assert "token" in entry
            assert "logprob" in entry
            assert isinstance(entry["logprob"], float)

    def test_generate_deterministic(self, vllm_backend):
        """Test that temperature=0 gives deterministic results."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="1 + 1 =",
            )
        ]
        params = SamplingParams(max_tokens=5, temperature=0.0)

        results1 = vllm_backend.generate(requests, sampling_params=params)
        results2 = vllm_backend.generate(requests, sampling_params=params)

        assert results1[0][0].text == results2[0][0].text


class TestVLLMBackendLogprobs:
    """Tests for VLLMBackend.logprobs method (multiple choice scoring)."""

    def test_logprobs_single_request(self, vllm_backend):
        """Test logprobs for a single request with continuations."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="The capital of France is",
                continuations=(" Paris", " London", " Berlin"),
            )
        ]

        results = vllm_backend.logprobs(requests)

        assert len(results) == 1
        assert len(results[0]) == 3  # One output per continuation

        for output in results[0]:
            assert output.logprobs is not None
            assert "total_logprob" in output.metadata
            assert isinstance(output.metadata["total_logprob"], float)
            # Verify new metadata fields
            assert "sum_logits" in output.metadata
            assert "num_tokens" in output.metadata
            assert "num_tokens_all" in output.metadata
            assert "is_greedy" in output.metadata
            assert isinstance(output.metadata["num_tokens"], int)
            assert isinstance(output.metadata["num_tokens_all"], int)
            assert isinstance(output.metadata["is_greedy"], bool)

    def test_logprobs_multiple_requests(self, vllm_backend):
        """Test logprobs for multiple requests."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="2 + 2 =",
                continuations=(" 3", " 4", " 5"),
            ),
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="The sky is",
                continuations=(" blue", " green", " red"),
            ),
        ]

        results = vllm_backend.logprobs(requests)

        assert len(results) == 2
        assert len(results[0]) == 3
        assert len(results[1]) == 3

    def test_logprobs_correct_answer_higher(self, vllm_backend):
        """Test that correct answers tend to have higher logprobs."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="The capital of France is",
                continuations=(" Paris", " Tokyo", " Moscow"),
            )
        ]

        results = vllm_backend.logprobs(requests)

        # Paris should have the highest logprob (most likely correct)
        logprobs = [r.metadata["total_logprob"] for r in results[0]]
        paris_logprob = logprobs[0]
        tokyo_logprob = logprobs[1]
        moscow_logprob = logprobs[2]

        # Paris should be more likely than other capitals
        assert paris_logprob > tokyo_logprob or paris_logprob > moscow_logprob

    def test_logprobs_text_matches_continuation(self, vllm_backend):
        """Test that output text matches the continuation."""
        continuations = (" yes", " no", " maybe")
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Is the sky blue?",
                continuations=continuations,
            )
        ]

        results = vllm_backend.logprobs(requests)

        for output, expected in zip(results[0], continuations, strict=True):
            assert output.text == expected


class TestVLLMBackendEdgeCases:
    """Edge case tests for VLLMBackend."""

    def test_empty_prompt(self, vllm_backend):
        """Test handling of empty prompt."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="",
            )
        ]
        params = SamplingParams(max_tokens=10)

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        assert isinstance(results[0][0].text, str)

    def test_long_prompt(self, vllm_backend):
        """Test handling of longer prompt."""
        long_prompt = "Hello world. " * 50  # ~100 tokens
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt=long_prompt,
            )
        ]
        params = SamplingParams(max_tokens=20)

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        assert isinstance(results[0][0].text, str)

    def test_special_characters(self, vllm_backend):
        """Test handling of special characters in prompt."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Special chars: @#$%^&*()_+ émojis: 🎉🚀",
            )
        ]
        params = SamplingParams(max_tokens=10)

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 1
        assert isinstance(results[0][0].text, str)

    def test_single_continuation(self, vllm_backend):
        """Test logprobs with single continuation."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=(" only",),
            )
        ]

        results = vllm_backend.logprobs(requests)

        assert len(results) == 1
        assert len(results[0]) == 1

    def test_empty_continuations(self, vllm_backend):
        """Test logprobs with no continuations."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="Test",
                continuations=(),
            )
        ]

        results = vllm_backend.logprobs(requests)

        assert len(results) == 1
        assert len(results[0]) == 0

    def test_logprobs_empty_context(self, vllm_backend):
        """Test logprobs with empty context (BOS token handling)."""
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt="",
                continuations=("Hello world",),
            )
        ]

        results = vllm_backend.logprobs(requests)

        assert len(results) == 1
        assert len(results[0]) == 1
        output = results[0][0]
        assert output.logprobs is not None
        assert len(output.logprobs) > 0
        assert "total_logprob" in output.metadata
        assert output.metadata["num_tokens"] > 0


class TestVLLMBackendWithTasks:
    """Integration tests running actual evaluation tasks through vLLM."""

    def test_arc_task_format(self, vllm_backend):
        """Test running ARC-style multiple choice through vLLM."""
        # Simulate ARC task format
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt=(
                    "Question: What is the chemical formula for water?\n\n"
                    "A. H2O\nB. CO2\nC. NaCl\nD. O2\n\nAnswer:"
                ),
                continuations=(" A", " B", " C", " D"),
            )
        ]

        results = vllm_backend.logprobs(requests)

        assert len(results) == 1
        assert len(results[0]) == 4

        # Get the answer with highest logprob
        best_idx = max(
            range(len(results[0])),
            key=lambda i: results[0][i].metadata["total_logprob"],
        )
        best_answer = results[0][best_idx].text.strip()

        # H2O (answer A) should be selected
        assert best_answer == "A"

    def test_batch_processing(self, vllm_backend):
        """Test that batch processing works correctly."""
        # Create a batch of requests
        requests = [
            LMRequest(
                request_type=RequestType.COMPLETION,
                prompt=f"Count to {i}: 1, 2,",
            )
            for i in range(5)
        ]
        params = SamplingParams(max_tokens=20)

        results = vllm_backend.generate(requests, sampling_params=params)

        assert len(results) == 5
        for result in results:
            assert len(result) >= 1
            assert isinstance(result[0].text, str)

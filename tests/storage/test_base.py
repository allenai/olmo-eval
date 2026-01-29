"""Tests for storage base classes and data models."""

from datetime import datetime

import pytest

from olmo_eval.core.types import EvalResult, StoredTaskResult, compute_model_hash
from olmo_eval.storage.base import convert_runner_results


class TestStoredTaskResult:
    """Tests for StoredTaskResult dataclass."""

    def test_create_minimal(self):
        """Test creating with only required fields."""
        result = StoredTaskResult(
            task_name="mmlu",
            metrics={"accuracy": 0.75},
            task_hash="mmlu-hash-001",
        )
        assert result.task_name == "mmlu"
        assert result.metrics == {"accuracy": 0.75}
        assert result.task_hash == "mmlu-hash-001"
        assert result.num_instances is None

    def test_create_full(self):
        """Test creating with all fields."""
        result = StoredTaskResult(
            task_name="mmlu",
            metrics={"accuracy": 0.75, "f1": 0.72},
            task_hash="abc123",
            num_instances=1000,
            primary_metric="accuracy",
            primary_score=0.75,
            s3_metrics_key="s3://bucket/metrics.json",
            s3_predictions_key="s3://bucket/predictions.jsonl",
        )
        assert result.task_name == "mmlu"
        assert result.task_hash == "abc123"
        assert result.num_instances == 1000
        assert result.primary_metric == "accuracy"
        assert result.primary_score == 0.75
        assert result.s3_metrics_key == "s3://bucket/metrics.json"
        assert result.s3_predictions_key == "s3://bucket/predictions.jsonl"

    def test_equality(self):
        """Test dataclass equality."""
        r1 = StoredTaskResult(task_name="mmlu", metrics={"accuracy": 0.75}, task_hash="hash1")
        r2 = StoredTaskResult(task_name="mmlu", metrics={"accuracy": 0.75}, task_hash="hash1")
        r3 = StoredTaskResult(task_name="gsm8k", metrics={"accuracy": 0.75}, task_hash="hash2")
        assert r1 == r2
        assert r1 != r3


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample task results."""
        return [
            StoredTaskResult(task_name="mmlu", metrics={"accuracy": 0.65}, task_hash="mmlu-hash"),
            StoredTaskResult(
                task_name="gsm8k", metrics={"exact_match": 0.58}, task_hash="gsm8k-hash"
            ),
        ]

    @pytest.fixture
    def sample_timestamp(self):
        """Create a sample timestamp."""
        return datetime(2024, 1, 15, 10, 30, 0)

    def test_create_minimal(self, sample_tasks, sample_timestamp):
        """Test creating with only required fields."""
        result = EvalResult(
            experiment_id="abc123",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
        )
        assert result.experiment_id == "abc123"
        assert result.model_name == "llama3.1-8b"
        assert result.backend_name == "vllm"
        assert result.timestamp == sample_timestamp
        assert len(result.tasks) == 2
        assert result.model_config is None
        assert result.metadata is None

    def test_create_full(self, sample_tasks, sample_timestamp):
        """Test creating with all fields."""
        result = EvalResult(
            experiment_id="def456",
            model_name="olmo-2-7b",
            backend_name="hf",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
            experiment_name="benchmark-run-1",
            workspace="ai2/olmo",
            author="test-user",
            tags=["benchmark", "release"],
            git_ref="abc123def",
            model_hash="model-hash-123",
            revision="main",
            s3_location="s3://bucket/results/run-1/",
            model_config={"batch_size": 32},
            metadata={"git_sha": "abc123"},
        )
        assert result.model_config == {"batch_size": 32}
        assert result.metadata == {"git_sha": "abc123"}
        assert result.experiment_name == "benchmark-run-1"
        assert result.model_hash == "model-hash-123"

    def test_model_hash_auto_computed(self, sample_tasks, sample_timestamp):
        """Test that model_hash is auto-computed from config."""
        config = {"model": "llama", "temperature": 0.7}
        result = EvalResult(
            experiment_id="test",
            model_name="test-model",
            backend_name="vllm",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
            model_config=config,
        )
        assert result.model_hash is not None
        assert result.model_hash == compute_model_hash(config)

    def test_model_hash_not_overwritten(self, sample_tasks, sample_timestamp):
        """Test that explicit model_hash is not overwritten."""
        result = EvalResult(
            experiment_id="test",
            model_name="test-model",
            backend_name="vllm",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
            model_hash="explicit-hash",
            model_config={"model": "llama"},
        )
        assert result.model_hash == "explicit-hash"

    def test_empty_tasks(self, sample_timestamp):
        """Test with empty tasks list."""
        result = EvalResult(
            experiment_id="empty-tasks",
            model_name="test",
            backend_name="mock",
            timestamp=sample_timestamp,
            tasks=[],
        )
        assert result.tasks == []


class TestComputeModelHash:
    """Tests for compute_model_hash function."""

    def test_deterministic(self):
        """Test that same config produces same hash."""
        config = {"model": "llama", "temperature": 0.7}
        hash1 = compute_model_hash(config)
        hash2 = compute_model_hash(config)
        assert hash1 == hash2

    def test_different_configs(self):
        """Test that different configs produce different hashes."""
        config1 = {"model": "llama", "temperature": 0.7}
        config2 = {"model": "llama", "temperature": 0.8}
        assert compute_model_hash(config1) != compute_model_hash(config2)

    def test_none_config(self):
        """Test that None config returns None."""
        assert compute_model_hash(None) is None

    def test_empty_config(self):
        """Test that empty config returns None."""
        assert compute_model_hash({}) is None

    def test_hash_length(self):
        """Test that hash is 16 characters."""
        config = {"model": "test"}
        h = compute_model_hash(config)
        assert len(h) == 16


class TestConvertRunnerResults:
    """Tests for convert_runner_results function."""

    def test_converts_provider_field(self):
        """Test that 'provider' field from runner results maps to backend_name.

        This catches regressions where the wrong key is used (e.g., 'backend' vs 'provider').
        """
        results = {
            "model": "llama3.1-8b",
            "provider": "vllm",
            "timestamp": "2024-01-15T10:30:00",
            "tasks": {
                "mmlu": {
                    "metrics": {"accuracy": 0.75},
                    "task_hash": "mmlu-hash-001",
                }
            },
        }

        eval_result = convert_runner_results(results, experiment_id="test-123")

        assert eval_result.model_name == "llama3.1-8b"
        assert eval_result.backend_name == "vllm"
        assert eval_result.experiment_id == "test-123"

    def test_missing_provider_raises_key_error(self):
        """Test that missing 'provider' field raises KeyError."""
        results = {
            "model": "llama3.1-8b",
            # "provider" is missing - should fail
            "timestamp": "2024-01-15T10:30:00",
            "tasks": {},
        }

        with pytest.raises(KeyError, match="provider"):
            convert_runner_results(results, experiment_id="test-123")

    def test_converts_all_required_fields(self):
        """Test that all required fields from runner results are converted."""
        results = {
            "model": "olmo-2-7b",
            "provider": "hf",
            "timestamp": "2024-06-20T14:00:00",
            "tasks": {
                "gsm8k": {
                    "metrics": {"exact_match": 0.58},
                    "task_hash": "gsm8k-hash",
                    "num_instances": 1000,
                    "primary_metric": "exact_match",
                },
            },
            "model_config": {"temperature": 0.0},
            "metadata": {"run_id": "abc"},
        }

        eval_result = convert_runner_results(
            results,
            experiment_id="exp-456",
            experiment_name="test-run",
            workspace="ai2/test",
            author="tester",
        )

        assert eval_result.model_name == "olmo-2-7b"
        assert eval_result.backend_name == "hf"
        assert eval_result.timestamp == datetime(2024, 6, 20, 14, 0, 0)
        assert eval_result.experiment_name == "test-run"
        assert eval_result.workspace == "ai2/test"
        assert eval_result.author == "tester"
        assert eval_result.model_config == {"temperature": 0.0}
        assert eval_result.metadata == {"run_id": "abc"}
        assert len(eval_result.tasks) == 1
        assert eval_result.tasks[0].task_name == "gsm8k"
        assert eval_result.tasks[0].metrics == {"exact_match": 0.58}

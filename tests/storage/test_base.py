"""Tests for storage base classes and data models."""

from datetime import datetime

import pytest

from olmo_eval.storage.base import EvalResult, TaskResult


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_to_dict_minimal(self):
        """Test to_dict with only required fields."""
        result = TaskResult(
            task_name="mmlu",
            metrics={"accuracy": 0.75},
        )
        d = result.to_dict()
        assert d == {
            "task_name": "mmlu",
            "metrics": {"accuracy": 0.75},
        }

    def test_to_dict_full(self):
        """Test to_dict with all fields."""
        result = TaskResult(
            task_name="mmlu",
            metrics={"accuracy": 0.75, "f1": 0.72},
            num_instances=1000,
            task_hash="abc123",
            primary_metric="accuracy",
            primary_score=0.75,
            s3_metrics_key="s3://bucket/metrics.json",
            s3_predictions_key="s3://bucket/predictions.jsonl",
        )
        d = result.to_dict()
        assert d == {
            "task_name": "mmlu",
            "metrics": {"accuracy": 0.75, "f1": 0.72},
            "num_instances": 1000,
            "task_hash": "abc123",
            "primary_metric": "accuracy",
            "primary_score": 0.75,
            "s3_metrics_key": "s3://bucket/metrics.json",
            "s3_predictions_key": "s3://bucket/predictions.jsonl",
        }

    def test_from_dict_minimal(self):
        """Test from_dict with only required fields."""
        data = {
            "task_name": "gsm8k",
            "metrics": {"exact_match": 0.58},
        }
        result = TaskResult.from_dict(data)
        assert result.task_name == "gsm8k"
        assert result.metrics == {"exact_match": 0.58}
        assert result.num_instances is None
        assert result.task_hash is None
        assert result.primary_metric is None
        assert result.primary_score is None

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "task_name": "arc_challenge",
            "metrics": {"accuracy": 0.52},
            "num_instances": 500,
            "task_hash": "def456",
            "primary_metric": "accuracy",
            "primary_score": 0.52,
            "s3_metrics_key": "s3://bucket/arc-metrics.json",
            "s3_predictions_key": "s3://bucket/arc-predictions.jsonl",
        }
        result = TaskResult.from_dict(data)
        assert result.task_name == "arc_challenge"
        assert result.metrics == {"accuracy": 0.52}
        assert result.num_instances == 500
        assert result.task_hash == "def456"
        assert result.primary_metric == "accuracy"
        assert result.primary_score == 0.52
        assert result.s3_metrics_key == "s3://bucket/arc-metrics.json"
        assert result.s3_predictions_key == "s3://bucket/arc-predictions.jsonl"

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = TaskResult(
            task_name="hellaswag",
            metrics={"accuracy": 0.79},
            num_instances=10042,
            task_hash="xyz789",
            primary_metric="accuracy",
            primary_score=0.79,
        )
        restored = TaskResult.from_dict(original.to_dict())
        assert restored == original


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    @pytest.fixture
    def sample_tasks(self):
        """Create sample task results."""
        return [
            TaskResult(task_name="mmlu", metrics={"accuracy": 0.65}),
            TaskResult(task_name="gsm8k", metrics={"exact_match": 0.58}),
        ]

    @pytest.fixture
    def sample_timestamp(self):
        """Create a sample timestamp."""
        return datetime(2024, 1, 15, 10, 30, 0)

    def test_to_dict_minimal(self, sample_tasks, sample_timestamp):
        """Test to_dict with only required fields."""
        result = EvalResult(
            run_id="abc123",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
        )
        d = result.to_dict()
        assert d["run_id"] == "abc123"
        assert d["model_name"] == "llama3.1-8b"
        assert d["backend_name"] == "vllm"
        assert d["timestamp"] == "2024-01-15T10:30:00"
        assert len(d["tasks"]) == 2
        assert "config" not in d
        assert "metadata" not in d
        assert "experiment_name" not in d
        assert "s3_location" not in d

    def test_to_dict_full(self, sample_tasks, sample_timestamp):
        """Test to_dict with all fields."""
        result = EvalResult(
            run_id="def456",
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
            config={"batch_size": 32},
            metadata={"git_sha": "abc123"},
        )
        d = result.to_dict()
        assert d["config"] == {"batch_size": 32}
        assert d["metadata"] == {"git_sha": "abc123"}
        assert d["experiment_name"] == "benchmark-run-1"
        assert d["workspace"] == "ai2/olmo"
        assert d["author"] == "test-user"
        assert d["tags"] == ["benchmark", "release"]
        assert d["git_ref"] == "abc123def"
        assert d["model_hash"] == "model-hash-123"
        assert d["revision"] == "main"
        assert d["s3_location"] == "s3://bucket/results/run-1/"

    def test_from_dict_minimal(self, sample_timestamp):
        """Test from_dict with only required fields."""
        data = {
            "run_id": "xyz789",
            "model_name": "gpt-4",
            "backend_name": "litellm",
            "timestamp": "2024-01-15T10:30:00",
            "tasks": [
                {"task_name": "arc_easy", "metrics": {"accuracy": 0.85}},
            ],
        }
        result = EvalResult.from_dict(data)
        assert result.run_id == "xyz789"
        assert result.model_name == "gpt-4"
        assert result.backend_name == "litellm"
        assert result.timestamp == sample_timestamp
        assert len(result.tasks) == 1
        assert result.config is None
        assert result.metadata is None
        assert result.experiment_name is None
        assert result.s3_location is None

    def test_from_dict_full(self, sample_timestamp):
        """Test from_dict with all fields."""
        data = {
            "run_id": "full-test",
            "model_name": "claude-3",
            "backend_name": "litellm",
            "timestamp": "2024-01-15T10:30:00",
            "tasks": [
                {"task_name": "mmlu", "metrics": {"accuracy": 0.85}},
            ],
            "experiment_name": "full-test-exp",
            "workspace": "test-workspace",
            "author": "tester",
            "tags": ["test", "full"],
            "git_ref": "main",
            "model_hash": "hash123",
            "revision": "v1.0",
            "s3_location": "s3://test-bucket/results/",
            "config": {"temperature": 0.0},
            "metadata": {"version": "1.0"},
        }
        result = EvalResult.from_dict(data)
        assert result.config == {"temperature": 0.0}
        assert result.metadata == {"version": "1.0"}
        assert result.experiment_name == "full-test-exp"
        assert result.workspace == "test-workspace"
        assert result.author == "tester"
        assert result.tags == ["test", "full"]
        assert result.git_ref == "main"
        assert result.model_hash == "hash123"
        assert result.revision == "v1.0"
        assert result.s3_location == "s3://test-bucket/results/"

    def test_roundtrip(self, sample_tasks, sample_timestamp):
        """Test to_dict/from_dict roundtrip."""
        original = EvalResult(
            run_id="roundtrip-test",
            model_name="test-model",
            backend_name="mock",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
            experiment_name="roundtrip-exp",
            workspace="test-ws",
            author="tester",
            tags=["tag1", "tag2"],
            git_ref="abc123",
            model_hash="hash456",
            revision="v1",
            s3_location="s3://bucket/path/",
            config={"key": "value"},
            metadata={"info": "test"},
        )
        restored = EvalResult.from_dict(original.to_dict())
        assert restored.run_id == original.run_id
        assert restored.model_name == original.model_name
        assert restored.backend_name == original.backend_name
        assert restored.timestamp == original.timestamp
        assert len(restored.tasks) == len(original.tasks)
        assert restored.experiment_name == original.experiment_name
        assert restored.workspace == original.workspace
        assert restored.author == original.author
        assert restored.tags == original.tags
        assert restored.git_ref == original.git_ref
        assert restored.model_hash == original.model_hash
        assert restored.revision == original.revision
        assert restored.s3_location == original.s3_location
        assert restored.config == original.config
        assert restored.metadata == original.metadata

    def test_empty_tasks(self, sample_timestamp):
        """Test with empty tasks list."""
        result = EvalResult(
            run_id="empty-tasks",
            model_name="test",
            backend_name="mock",
            timestamp=sample_timestamp,
            tasks=[],
        )
        d = result.to_dict()
        assert d["tasks"] == []

        restored = EvalResult.from_dict(d)
        assert restored.tasks == []

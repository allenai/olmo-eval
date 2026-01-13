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
            num_samples=1000,
            subset="validation",
        )
        d = result.to_dict()
        assert d == {
            "task_name": "mmlu",
            "metrics": {"accuracy": 0.75, "f1": 0.72},
            "num_samples": 1000,
            "subset": "validation",
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
        assert result.num_samples is None
        assert result.subset is None

    def test_from_dict_full(self):
        """Test from_dict with all fields."""
        data = {
            "task_name": "arc_challenge",
            "metrics": {"accuracy": 0.52},
            "num_samples": 500,
            "subset": "test",
        }
        result = TaskResult.from_dict(data)
        assert result.task_name == "arc_challenge"
        assert result.metrics == {"accuracy": 0.52}
        assert result.num_samples == 500
        assert result.subset == "test"

    def test_roundtrip(self):
        """Test to_dict/from_dict roundtrip."""
        original = TaskResult(
            task_name="hellaswag",
            metrics={"accuracy": 0.79},
            num_samples=10042,
            subset="validation",
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

    def test_to_dict_full(self, sample_tasks, sample_timestamp):
        """Test to_dict with all fields."""
        result = EvalResult(
            run_id="def456",
            model_name="olmo-2-7b",
            backend_name="hf",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
            config={"batch_size": 32},
            metadata={"git_sha": "abc123"},
        )
        d = result.to_dict()
        assert d["config"] == {"batch_size": 32}
        assert d["metadata"] == {"git_sha": "abc123"}

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
            "config": {"temperature": 0.0},
            "metadata": {"version": "1.0"},
        }
        result = EvalResult.from_dict(data)
        assert result.config == {"temperature": 0.0}
        assert result.metadata == {"version": "1.0"}

    def test_roundtrip(self, sample_tasks, sample_timestamp):
        """Test to_dict/from_dict roundtrip."""
        original = EvalResult(
            run_id="roundtrip-test",
            model_name="test-model",
            backend_name="mock",
            timestamp=sample_timestamp,
            tasks=sample_tasks,
            config={"key": "value"},
            metadata={"info": "test"},
        )
        restored = EvalResult.from_dict(original.to_dict())
        assert restored.run_id == original.run_id
        assert restored.model_name == original.model_name
        assert restored.backend_name == original.backend_name
        assert restored.timestamp == original.timestamp
        assert len(restored.tasks) == len(original.tasks)
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

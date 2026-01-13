"""Tests for FileBackend storage."""

import json
from datetime import datetime, timedelta

import pytest

from olmo_eval.storage import EvalResult, FileBackend, TaskResult
from olmo_eval.storage.file import convert_runner_results


@pytest.fixture
def temp_storage(tmp_path):
    """Create a FileBackend with a temporary directory."""
    return FileBackend(output_dir=tmp_path)


@pytest.fixture
def sample_result():
    """Create a sample EvalResult."""
    return EvalResult(
        run_id="test-run-001",
        model_name="llama3.1-8b",
        backend_name="vllm",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        tasks=[
            TaskResult(task_name="mmlu", metrics={"accuracy": 0.65}, num_samples=100),
            TaskResult(task_name="gsm8k", metrics={"exact_match": 0.58}, num_samples=50),
        ],
        config={"batch_size": 32},
        metadata={"git_sha": "abc123"},
    )


class TestFileBackend:
    """Tests for FileBackend."""

    def test_save_creates_file(self, temp_storage, sample_result, tmp_path):
        """Test that save creates a JSON file."""
        run_id = temp_storage.save(sample_result)
        assert run_id == sample_result.run_id

        # Check file exists (model slug: llama3.1-8b -> llama3-1-8b)
        model_dir = tmp_path / "llama3-1-8b"
        assert model_dir.exists()
        files = list(model_dir.glob("*.json"))
        assert len(files) == 1

        # Check content
        with open(files[0]) as f:
            data = json.load(f)
        assert data["run_id"] == "test-run-001"
        assert data["model_name"] == "llama3.1-8b"

    def test_save_updates_index(self, temp_storage, sample_result, tmp_path):
        """Test that save updates the index file."""
        temp_storage.save(sample_result)

        index_path = tmp_path / ".index.json"
        assert index_path.exists()

        with open(index_path) as f:
            index = json.load(f)
        assert "test-run-001" in index
        assert index["test-run-001"]["model_name"] == "llama3.1-8b"
        assert "mmlu" in index["test-run-001"]["task_names"]
        assert "gsm8k" in index["test-run-001"]["task_names"]

    def test_get_retrieves_result(self, temp_storage, sample_result):
        """Test that get retrieves a saved result."""
        temp_storage.save(sample_result)

        retrieved = temp_storage.get("test-run-001")
        assert retrieved is not None
        assert retrieved.run_id == sample_result.run_id
        assert retrieved.model_name == sample_result.model_name
        assert len(retrieved.tasks) == 2

    def test_get_returns_none_for_missing(self, temp_storage):
        """Test that get returns None for non-existent run_id."""
        result = temp_storage.get("nonexistent")
        assert result is None

    def test_delete_removes_file_and_index(self, temp_storage, sample_result, tmp_path):
        """Test that delete removes both file and index entry."""
        temp_storage.save(sample_result)

        # Verify file exists (model slug: llama3.1-8b -> llama3-1-8b)
        model_dir = tmp_path / "llama3-1-8b"
        files = list(model_dir.glob("*.json"))
        assert len(files) == 1

        # Delete
        deleted = temp_storage.delete("test-run-001")
        assert deleted is True

        # Verify file removed
        files = list(model_dir.glob("*.json"))
        assert len(files) == 0

        # Verify index updated
        with open(tmp_path / ".index.json") as f:
            index = json.load(f)
        assert "test-run-001" not in index

    def test_delete_returns_false_for_missing(self, temp_storage):
        """Test that delete returns False for non-existent run_id."""
        deleted = temp_storage.delete("nonexistent")
        assert deleted is False

    def test_query_by_model(self, temp_storage):
        """Test querying results by model name."""
        # Create results for different models
        for i, model in enumerate(["llama3.1-8b", "llama3.1-8b", "olmo-2-7b"]):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name=model,
                backend_name="vllm",
                timestamp=datetime(2024, 1, 15, 10, i, 0),
                tasks=[TaskResult(task_name="mmlu", metrics={"accuracy": 0.5 + i * 0.1})],
            )
            temp_storage.save(result)

        # Query for llama results
        llama_results = temp_storage.query(model_name="llama3.1-8b")
        assert len(llama_results) == 2
        for r in llama_results:
            assert r.model_name == "llama3.1-8b"

    def test_query_by_task(self, temp_storage):
        """Test querying results by task name."""
        # Create results with different tasks
        result1 = EvalResult(
            run_id="run-1",
            model_name="test",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            tasks=[TaskResult(task_name="mmlu", metrics={"accuracy": 0.65})],
        )
        result2 = EvalResult(
            run_id="run-2",
            model_name="test",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            tasks=[TaskResult(task_name="gsm8k", metrics={"exact_match": 0.58})],
        )
        temp_storage.save(result1)
        temp_storage.save(result2)

        # Query for mmlu results
        mmlu_results = temp_storage.query(task_name="mmlu")
        assert len(mmlu_results) == 1
        assert mmlu_results[0].run_id == "run-1"

    def test_query_by_time_range(self, temp_storage):
        """Test querying results by time range."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        for i in range(5):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name="test",
                backend_name="vllm",
                timestamp=base_time + timedelta(hours=i),
                tasks=[TaskResult(task_name="mmlu", metrics={"accuracy": 0.5})],
            )
            temp_storage.save(result)

        # Query for middle time range
        start = base_time + timedelta(hours=1)
        end = base_time + timedelta(hours=3)
        results = temp_storage.query(start_time=start, end_time=end)
        assert len(results) == 3  # hours 1, 2, 3

    def test_query_limit(self, temp_storage):
        """Test that query respects limit."""
        for i in range(10):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name="test",
                backend_name="vllm",
                timestamp=datetime(2024, 1, 15, 10, i, 0),
                tasks=[TaskResult(task_name="mmlu", metrics={"accuracy": 0.5})],
            )
            temp_storage.save(result)

        results = temp_storage.query(limit=5)
        assert len(results) == 5

    def test_query_returns_sorted_by_timestamp_desc(self, temp_storage):
        """Test that query returns results sorted by timestamp descending."""
        for i in range(5):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name="test",
                backend_name="vllm",
                timestamp=datetime(2024, 1, 15, 10, i, 0),
                tasks=[TaskResult(task_name="mmlu", metrics={"accuracy": 0.5})],
            )
            temp_storage.save(result)

        results = temp_storage.query()
        timestamps = [r.timestamp for r in results]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_models(self, temp_storage):
        """Test listing all models."""
        models = ["llama3.1-8b", "olmo-2-7b", "gpt-4"]
        for i, model in enumerate(models):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name=model,
                backend_name="vllm",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                tasks=[],
            )
            temp_storage.save(result)

        listed = temp_storage.list_models()
        assert sorted(listed) == sorted(models)

    def test_list_runs(self, temp_storage, sample_result):
        """Test listing runs with summary info."""
        temp_storage.save(sample_result)

        runs = temp_storage.list_runs()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "test-run-001"
        assert runs[0]["model_name"] == "llama3.1-8b"
        assert "mmlu" in runs[0]["task_names"]

    def test_list_runs_filtered_by_model(self, temp_storage):
        """Test listing runs filtered by model."""
        for i, model in enumerate(["llama3.1-8b", "llama3.1-8b", "olmo-2-7b"]):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name=model,
                backend_name="vllm",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                tasks=[],
            )
            temp_storage.save(result)

        llama_runs = temp_storage.list_runs(model_name="llama3.1-8b")
        assert len(llama_runs) == 2


class TestConvertRunnerResults:
    """Tests for convert_runner_results helper."""

    def test_convert_basic(self):
        """Test converting basic runner results."""
        runner_results = {
            "model": "llama3.1-8b",
            "backend": "vllm",
            "timestamp": "2024-01-15T10:30:00",
            "tasks": {
                "mmlu": {
                    "config": {"name": "mmlu"},
                    "num_instances": 100,
                    "metrics": {"accuracy": 0.65},
                },
                "gsm8k": {
                    "config": {"name": "gsm8k"},
                    "num_instances": 50,
                    "metrics": {"exact_match": 0.58},
                },
            },
        }

        result = convert_runner_results(runner_results, "test-run-id")

        assert result.run_id == "test-run-id"
        assert result.model_name == "llama3.1-8b"
        assert result.backend_name == "vllm"
        assert result.timestamp == datetime(2024, 1, 15, 10, 30, 0)
        assert len(result.tasks) == 2

        # Check task results
        task_names = {t.task_name for t in result.tasks}
        assert task_names == {"mmlu", "gsm8k"}

        mmlu_task = next(t for t in result.tasks if t.task_name == "mmlu")
        assert mmlu_task.metrics == {"accuracy": 0.65}
        assert mmlu_task.num_samples == 100

    def test_convert_empty_tasks(self):
        """Test converting results with no tasks."""
        runner_results = {
            "model": "test-model",
            "backend": "mock",
            "timestamp": "2024-01-15T10:30:00",
            "tasks": {},
        }

        result = convert_runner_results(runner_results, "empty-run")
        assert result.tasks == []

"""Integration tests for storage backends.

These tests require Docker to run postgres and localstack containers.
Run with: pytest --integration tests/integration/test_storage.py
"""

from datetime import datetime

import pytest


class TestPostgresBackend:
    """Integration tests for PostgresBackend."""

    @pytest.mark.integration
    def test_save_and_get(self, postgres_backend, sample_eval_result):
        """Test saving and retrieving a result."""
        run_id = postgres_backend.save(sample_eval_result)
        assert run_id == sample_eval_result.run_id

        retrieved = postgres_backend.get(run_id)
        assert retrieved is not None
        assert retrieved.run_id == sample_eval_result.run_id
        assert retrieved.model_name == sample_eval_result.model_name
        assert retrieved.backend_name == sample_eval_result.backend_name
        assert len(retrieved.tasks) == 2

    @pytest.mark.integration
    def test_get_nonexistent(self, postgres_backend):
        """Test getting a non-existent result returns None."""
        result = postgres_backend.get("nonexistent-id")
        assert result is None

    @pytest.mark.integration
    def test_delete(self, postgres_backend, sample_eval_result):
        """Test deleting a result."""
        postgres_backend.save(sample_eval_result)

        deleted = postgres_backend.delete(sample_eval_result.run_id)
        assert deleted is True

        # Verify it's gone
        retrieved = postgres_backend.get(sample_eval_result.run_id)
        assert retrieved is None

    @pytest.mark.integration
    def test_delete_nonexistent(self, postgres_backend):
        """Test deleting a non-existent result returns False."""
        deleted = postgres_backend.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.integration
    def test_query_by_model(self, postgres_backend, multiple_eval_results):
        """Test querying by model name."""
        # Save all results
        for result in multiple_eval_results:
            postgres_backend.save(result)

        # Query for llama3.1-8b
        results = postgres_backend.query(model_name="llama3.1-8b")
        assert len(results) == 3  # 3 tasks for llama3.1-8b
        for r in results:
            assert r.model_name == "llama3.1-8b"

    @pytest.mark.integration
    def test_query_by_task(self, postgres_backend, multiple_eval_results):
        """Test querying by task name."""
        for result in multiple_eval_results:
            postgres_backend.save(result)

        # Query for mmlu results
        results = postgres_backend.query(task_name="mmlu")
        assert len(results) == 3  # 3 models have mmlu results

    @pytest.mark.integration
    def test_query_by_time_range(self, postgres_backend, multiple_eval_results):
        """Test querying by time range."""
        for result in multiple_eval_results:
            postgres_backend.save(result)

        # Query for results in the middle time range
        # Results are at 10:00, 10:01, 10:02, 11:00, 11:01, 11:02, 12:00, 12:01, 12:02
        start = datetime(2024, 1, 15, 10, 30, 0)
        end = datetime(2024, 1, 15, 11, 30, 0)

        results = postgres_backend.query(start_time=start, end_time=end)
        # Should get 11:00, 11:01, 11:02
        assert len(results) == 3

    @pytest.mark.integration
    def test_query_limit(self, postgres_backend, multiple_eval_results):
        """Test that query respects limit."""
        for result in multiple_eval_results:
            postgres_backend.save(result)

        results = postgres_backend.query(limit=5)
        assert len(results) == 5

    @pytest.mark.integration
    def test_upsert_behavior(self, postgres_backend, sample_eval_result):
        """Test that saving the same run_id updates the record."""
        postgres_backend.save(sample_eval_result)

        # Modify and save again
        from olmo_eval.storage import EvalResult, TaskResult

        updated = EvalResult(
            run_id=sample_eval_result.run_id,
            model_name="updated-model",
            backend_name="hf",
            timestamp=sample_eval_result.timestamp,
            tasks=[TaskResult(task_name="new_task", metrics={"score": 0.99})],
        )
        postgres_backend.save(updated)

        retrieved = postgres_backend.get(sample_eval_result.run_id)
        assert retrieved.model_name == "updated-model"
        assert len(retrieved.tasks) == 1
        assert retrieved.tasks[0].task_name == "new_task"


class TestS3Backend:
    """Integration tests for S3Backend with LocalStack."""

    @pytest.mark.integration
    def test_save_and_get(self, s3_backend, sample_eval_result):
        """Test saving and retrieving a result."""
        run_id = s3_backend.save(sample_eval_result)
        assert run_id == sample_eval_result.run_id

        retrieved = s3_backend.get(run_id)
        assert retrieved is not None
        assert retrieved.run_id == sample_eval_result.run_id
        assert retrieved.model_name == sample_eval_result.model_name
        assert len(retrieved.tasks) == 2

    @pytest.mark.integration
    def test_get_nonexistent(self, s3_backend):
        """Test getting a non-existent result returns None."""
        result = s3_backend.get("nonexistent-id")
        assert result is None

    @pytest.mark.integration
    def test_delete(self, s3_backend, sample_eval_result):
        """Test deleting a result."""
        s3_backend.save(sample_eval_result)

        deleted = s3_backend.delete(sample_eval_result.run_id)
        assert deleted is True

        # Verify it's gone
        retrieved = s3_backend.get(sample_eval_result.run_id)
        assert retrieved is None

    @pytest.mark.integration
    def test_delete_nonexistent(self, s3_backend):
        """Test deleting a non-existent result returns False."""
        deleted = s3_backend.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.integration
    def test_query_by_model(self, s3_backend, multiple_eval_results):
        """Test querying by model name."""
        for result in multiple_eval_results:
            s3_backend.save(result)

        results = s3_backend.query(model_name="llama3.1-8b")
        assert len(results) == 3
        for r in results:
            assert r.model_name == "llama3.1-8b"

    @pytest.mark.integration
    def test_query_by_task(self, s3_backend, multiple_eval_results):
        """Test querying by task name."""
        for result in multiple_eval_results:
            s3_backend.save(result)

        results = s3_backend.query(task_name="mmlu")
        assert len(results) == 3

    @pytest.mark.integration
    def test_query_limit(self, s3_backend, multiple_eval_results):
        """Test that query respects limit."""
        for result in multiple_eval_results:
            s3_backend.save(result)

        results = s3_backend.query(limit=5)
        assert len(results) == 5

    @pytest.mark.integration
    def test_multiple_models_index(self, s3_backend):
        """Test that indexes are maintained per model."""
        from olmo_eval.storage import EvalResult, TaskResult

        # Save results for different models
        for i, model in enumerate(["model-a", "model-b", "model-c"]):
            result = EvalResult(
                run_id=f"run-{i}",
                model_name=model,
                backend_name="vllm",
                timestamp=datetime(2024, 1, 15, 10, 0, 0),
                tasks=[TaskResult(task_name="test", metrics={"score": 0.5})],
            )
            s3_backend.save(result)

        # Each model should have its own results
        for model in ["model-a", "model-b", "model-c"]:
            results = s3_backend.query(model_name=model)
            assert len(results) == 1
            assert results[0].model_name == model

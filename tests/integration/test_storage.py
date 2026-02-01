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
        experiment_id = postgres_backend.save(sample_eval_result)
        assert experiment_id == sample_eval_result.experiment_id

        retrieved = postgres_backend.get(experiment_id)
        assert retrieved is not None
        assert retrieved.experiment_id == sample_eval_result.experiment_id
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

        deleted = postgres_backend.delete(sample_eval_result.experiment_id)
        assert deleted is True

        # Verify it's gone
        retrieved = postgres_backend.get(sample_eval_result.experiment_id)
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
    def test_same_model_different_experiments(self, postgres_backend):
        """Test that same model config produces same model_hash across different experiments."""
        from datetime import datetime

        from olmo_eval.core.types import EvalResult, StoredTaskResult
        from olmo_eval.storage import compute_model_hash

        # Same config, different authors
        config = {"model": "llama3.1-8b", "temperature": 0.7}

        eval1 = EvalResult(
            experiment_id="eval-user1",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            tasks=[
                StoredTaskResult(
                    task_name="mmlu",
                    metrics={"accuracy": {"exact_match": 0.65}},
                    task_hash="mmlu-hash-user1",
                    primary_metric="accuracy:exact_match",
                )
            ],
            model_config=config,
            author="user1",
            experiment_name="test",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        eval2 = EvalResult(
            experiment_id="eval-user2",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 5, 0),
            tasks=[
                StoredTaskResult(
                    task_name="mmlu",
                    metrics={"accuracy": {"exact_match": 0.66}},
                    task_hash="mmlu-hash-user2",
                    primary_metric="accuracy:exact_match",
                )
            ],
            model_config=config,
            author="user2",
            experiment_name="test",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        # Save both evaluations
        postgres_backend.save(eval1)
        postgres_backend.save(eval2)

        # Query database directly to check the relationship
        with postgres_backend.db.session() as session:
            from sqlalchemy import select

            from olmo_eval.storage.backends.postgres.models import Experiment

            stmt = select(Experiment).where(Experiment.experiment_id == "eval-user1")
            exp1 = session.execute(stmt).scalar_one_or_none()

            stmt = select(Experiment).where(Experiment.experiment_id == "eval-user2")
            exp2 = session.execute(stmt).scalar_one_or_none()

            assert exp1 is not None
            assert exp2 is not None

            # Same config should give same model_hash
            expected_model_hash = compute_model_hash(config)
            assert exp1.model_hash == expected_model_hash
            assert exp2.model_hash == expected_model_hash
            assert exp1.model_hash == exp2.model_hash  # Same model!

            # But different experiments
            assert exp1.experiment_id != exp2.experiment_id
            assert exp1.author != exp2.author

            # And different metrics (nested structure)
            assert exp1.task_results[0].metrics["accuracy"]["exact_match"] == 0.65
            assert exp2.task_results[0].metrics["accuracy"]["exact_match"] == 0.66

    @pytest.mark.integration
    def test_get_all_returns_multiple_experiments(self, postgres_backend):
        """Test that get_all() returns all experiments with shared experiment_id."""
        from olmo_eval.core.types import EvalResult, StoredTaskResult

        # Create two experiments with the same experiment_id (simulating multi-model launch)
        shared_experiment_id = "beaker-multi-model-run"

        eval1 = EvalResult(
            experiment_id=shared_experiment_id,
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            tasks=[
                StoredTaskResult(
                    task_name="mmlu",
                    metrics={"accuracy": {"exact_match": 0.65}},
                    task_hash="mmlu-hash-1",
                    primary_metric="accuracy:exact_match",
                )
            ],
            model_config={"model": "llama3.1-8b"},
            author="user",
            experiment_name="test",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        eval2 = EvalResult(
            experiment_id=shared_experiment_id,
            model_name="llama3.1-70b",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            tasks=[
                StoredTaskResult(
                    task_name="mmlu",
                    metrics={"accuracy": {"exact_match": 0.75}},
                    task_hash="mmlu-hash-2",
                    primary_metric="accuracy:exact_match",
                )
            ],
            model_config={"model": "llama3.1-70b"},
            author="user",
            experiment_name="test",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        # Save both with same experiment_id
        postgres_backend.save(eval1)
        postgres_backend.save(eval2)

        # get() returns only the first one
        first_result = postgres_backend.get(shared_experiment_id)
        assert first_result is not None

        # get_all() returns all experiments with that ID
        all_results = postgres_backend.get_all(shared_experiment_id)
        assert len(all_results) == 2
        model_names = {r.model_name for r in all_results}
        assert "llama3.1-8b" in model_names
        assert "llama3.1-70b" in model_names

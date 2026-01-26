"""Integration tests for new PostgreSQL features.

Tests for instance-level predictions and query helpers.
"""

from datetime import datetime

import pytest

from olmo_eval.storage.base import compute_model_id


class TestPostgresBackendWithInstances:
    """Integration tests for PostgresBackend with instance predictions."""

    @pytest.mark.integration
    def test_save_with_instances(self, postgres_backend, sample_eval_result):
        """Test saving an evaluation with instance predictions."""
        instances_by_task = {
            "mmlu": [
                {
                    "native_id": "mmlu_doc_0",
                    "doc_id": 0,
                    "instance_metrics": {"acc": 1.0, "f1": 1.0},
                    "s3_prediction_key": "s3://bucket/mmlu_pred_0.json",
                },
                {
                    "native_id": "mmlu_doc_1",
                    "doc_id": 1,
                    "instance_metrics": {"acc": 0.0, "f1": 0.5},
                    "s3_prediction_key": "s3://bucket/mmlu_pred_1.json",
                },
            ],
            "gsm8k": [
                {
                    "native_id": "gsm8k_doc_0",
                    "doc_id": 0,
                    "instance_metrics": {"exact_match": 1.0},
                },
            ],
        }

        experiment_id = postgres_backend.save_with_instances(sample_eval_result, instances_by_task)
        assert experiment_id == sample_eval_result.experiment_id

        # Verify experiment was saved
        retrieved = postgres_backend.get(experiment_id)
        assert retrieved is not None
        assert len(retrieved.tasks) == 2

    @pytest.mark.integration
    def test_query_instances_by_model(self, postgres_backend, sample_eval_result):
        """Test querying instance predictions by model_id."""
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        # Add model_id to config for testing
        sample_eval_result.config = {"model": "test-model"}

        instances_by_task = {
            "mmlu": [
                {
                    "native_id": "doc_0",
                    "doc_id": 0,
                    "instance_metrics": {"acc": 1.0},
                }
            ]
        }

        postgres_backend.save_with_instances(sample_eval_result, instances_by_task)

        # Query instances
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            instances = repo.get_instances(
                experiment_id=sample_eval_result.experiment_id, task_name="mmlu"
            )

        assert len(instances) == 1
        assert instances[0]["native_id"] == "doc_0"
        assert instances[0]["instance_metrics"] == {"acc": 1.0}


class TestQueryHelpers:
    """Integration tests for query helper functions."""

    @pytest.mark.integration
    def test_get_model_task_metrics(self, postgres_backend, sample_eval_result):
        """Test getting metrics for a specific model."""
        from olmo_eval.storage.db.queries import QueryHelper

        postgres_backend.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_metrics(model_name="llama3.1-8b")

        assert "mmlu" in results
        assert "gsm8k" in results
        assert results["mmlu"] == 0.65
        assert results["gsm8k"] == 0.58

    @pytest.mark.integration
    def test_get_model_task_instances(self, postgres_backend):
        """Test getting instances for a model and task."""
        from olmo_eval.core import EvalResult, StoredTaskResult
        from olmo_eval.storage.db.queries import QueryHelper

        exp = EvalResult(
            experiment_id="test-exp",
            model_name="test-model",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[StoredTaskResult(task_name="test", metrics={"accuracy": 0.7})],
            config={"model": "test"},
            author="test-user",
        )

        instances = [
            {"native_id": "doc_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}},
            {"native_id": "doc_1", "doc_id": 1, "instance_metrics": {"acc": 0.5}},
        ]

        postgres_backend.save_with_instances(exp, {"test": instances})

        model_id = compute_model_id(exp.config)

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(task_name="test", model_id=model_id)

        assert len(results) == 2
        assert results[0]["native_id"] == "doc_0"
        assert results[1]["native_id"] == "doc_1"

    @pytest.mark.integration
    def test_get_model_task_instances_multiple_tasks(self, postgres_backend):
        """Test getting instances for a model across multiple tasks."""
        from olmo_eval.core import EvalResult, StoredTaskResult
        from olmo_eval.storage.db.queries import QueryHelper

        exp = EvalResult(
            experiment_id="test-exp-multi",
            model_name="test-model",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[
                StoredTaskResult(task_name="task1", metrics={"accuracy": 0.7}),
                StoredTaskResult(task_name="task2", metrics={"accuracy": 0.8}),
                StoredTaskResult(task_name="task3", metrics={"accuracy": 0.6}),
            ],
            config={"model": "test"},
            author="test-user",
        )

        instances_task1 = [
            {"native_id": "task1_doc_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}},
            {"native_id": "task1_doc_1", "doc_id": 1, "instance_metrics": {"acc": 0.5}},
        ]
        instances_task2 = [
            {"native_id": "task2_doc_0", "doc_id": 0, "instance_metrics": {"acc": 0.8}},
            {"native_id": "task2_doc_1", "doc_id": 1, "instance_metrics": {"acc": 0.9}},
        ]
        instances_task3 = [
            {"native_id": "task3_doc_0", "doc_id": 0, "instance_metrics": {"acc": 0.6}},
        ]

        postgres_backend.save_with_instances(
            exp, {"task1": instances_task1, "task2": instances_task2, "task3": instances_task3}
        )

        model_id = compute_model_id(exp.config)

        # Test querying multiple tasks
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(
                task_name=["task1", "task2"], model_id=model_id
            )

        assert len(results) == 4  # 2 from task1 + 2 from task2
        native_ids = {r["native_id"] for r in results}
        assert "task1_doc_0" in native_ids
        assert "task1_doc_1" in native_ids
        assert "task2_doc_0" in native_ids
        assert "task2_doc_1" in native_ids
        assert "task3_doc_0" not in native_ids  # task3 not included

        # Test querying single task still works
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(task_name="task3", model_id=model_id)

        assert len(results) == 1
        assert results[0]["native_id"] == "task3_doc_0"

    @pytest.mark.integration
    def test_get_instances_by_model_name(self, postgres_backend):
        """Test querying instances by human-readable model_name instead of model_id."""
        from olmo_eval.core import EvalResult, StoredTaskResult
        from olmo_eval.storage.db.queries import QueryHelper

        # Create evaluation with specific model name
        exp = EvalResult(
            experiment_id="test-by-name",
            model_name="llama3.1-8b-instruct",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[StoredTaskResult(task_name="mmlu", metrics={"accuracy": 0.75})],
            config={"model": "llama3.1-8b", "mode": "instruct"},
            author="test-user",
        )

        instances = [
            {"native_id": "doc_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}},
            {"native_id": "doc_1", "doc_id": 1, "instance_metrics": {"acc": 0.5}},
            {"native_id": "doc_2", "doc_id": 2, "instance_metrics": {"acc": 0.75}},
        ]

        postgres_backend.save_with_instances(exp, {"mmlu": instances})

        # Query by model_name instead of model_id (more user-friendly!)
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(
                task_name="mmlu", model_name="llama3.1-8b-instruct"
            )

        assert len(results) == 3
        assert results[0]["native_id"] == "doc_0"
        assert results[1]["native_id"] == "doc_1"
        assert results[2]["native_id"] == "doc_2"

        # Query by non-existent model name returns empty
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(
                task_name="mmlu", model_name="non-existent-model"
            )

        assert len(results) == 0


class TestUserIsolation:
    """Tests to verify user results are isolated by experiment_id."""

    @pytest.mark.integration
    def test_concurrent_users_same_model_share_model_id(self, postgres_backend):
        """Test that same model config produces same model_id, but experiments are isolated."""
        from olmo_eval.core import EvalResult, StoredTaskResult, compute_model_id
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        # Same config, tasks, and model - only difference is author and experiment_id
        config = {"model": "llama3.1-8b", "temperature": 0.7}

        # User 1's evaluation
        eval_user1 = EvalResult(
            experiment_id="user1-run-123",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[StoredTaskResult(task_name="mmlu", metrics={"accuracy": 0.65})],
            config=config,
            author="alice@example.com",
        )

        instances_user1 = [
            {"native_id": "mmlu_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}},
            {"native_id": "mmlu_1", "doc_id": 1, "instance_metrics": {"acc": 0.5}},
        ]

        # User 2's evaluation - same model, config, task
        eval_user2 = EvalResult(
            experiment_id="user2-run-456",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[StoredTaskResult(task_name="mmlu", metrics={"accuracy": 0.70})],
            config=config,
            author="bob@example.com",
        )

        instances_user2 = [
            {"native_id": "mmlu_0", "doc_id": 0, "instance_metrics": {"acc": 0.8}},
            {"native_id": "mmlu_1", "doc_id": 1, "instance_metrics": {"acc": 0.6}},
        ]

        # Both users save their results
        postgres_backend.save_with_instances(eval_user1, {"mmlu": instances_user1})
        postgres_backend.save_with_instances(eval_user2, {"mmlu": instances_user2})

        # Same config = same model_id (this is correct!)
        model_id = compute_model_id(config)

        # Query instances by experiment_id - this is how users are isolated
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)

            # User 1's instances via experiment_id
            user1_instances = repo.get_instances(experiment_id="user1-run-123")
            assert len(user1_instances) == 2
            assert any(inst["instance_metrics"]["acc"] == 1.0 for inst in user1_instances)
            assert any(inst["instance_metrics"]["acc"] == 0.5 for inst in user1_instances)

            # User 2's instances via experiment_id
            user2_instances = repo.get_instances(experiment_id="user2-run-456")
            assert len(user2_instances) == 2
            assert any(inst["instance_metrics"]["acc"] == 0.8 for inst in user2_instances)
            assert any(inst["instance_metrics"]["acc"] == 0.6 for inst in user2_instances)

            # Both share the same model_id (as they should)
            assert all(inst["model_id"] == model_id for inst in user1_instances)
            assert all(inst["model_id"] == model_id for inst in user2_instances)

            # But querying by model_id returns ALL instances for that model (from all experiments)
            all_model_instances = repo.get_instances(model_id=model_id, task_name="mmlu")
            assert len(all_model_instances) == 4  # 2 from user1 + 2 from user2

            # This is useful for comparing results across different experiments of the same model


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing API."""

    @pytest.mark.integration
    def test_save_without_instances_still_works(self, postgres_backend, sample_eval_result):
        """Test that save() without instances still works."""
        experiment_id = postgres_backend.save(sample_eval_result)
        assert experiment_id == sample_eval_result.experiment_id

        retrieved = postgres_backend.get(experiment_id)
        assert retrieved is not None
        assert len(retrieved.tasks) == 2

    @pytest.mark.integration
    def test_query_api_unchanged(self, postgres_backend, multiple_eval_results):
        """Test that query() API is unchanged."""
        for result in multiple_eval_results:
            postgres_backend.save(result)

        # All existing query patterns should still work
        results_by_model = postgres_backend.query(model_name="llama3.1-8b")
        assert len(results_by_model) > 0

        results_by_task = postgres_backend.query(task_name="mmlu")
        assert len(results_by_task) > 0

        results_with_limit = postgres_backend.query(limit=5)
        assert len(results_with_limit) <= 5

    @pytest.mark.integration
    def test_delete_cascades_to_instances(self, postgres_backend, sample_eval_result):
        """Test that deleting an experiment cascades to instance predictions."""
        instances = {
            "mmlu": [{"native_id": "doc_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}}]
        }

        postgres_backend.save_with_instances(sample_eval_result, instances)

        # Delete experiment
        deleted = postgres_backend.delete(sample_eval_result.experiment_id)
        assert deleted is True

        # Verify experiment is gone
        retrieved = postgres_backend.get(sample_eval_result.experiment_id)
        assert retrieved is None

        # Verify instances are also gone (cascade delete)
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            instances = repo.get_instances(experiment_id=sample_eval_result.experiment_id)

        assert len(instances) == 0

"""Integration tests for new PostgreSQL features.

Tests for instance-level predictions and query helpers.
"""

from datetime import datetime

import pytest

from olmo_eval.storage.db.queries import QueryHelper


class TestPostgresBackendWithInstances:
    """Integration tests for PostgresBackend with instance predictions."""

    @pytest.mark.integration
    def test_save_with_instances(self, postgres_backend, sample_eval_result):
        """Test saving an evaluation with instance predictions."""
        instances_by_task = {
            "mmlu": [
                {
                    "native_id": "mmlu_doc_0",
                    "instance_metrics": {"acc": 1.0, "f1": 1.0},
                },
                {
                    "native_id": "mmlu_doc_1",
                    "instance_metrics": {"acc": 0.0, "f1": 0.5},
                },
            ],
            "gsm8k": [
                {
                    "native_id": "gsm8k_doc_0",
                    "instance_metrics": {"exact_match": 1.0},
                },
            ],
        }

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            experiment_pk = helper.save_with_instances(sample_eval_result, instances_by_task)

        # experiment_pk is now an int
        assert isinstance(experiment_pk, int)

        # Verify experiment was saved
        retrieved = postgres_backend.get(sample_eval_result.experiment_id)
        assert retrieved is not None
        assert len(retrieved.tasks) == 2

    @pytest.mark.integration
    def test_query_instances_by_experiment(self, postgres_backend, sample_eval_result):
        """Test querying instance predictions by experiment_pk."""
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        instances_by_task = {
            "mmlu": [
                {
                    "native_id": "doc_0",
                    "instance_metrics": {"acc": 1.0},
                }
            ]
        }

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            experiment_pk = helper.save_with_instances(sample_eval_result, instances_by_task)

        # Query instances by experiment_pk
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            instances = repo.get_instances(experiment_pk=experiment_pk, task_name="mmlu")

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
        """Test getting instances for a task."""
        from olmo_eval.core import EvalResult, StoredTaskResult
        from olmo_eval.storage.db.queries import QueryHelper

        exp = EvalResult(
            experiment_id="test-exp",
            model_name="test-model",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[
                StoredTaskResult(task_name="test", metrics={"accuracy": 0.7}, task_hash="test-hash")
            ],
            model_config={"model": "test"},
            author="test-user",
            experiment_name="test-exp",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        instances = [
            {"native_id": "doc_0", "instance_metrics": {"acc": 1.0}},
            {"native_id": "doc_1", "instance_metrics": {"acc": 0.5}},
        ]

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            experiment_pk = helper.save_with_instances(exp, {"test": instances})

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(task_name="test", experiment_pk=experiment_pk)

        assert len(results) == 2
        assert results[0]["native_id"] == "doc_0"
        assert results[1]["native_id"] == "doc_1"

    @pytest.mark.integration
    def test_get_model_task_instances_multiple_tasks(self, postgres_backend):
        """Test getting instances for multiple tasks."""
        from olmo_eval.core import EvalResult, StoredTaskResult
        from olmo_eval.storage.db.queries import QueryHelper

        exp = EvalResult(
            experiment_id="test-exp-multi",
            model_name="test-model",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[
                StoredTaskResult(
                    task_name="task1", metrics={"accuracy": 0.7}, task_hash="task1-hash"
                ),
                StoredTaskResult(
                    task_name="task2", metrics={"accuracy": 0.8}, task_hash="task2-hash"
                ),
                StoredTaskResult(
                    task_name="task3", metrics={"accuracy": 0.6}, task_hash="task3-hash"
                ),
            ],
            model_config={"model": "test"},
            author="test-user",
            experiment_name="test-exp",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        instances_task1 = [
            {"native_id": "task1_doc_0", "instance_metrics": {"acc": 1.0}},
            {"native_id": "task1_doc_1", "instance_metrics": {"acc": 0.5}},
        ]
        instances_task2 = [
            {"native_id": "task2_doc_0", "instance_metrics": {"acc": 0.8}},
            {"native_id": "task2_doc_1", "instance_metrics": {"acc": 0.9}},
        ]
        instances_task3 = [
            {"native_id": "task3_doc_0", "instance_metrics": {"acc": 0.6}},
        ]

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            experiment_pk = helper.save_with_instances(
                exp, {"task1": instances_task1, "task2": instances_task2, "task3": instances_task3}
            )

        # Test querying multiple tasks
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(
                task_name=["task1", "task2"], experiment_pk=experiment_pk
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
            results = helper.get_model_task_instances(
                task_name="task3", experiment_pk=experiment_pk
            )

        assert len(results) == 1
        assert results[0]["native_id"] == "task3_doc_0"


class TestUserIsolation:
    """Tests to verify user results are isolated by experiment."""

    @pytest.mark.integration
    def test_concurrent_users_same_model_different_experiments(self, postgres_backend):
        """Test that same model config from different users creates separate experiments."""
        from olmo_eval.core import EvalResult, StoredTaskResult
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        # Same config, tasks, and model - only difference is author and experiment_id
        config = {"model": "llama3.1-8b", "temperature": 0.7}

        # User 1's evaluation
        eval_user1 = EvalResult(
            experiment_id="user1-run-123",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[
                StoredTaskResult(
                    task_name="mmlu", metrics={"accuracy": 0.65}, task_hash="mmlu-hash-user1"
                )
            ],
            model_config=config,
            author="alice@example.com",
            experiment_name="test",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        instances_user1 = [
            {"native_id": "mmlu_0", "instance_metrics": {"acc": 1.0}},
            {"native_id": "mmlu_1", "instance_metrics": {"acc": 0.5}},
        ]

        # User 2's evaluation - same model, config, task
        eval_user2 = EvalResult(
            experiment_id="user2-run-456",
            model_name="llama3.1-8b",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[
                StoredTaskResult(
                    task_name="mmlu", metrics={"accuracy": 0.70}, task_hash="mmlu-hash-user2"
                )
            ],
            model_config=config,
            author="bob@example.com",
            experiment_name="test",
            workspace="test",
            git_ref="abc123",
            revision="main",
        )

        instances_user2 = [
            {"native_id": "mmlu_0", "instance_metrics": {"acc": 0.8}},
            {"native_id": "mmlu_1", "instance_metrics": {"acc": 0.6}},
        ]

        # Both users save their results
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            exp_pk_1 = helper.save_with_instances(eval_user1, {"mmlu": instances_user1})
            exp_pk_2 = helper.save_with_instances(eval_user2, {"mmlu": instances_user2})

        # Query instances by experiment_pk
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)

            # User 1's instances via experiment_pk
            user1_instances = repo.get_instances(experiment_pk=exp_pk_1)
            assert len(user1_instances) == 2
            assert any(inst["instance_metrics"]["acc"] == 1.0 for inst in user1_instances)
            assert any(inst["instance_metrics"]["acc"] == 0.5 for inst in user1_instances)

            # User 2's instances via experiment_pk
            user2_instances = repo.get_instances(experiment_pk=exp_pk_2)
            assert len(user2_instances) == 2
            assert any(inst["instance_metrics"]["acc"] == 0.8 for inst in user2_instances)
            assert any(inst["instance_metrics"]["acc"] == 0.6 for inst in user2_instances)

        # Query all instances for mmlu task (from all experiments)
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            all_mmlu_instances = repo.get_instances(task_name="mmlu")

        # Should get instances from both users
        assert len(all_mmlu_instances) == 4  # 2 from user1 + 2 from user2


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
        instances = {"mmlu": [{"native_id": "doc_0", "instance_metrics": {"acc": 1.0}}]}

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            experiment_pk = helper.save_with_instances(sample_eval_result, instances)

        # Delete experiment by experiment_id
        deleted = postgres_backend.delete(sample_eval_result.experiment_id)
        assert deleted is True

        # Verify experiment is gone
        retrieved = postgres_backend.get(sample_eval_result.experiment_id)
        assert retrieved is None

        # Verify instances are also gone (cascade delete)
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            instances = repo.get_instances(experiment_pk=experiment_pk)

        assert len(instances) == 0

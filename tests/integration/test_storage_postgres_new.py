"""Integration tests for new PostgreSQL features.

Tests for instance-level predictions and query helpers.
"""

from datetime import datetime

import pytest


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
        from olmo_eval.storage.repository import InstancePredictionRepository

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
        from olmo_eval.storage.queries import QueryHelper

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
        from olmo_eval.storage import EvalResult, TaskResult
        from olmo_eval.storage.queries import QueryHelper

        exp = EvalResult(
            experiment_id="test-exp",
            model_name="test-model",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[TaskResult(task_name="test", metrics={"accuracy": 0.7})],
            config={"model": "test"},
        )

        instances = [
            {"native_id": "doc_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}},
            {"native_id": "doc_1", "doc_id": 1, "instance_metrics": {"acc": 0.5}},
        ]

        postgres_backend.save_with_instances(exp, {"test": instances})

        # Get model_id
        import hashlib
        import json

        model_id = hashlib.sha256(json.dumps(exp.config, sort_keys=True).encode()).hexdigest()[:16]

        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(model_id=model_id, task_name="test")

        assert len(results) == 2
        assert results[0]["native_id"] == "doc_0"
        assert results[1]["native_id"] == "doc_1"

    @pytest.mark.integration
    def test_get_model_task_instances_multiple_tasks(self, postgres_backend):
        """Test getting instances for a model across multiple tasks."""
        from olmo_eval.storage import EvalResult, TaskResult
        from olmo_eval.storage.queries import QueryHelper

        exp = EvalResult(
            experiment_id="test-exp-multi",
            model_name="test-model",
            backend_name="vllm",
            timestamp=datetime.now(),
            tasks=[
                TaskResult(task_name="task1", metrics={"accuracy": 0.7}),
                TaskResult(task_name="task2", metrics={"accuracy": 0.8}),
                TaskResult(task_name="task3", metrics={"accuracy": 0.6}),
            ],
            config={"model": "test"},
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

        # Get model_id
        import hashlib
        import json

        model_id = hashlib.sha256(json.dumps(exp.config, sort_keys=True).encode()).hexdigest()[:16]

        # Test querying multiple tasks
        with postgres_backend.db.session() as session:
            helper = QueryHelper(session)
            results = helper.get_model_task_instances(
                model_id=model_id, task_name=["task1", "task2"]
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
            results = helper.get_model_task_instances(model_id=model_id, task_name="task3")

        assert len(results) == 1
        assert results[0]["native_id"] == "task3_doc_0"


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
        from olmo_eval.storage.repository import InstancePredictionRepository

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            instances = repo.get_instances(experiment_id=sample_eval_result.experiment_id)

        assert len(instances) == 0

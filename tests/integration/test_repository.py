"""Integration tests for repository layer."""

from datetime import UTC, datetime

import pytest


class TestExperimentRepository:
    """Integration tests for ExperimentRepository."""

    @pytest.mark.integration
    def test_save_experiment(self, postgres_backend, sample_eval_result):
        """Test saving an experiment through repository."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            experiment_id = repo.save(sample_eval_result)

        assert experiment_id == sample_eval_result.experiment_id

        # Verify it was saved
        retrieved = postgres_backend.get(experiment_id)
        assert retrieved is not None
        assert retrieved.model_name == sample_eval_result.model_name

    @pytest.mark.integration
    def test_get_experiment(self, postgres_backend, sample_eval_result):
        """Test retrieving an experiment."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        postgres_backend.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            result = repo.get(sample_eval_result.experiment_id)

        assert result is not None
        assert result.experiment_id == sample_eval_result.experiment_id
        assert result.model_name == sample_eval_result.model_name
        assert len(result.tasks) == 2

    @pytest.mark.integration
    def test_delete_experiment(self, postgres_backend, sample_eval_result):
        """Test deleting an experiment."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        postgres_backend.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            deleted = repo.delete(sample_eval_result.experiment_id)

        assert deleted is True

        # Verify deletion
        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            result = repo.get(sample_eval_result.experiment_id)

        assert result is None

    @pytest.mark.integration
    def test_query_by_model_name(self, postgres_backend, multiple_eval_results):
        """Test querying experiments by model name."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        for result in multiple_eval_results:
            postgres_backend.save(result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.query(model_name="llama3.1-8b")

        assert len(results) == 3
        for r in results:
            assert r.model_name == "llama3.1-8b"

    @pytest.mark.integration
    def test_query_by_task_name(self, postgres_backend, multiple_eval_results):
        """Test querying experiments by task name."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        for result in multiple_eval_results:
            postgres_backend.save(result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.query(task_name="mmlu")

        assert len(results) > 0
        # Verify all have mmlu task
        for r in results:
            task_names = [t.task_name for t in r.tasks]
            assert "mmlu" in task_names

    @pytest.mark.integration
    def test_query_by_time_range(self, postgres_backend, multiple_eval_results):
        """Test querying experiments by time range."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        for result in multiple_eval_results:
            postgres_backend.save(result)

        start = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        end = datetime(2024, 1, 15, 11, 30, 0, tzinfo=UTC)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.query(start_time=start, end_time=end)

        assert len(results) > 0
        for r in results:
            assert start <= r.timestamp <= end

    @pytest.mark.integration
    def test_query_pagination(self, postgres_backend, multiple_eval_results):
        """Test query pagination."""
        from olmo_eval.storage.db.repository import ExperimentRepository

        for result in multiple_eval_results:
            postgres_backend.save(result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)

            # Get first page
            page1 = repo.query(limit=5, offset=0)
            # Get second page
            page2 = repo.query(limit=5, offset=5)

        assert len(page1) <= 5
        assert len(page2) <= 5

        # Verify no overlap
        page1_ids = {r.experiment_id for r in page1}
        page2_ids = {r.experiment_id for r in page2}
        assert len(page1_ids & page2_ids) == 0


class TestInstancePredictionRepository:
    """Integration tests for InstancePredictionRepository."""

    @pytest.mark.integration
    def test_save_instances(self, postgres_backend, sample_eval_result):
        """Test saving instance predictions."""
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        postgres_backend.save(sample_eval_result)

        instances = [
            {
                "native_id": "doc_0",
                "doc_id": 0,
                "instance_metrics": {"acc": 1.0},
                "s3_prediction_key": "s3://bucket/pred_0.json",
            },
            {
                "native_id": "doc_1",
                "doc_id": 1,
                "instance_metrics": {"acc": 0.5},
            },
        ]

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            repo.save_instances(
                experiment_id=sample_eval_result.experiment_id,
                task_name="mmlu",
                instances=instances,
                model_id="test-model-id",
            )

        # Verify instances were saved
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            saved = repo.get_instances(experiment_id=sample_eval_result.experiment_id)

        assert len(saved) == 2

    @pytest.mark.integration
    def test_get_instances_by_model(self, postgres_backend, sample_eval_result):
        """Test retrieving instances by model_id."""
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        postgres_backend.save(sample_eval_result)

        instances = [{"native_id": "doc_0", "doc_id": 0, "instance_metrics": {"acc": 1.0}}]

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            repo.save_instances(
                experiment_id=sample_eval_result.experiment_id,
                task_name="mmlu",
                instances=instances,
                model_id="model-123",
            )

        # Query by model_id
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            results = repo.get_instances(model_id="model-123", task_name="mmlu")

        assert len(results) == 1
        assert results[0]["model_id"] == "model-123"

    @pytest.mark.integration
    def test_get_instances_pagination(self, postgres_backend, sample_eval_result):
        """Test instance pagination."""
        from olmo_eval.storage.db.repository import InstancePredictionRepository

        postgres_backend.save(sample_eval_result)

        # Save 10 instances
        instances = [
            {"native_id": f"doc_{i}", "doc_id": i, "instance_metrics": {"acc": 0.5}}
            for i in range(10)
        ]

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            repo.save_instances(
                experiment_id=sample_eval_result.experiment_id,
                task_name="test",
                instances=instances,
            )

        # Get with pagination
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            page1 = repo.get_instances(
                experiment_id=sample_eval_result.experiment_id, limit=5, offset=0
            )
            page2 = repo.get_instances(
                experiment_id=sample_eval_result.experiment_id, limit=5, offset=5
            )

        assert len(page1) == 5
        assert len(page2) == 5

        # Verify no overlap
        page1_ids = {inst["native_id"] for inst in page1}
        page2_ids = {inst["native_id"] for inst in page2}
        assert len(page1_ids & page2_ids) == 0

"""Integration tests for repository layer."""

from datetime import UTC, datetime

import pytest


class TestExperimentRepository:
    """Integration tests for ExperimentRepository."""

    @pytest.mark.integration
    def test_save_experiment(self, postgres_backend, sample_eval_result):
        """Test saving an experiment through repository."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            experiment_pk = repo.save(sample_eval_result)

        # experiment_pk is now an int (auto-increment PK)
        assert isinstance(experiment_pk, int)

        # Verify it was saved (using backend's get which uses experiment_id)
        retrieved = postgres_backend.get(sample_eval_result.experiment_id)
        assert retrieved is not None
        assert retrieved.model_name == sample_eval_result.model_name

    @pytest.mark.integration
    def test_get_experiment_by_pk(self, postgres_backend, sample_eval_result):
        """Test retrieving an experiment by primary key."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            experiment_pk = repo.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            result = repo.get(experiment_pk)

        assert result is not None
        assert result.experiment_id == sample_eval_result.experiment_id
        assert result.model_name == sample_eval_result.model_name
        assert len(result.tasks) == 2

    @pytest.mark.integration
    def test_get_experiment_by_experiment_id(self, postgres_backend, sample_eval_result):
        """Test retrieving experiments by experiment_id."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        postgres_backend.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.get_by_experiment_id(sample_eval_result.experiment_id)

        assert len(results) == 1
        assert results[0].experiment_id == sample_eval_result.experiment_id
        assert results[0].model_name == sample_eval_result.model_name

    @pytest.mark.integration
    def test_delete_experiment(self, postgres_backend, sample_eval_result):
        """Test deleting an experiment by primary key."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            experiment_pk = repo.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            deleted = repo.delete(experiment_pk)

        assert deleted is True

        # Verify deletion
        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            result = repo.get(experiment_pk)

        assert result is None

    @pytest.mark.integration
    def test_delete_by_experiment_id(self, postgres_backend, sample_eval_result):
        """Test deleting all experiments with an experiment_id."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        postgres_backend.save(sample_eval_result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            deleted_count = repo.delete_by_experiment_id(sample_eval_result.experiment_id)

        assert deleted_count == 1

        # Verify deletion
        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.get_by_experiment_id(sample_eval_result.experiment_id)

        assert len(results) == 0

    @pytest.mark.integration
    def test_query_by_model_name(self, postgres_backend, multiple_eval_results):
        """Test querying experiments by model name."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        for result in multiple_eval_results:
            postgres_backend.save(result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.query(model_names=["llama3.1-8b"])

        assert len(results) == 3
        for r in results:
            assert r.model_name == "llama3.1-8b"

    @pytest.mark.integration
    def test_query_by_task_name(self, postgres_backend, multiple_eval_results):
        """Test querying experiments by task name."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

        for result in multiple_eval_results:
            postgres_backend.save(result)

        with postgres_backend.db.session() as session:
            repo = ExperimentRepository(session)
            results = repo.query(task_names=["mmlu"])

        assert len(results) > 0
        # Verify all have mmlu task
        for r in results:
            task_names = [t.task_name for t in r.tasks]
            assert "mmlu" in task_names

    @pytest.mark.integration
    def test_query_by_time_range(self, postgres_backend, multiple_eval_results):
        """Test querying experiments by time range."""
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

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
        from olmo_eval.storage.backends.postgres.repository import ExperimentRepository

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

        # Verify no overlap (using experiment_id since that's what we have)
        page1_ids = {r.experiment_id for r in page1}
        page2_ids = {r.experiment_id for r in page2}
        assert len(page1_ids & page2_ids) == 0


class TestInstancePredictionRepository:
    """Integration tests for InstancePredictionRepository."""

    @pytest.mark.integration
    def test_save_instances(self, postgres_backend, sample_eval_result):
        """Test saving instance predictions."""
        from olmo_eval.storage.backends.postgres.repository import (
            ExperimentRepository,
            InstancePredictionRepository,
        )

        with postgres_backend.db.session() as session:
            exp_repo = ExperimentRepository(session)
            experiment_pk = exp_repo.save(sample_eval_result)

        instances = [
            {
                "native_id": "doc_0",
                "instance_metrics": {"acc": 1.0},
            },
            {
                "native_id": "doc_1",
                "instance_metrics": {"acc": 0.5},
            },
        ]

        # Get task_hash from sample_eval_result
        task_hash = sample_eval_result.tasks[0].task_hash

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            repo.save_instances(
                experiment_pk=experiment_pk,
                task_hash=task_hash,
                instances=instances,
            )

        # Verify instances were saved
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            saved = repo.get_instances(experiment_pk=experiment_pk)

        assert len(saved) == 2

    @pytest.mark.integration
    def test_get_instances_by_task_name(self, postgres_backend, sample_eval_result):
        """Test retrieving instances by task name (via JOIN)."""
        from olmo_eval.storage.backends.postgres.repository import (
            ExperimentRepository,
            InstancePredictionRepository,
        )

        with postgres_backend.db.session() as session:
            exp_repo = ExperimentRepository(session)
            experiment_pk = exp_repo.save(sample_eval_result)

        instances = [{"native_id": "doc_0", "instance_metrics": {"acc": 1.0}}]
        task_hash = sample_eval_result.tasks[0].task_hash  # mmlu

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            repo.save_instances(
                experiment_pk=experiment_pk,
                task_hash=task_hash,
                instances=instances,
            )

        # Query by task_name (uses JOIN to task_results)
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            results = repo.get_instances(task_name="mmlu")

        assert len(results) == 1
        assert results[0]["task_hash"] == task_hash

    @pytest.mark.integration
    def test_get_instances_pagination(self, postgres_backend, sample_eval_result):
        """Test instance keyset pagination with after_id."""
        from olmo_eval.storage.backends.postgres.repository import (
            ExperimentRepository,
            InstancePredictionRepository,
        )

        with postgres_backend.db.session() as session:
            exp_repo = ExperimentRepository(session)
            experiment_pk = exp_repo.save(sample_eval_result)

        # Save 10 instances
        instances = [{"native_id": f"doc_{i}", "instance_metrics": {"acc": 0.5}} for i in range(10)]
        task_hash = sample_eval_result.tasks[0].task_hash

        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            repo.save_instances(
                experiment_pk=experiment_pk,
                task_hash=task_hash,
                instances=instances,
            )

        # Get with keyset pagination using after_id
        with postgres_backend.db.session() as session:
            repo = InstancePredictionRepository(session)
            # First page - no after_id
            page1 = repo.get_instances(experiment_pk=experiment_pk, limit=5)
            # Second page - after last id from first page
            last_id = page1[-1]["id"]
            page2 = repo.get_instances(experiment_pk=experiment_pk, limit=5, after_id=last_id)

        assert len(page1) == 5
        assert len(page2) == 5

        # Verify no overlap using native_id
        page1_ids = {inst["native_id"] for inst in page1}
        page2_ids = {inst["native_id"] for inst in page2}
        assert len(page1_ids & page2_ids) == 0

        # Verify each instance has an 'id' for pagination
        for inst in page1 + page2:
            assert "id" in inst

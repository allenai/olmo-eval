"""Unit tests for SQLAlchemy ORM models."""

from datetime import datetime

from olmo_eval.storage.db.models import Base, Experiment, InstancePrediction, TaskResult


class TestExperimentModel:
    """Tests for Experiment ORM model."""

    def test_create_minimal_experiment(self):
        """Test creating an experiment with only required fields."""
        exp = Experiment(
            experiment_id="test-001",
            model_name="llama3.1-8b",
            model_hash="abc123",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
            experiment_name="test",
            workspace="test-ws",
            author="test-author",
            git_ref="main",
            revision="v1",
        )
        assert exp.experiment_id == "test-001"
        assert exp.model_name == "llama3.1-8b"
        assert exp.backend_name == "vllm"
        assert exp.model_hash == "abc123"
        assert exp.workspace == "test-ws"

    def test_create_full_experiment(self):
        """Test creating an experiment with all fields."""
        exp = Experiment(
            experiment_id="test-002",
            model_name="olmo-2-7b",
            model_hash="model-abc123",
            backend_name="hf",
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            experiment_name="benchmark-run",
            workspace="ai2/olmo",
            author="test-user",
            tags=["benchmark", "release"],
            git_ref="abc123",
            revision="main",
            s3_location="s3://bucket/results/",
            model_config={"batch_size": 32},
            metadata_={"version": "1.0"},
        )
        assert exp.experiment_id == "test-002"
        assert exp.model_hash == "model-abc123"
        assert exp.workspace == "ai2/olmo"
        assert exp.tags == ["benchmark", "release"]
        assert exp.model_config == {"batch_size": 32}
        assert exp.metadata_ == {"version": "1.0"}

    def test_experiment_repr(self):
        """Test experiment string representation."""
        exp = Experiment(
            experiment_id="test-003",
            model_name="test-model",
            model_hash="hash123",
            backend_name="vllm",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            experiment_name="test",
            workspace="ws",
            author="author",
            git_ref="ref",
            revision="rev",
        )
        repr_str = repr(exp)
        assert "test-003" in repr_str
        assert "test-model" in repr_str


class TestTaskResultModel:
    """Tests for TaskResult ORM model."""

    def test_create_minimal_task_result(self):
        """Test creating a task result with only required fields."""
        task = TaskResult(
            experiment_pk=1,
            model_hash="model-abc",
            task_name="mmlu",
            task_hash="task-123",
            metrics={"accuracy": 0.75},
        )
        assert task.experiment_pk == 1
        assert task.model_hash == "model-abc"
        assert task.task_name == "mmlu"
        assert task.task_hash == "task-123"
        assert task.metrics == {"accuracy": 0.75}
        assert task.num_instances is None

    def test_create_full_task_result(self):
        """Test creating a task result with all fields."""
        task = TaskResult(
            experiment_pk=2,
            model_hash="model-xyz",
            task_name="gsm8k",
            task_hash="hash123",
            task_config={"shots": 5},
            metrics={"exact_match": 0.58, "f1": 0.62},
            num_instances=500,
            primary_metric="exact_match",
            primary_score=0.58,
            s3_metrics_key="s3://bucket/metrics.json",
            s3_predictions_key="s3://bucket/predictions.jsonl",
            s3_requests_key="s3://bucket/requests.jsonl",
        )
        assert task.task_hash == "hash123"
        assert task.task_config == {"shots": 5}
        assert task.num_instances == 500
        assert task.primary_metric == "exact_match"
        assert task.primary_score == 0.58
        assert task.s3_requests_key == "s3://bucket/requests.jsonl"

    def test_task_result_repr(self):
        """Test task result string representation."""
        task = TaskResult(
            experiment_pk=3,
            model_hash="model-repr",
            task_name="arc_challenge",
            task_hash="task-repr",
            metrics={"accuracy": 0.52},
            primary_score=0.52,
        )
        repr_str = repr(task)
        assert "arc_challenge" in repr_str


class TestInstancePredictionModel:
    """Tests for InstancePrediction ORM model."""

    def test_create_minimal_instance_prediction(self):
        """Test creating an instance prediction with required fields."""
        inst = InstancePrediction(
            experiment_pk=1,
            task_hash="task-def456",
            native_id="doc_123",
            instance_metrics={"acc": 1.0},
        )
        assert inst.experiment_pk == 1
        assert inst.task_hash == "task-def456"
        assert inst.native_id == "doc_123"
        assert inst.instance_metrics == {"acc": 1.0}

    def test_create_full_instance_prediction(self):
        """Test creating an instance prediction with all fields."""
        inst = InstancePrediction(
            experiment_pk=2,
            task_hash="task-xyz789",
            native_id="gsm8k_456",
            instance_metrics={"exact_match": 0.0, "f1": 0.3},
        )
        assert inst.task_hash == "task-xyz789"
        assert inst.instance_metrics == {"exact_match": 0.0, "f1": 0.3}

    def test_instance_prediction_repr(self):
        """Test instance prediction string representation."""
        inst = InstancePrediction(
            experiment_pk=3,
            task_hash="task-repr",
            native_id="hellaswag_789",
            instance_metrics={"acc": 1.0},
        )
        repr_str = repr(inst)
        assert "hellaswag_789" in repr_str


class TestModelRelationships:
    """Tests for ORM model relationships."""

    def test_experiment_task_relationship(self):
        """Test that experiment and task_results have proper relationship."""
        exp = Experiment(
            experiment_id="rel-test-001",
            model_name="test-model",
            model_hash="hash-001",
            backend_name="vllm",
            timestamp=datetime.now(),
            experiment_name="test",
            workspace="ws",
            author="author",
            git_ref="ref",
            revision="rev",
        )
        task1 = TaskResult(
            experiment_pk=1,
            model_hash="hash-001",
            task_name="task1",
            task_hash="th1",
            metrics={"score": 0.5},
        )
        task2 = TaskResult(
            experiment_pk=1,
            model_hash="hash-001",
            task_name="task2",
            task_hash="th2",
            metrics={"score": 0.6},
        )

        # Check relationships exist
        assert hasattr(exp, "task_results")
        assert hasattr(task1, "experiment")
        assert hasattr(task2, "experiment")

    def test_experiment_instance_relationship(self):
        """Test that experiment and instance_predictions have proper relationship."""
        exp = Experiment(
            experiment_id="rel-test-002",
            model_name="test-model",
            model_hash="hash-002",
            backend_name="vllm",
            timestamp=datetime.now(),
            experiment_name="test",
            workspace="ws",
            author="author",
            git_ref="ref",
            revision="rev",
        )
        inst = InstancePrediction(
            experiment_pk=1,
            task_hash="task-rel",
            native_id="doc_1",
            instance_metrics={"acc": 1.0},
        )

        assert hasattr(exp, "instance_predictions")
        assert hasattr(inst, "experiment")


class TestTableMetadata:
    """Tests for table metadata and constraints."""

    def test_table_names(self):
        """Test that tables have correct names."""
        assert Experiment.__tablename__ == "experiments"
        assert TaskResult.__tablename__ == "task_results"
        assert InstancePrediction.__tablename__ == "instance_predictions"

    def test_base_metadata(self):
        """Test that all models use the same Base metadata."""
        assert Experiment.metadata is Base.metadata
        assert TaskResult.metadata is Base.metadata
        assert InstancePrediction.metadata is Base.metadata

    def test_indexes_defined(self):
        """Test that important indexes are defined."""
        # Check that index names are in metadata
        index_names = [idx.name for table in Base.metadata.tables.values() for idx in table.indexes]

        # Experiment indexes
        assert "idx_experiments_model_hash" in index_names
        assert "idx_experiments_model_name" in index_names
        assert "idx_experiments_model_name_ts" in index_names

        # TaskResult indexes
        assert "idx_task_results_exp_task" in index_names
        assert "idx_task_results_score_desc" in index_names
        assert "idx_task_results_model_task" in index_names

        # InstancePrediction indexes
        assert "idx_instance_exp_task_hash" in index_names
        assert "idx_instance_task_hash_native" in index_names
        assert "ix_instance_predictions_task_hash" in index_names

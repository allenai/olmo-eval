"""Integration tests for inference metrics reporters."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from olmo_eval.inference.metrics.core.schema import BatchMetrics, GPUSnapshot, RequestMetrics
from olmo_eval.inference.metrics.reporters.console import ConsoleReporter
from olmo_eval.inference.metrics.reporters.jsonl import JSONLReporter
from olmo_eval.inference.metrics.reporters.postgres import PostgresReporter
from olmo_eval.storage.backends.postgres.metrics_models import (
    InferenceRequestMetric,
    InferenceRun,
)


@pytest.fixture
def metrics_db(storage_docker_services):
    """Provide a database session for metrics testing.

    Creates tables before yielding, drops them after.
    """
    pytest.importorskip("psycopg")
    pytest.importorskip("sqlalchemy")

    from olmo_eval.storage.backends.postgres import DatabaseSession, MetricsBase

    db = DatabaseSession(
        host="localhost",
        port=5433,
        database="olmo_eval_test",
        user="test",
        password="test",
        pool_size=2,
        sslmode="disable",
    )

    db.initialize()
    MetricsBase.metadata.create_all(db.engine)

    yield db

    MetricsBase.metadata.drop_all(db.engine)
    db.dispose()


@pytest.fixture
def sample_batch_metrics() -> BatchMetrics:
    """Create sample batch metrics for testing."""
    requests = (
        RequestMetrics(
            request_id="req-001",
            prompt_tokens=50,
            completion_tokens=100,
            end_to_end_latency_s=0.5,
            tokens_per_second=200.0,
            time_to_first_token_s=0.05,
            time_per_output_token_s=0.005,
            finish_reason="stop",
            model="llama-3.1-8b",
            timestamp=datetime.now(UTC),
        ),
        RequestMetrics(
            request_id="req-002",
            prompt_tokens=75,
            completion_tokens=150,
            end_to_end_latency_s=0.8,
            tokens_per_second=187.5,
            time_to_first_token_s=0.08,
            time_per_output_token_s=0.005,
            finish_reason="stop",
            model="llama-3.1-8b",
            timestamp=datetime.now(UTC),
        ),
        RequestMetrics(
            request_id="req-003",
            prompt_tokens=30,
            completion_tokens=50,
            end_to_end_latency_s=0.3,
            tokens_per_second=166.7,
            finish_reason="length",
            model="llama-3.1-8b",
            timestamp=datetime.now(UTC),
        ),
    )

    return BatchMetrics(
        total_requests=3,
        successful_requests=3,
        failed_requests=0,
        total_prompt_tokens=155,
        total_completion_tokens=300,
        wall_clock_time_s=1.6,
        output_tokens_per_second=187.5,
        mean_latency_s=0.533,
        experiment_id="test-exp-001",
        experiment_name="metrics-integration-test",
        experiment_group="integration-tests",
        model_name="llama-3.1-8b",
        model_hash="abc123def456",
        task_name="mmlu",
        task_hash="mmlu-v1-hash",
        workspace="ai2/olmo-test",
        author="test-runner",
        tags={"environment": "test", "version": "1.0"},
        requests=requests,
        timestamp=datetime.now(UTC),
    )


class TestPostgresReporter:
    """Integration tests for PostgresReporter."""

    @pytest.mark.integration
    def test_report_batch_without_requests(self, metrics_db, sample_batch_metrics):
        """Test storing batch metrics without per-request details."""
        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )

        try:
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
        finally:
            reporter.shutdown()

        # Verify the data was stored
        with metrics_db.session() as session:
            runs = session.query(InferenceRun).all()
            assert len(runs) == 1

            run = runs[0]
            assert run.experiment_id == "test-exp-001"
            assert run.experiment_name == "metrics-integration-test"
            assert run.experiment_group == "integration-tests"
            assert run.model_name == "llama-3.1-8b"
            assert run.model_hash == "abc123def456"
            assert run.task_name == "mmlu"
            assert run.task_hash == "mmlu-v1-hash"
            assert run.workspace == "ai2/olmo-test"
            assert run.author == "test-runner"
            assert run.total_requests == 3
            assert run.successful_requests == 3
            assert run.failed_requests == 0
            assert run.total_prompt_tokens == 155
            assert run.total_completion_tokens == 300
            assert abs(run.wall_clock_time_s - 1.6) < 0.01
            assert abs(run.mean_latency_s - 0.533) < 0.01
            assert run.tags is not None
            assert "environment:test" in run.tags

            # No request metrics should be stored
            request_metrics = session.query(InferenceRequestMetric).all()
            assert len(request_metrics) == 0

    @pytest.mark.integration
    def test_report_batch_with_requests(self, metrics_db, sample_batch_metrics):
        """Test storing batch metrics with per-request details."""
        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )
        reporter.configure(include_requests=True)

        try:
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
        finally:
            reporter.shutdown()

        # Verify batch and request data was stored
        with metrics_db.session() as session:
            runs = session.query(InferenceRun).all()
            assert len(runs) == 1

            run = runs[0]
            assert run.total_requests == 3

            # Check request metrics
            request_metrics = (
                session.query(InferenceRequestMetric)
                .filter(InferenceRequestMetric.inference_run_id == run.id)
                .all()
            )
            assert len(request_metrics) == 3

            request_ids = {rm.request_id for rm in request_metrics}
            assert request_ids == {"req-001", "req-002", "req-003"}

            # Check specific request
            req1 = next(rm for rm in request_metrics if rm.request_id == "req-001")
            assert req1.prompt_tokens == 50
            assert req1.completion_tokens == 100
            assert abs(req1.end_to_end_latency_s - 0.5) < 0.01
            assert req1.finish_reason == "stop"
            assert req1.model == "llama-3.1-8b"

    @pytest.mark.integration
    def test_multiple_batches(self, metrics_db, sample_batch_metrics):
        """Test storing multiple batches."""
        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )

        try:
            # Report first batch
            reporter.report_batch(sample_batch_metrics)

            # Create and report second batch with different experiment
            batch2 = BatchMetrics(
                total_requests=5,
                successful_requests=4,
                failed_requests=1,
                total_prompt_tokens=200,
                total_completion_tokens=400,
                wall_clock_time_s=2.5,
                output_tokens_per_second=160.0,
                mean_latency_s=0.5,
                experiment_id="test-exp-002",
                experiment_name="second-test",
                model_name="llama-3.1-70b",
                timestamp=datetime.now(UTC),
            )
            reporter.report_batch(batch2)
            reporter.flush()
        finally:
            reporter.shutdown()

        # Verify both batches were stored
        with metrics_db.session() as session:
            runs = session.query(InferenceRun).order_by(InferenceRun.id).all()
            assert len(runs) == 2

            assert runs[0].experiment_id == "test-exp-001"
            assert runs[0].model_name == "llama-3.1-8b"
            assert runs[0].total_requests == 3

            assert runs[1].experiment_id == "test-exp-002"
            assert runs[1].model_name == "llama-3.1-70b"
            assert runs[1].total_requests == 5
            assert runs[1].failed_requests == 1

    @pytest.mark.integration
    def test_gpu_snapshots_stored_in_metadata(self, metrics_db):
        """Test that GPU snapshots are stored in metadata field."""
        batch = BatchMetrics(
            total_requests=1,
            successful_requests=1,
            failed_requests=0,
            total_prompt_tokens=10,
            total_completion_tokens=20,
            wall_clock_time_s=0.5,
            output_tokens_per_second=40.0,
            mean_latency_s=0.5,
            experiment_id="gpu-test",
            gpu_snapshots=(
                GPUSnapshot(
                    device_id=0,
                    name="NVIDIA A100",
                    utilization_pct=85.0,
                    memory_used_mb=40000,
                    memory_total_mb=80000,
                    temperature_c=65.0,
                    power_watts=250.0,
                ),
                GPUSnapshot(
                    device_id=1,
                    name="NVIDIA A100",
                    utilization_pct=78.0,
                    memory_used_mb=35000,
                    memory_total_mb=80000,
                    temperature_c=62.0,
                    power_watts=230.0,
                ),
            ),
            timestamp=datetime.now(UTC),
        )

        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )

        try:
            reporter.report_batch(batch)
            reporter.flush()
        finally:
            reporter.shutdown()

        # Verify GPU snapshots are in metadata
        with metrics_db.session() as session:
            run = session.query(InferenceRun).filter_by(experiment_id="gpu-test").first()
            assert run is not None
            assert run.metadata_ is not None
            assert "gpu_snapshots" in run.metadata_
            assert len(run.metadata_["gpu_snapshots"]) == 2
            assert run.metadata_["gpu_snapshots"][0]["name"] == "NVIDIA A100"
            assert run.metadata_["gpu_snapshots"][0]["utilization_pct"] == 85.0


class TestMetricsQueryPatterns:
    """Integration tests for querying stored metrics."""

    @pytest.mark.integration
    def test_query_by_experiment_id(self, metrics_db, sample_batch_metrics):
        """Test querying metrics by experiment_id."""
        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )

        try:
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
        finally:
            reporter.shutdown()

        with metrics_db.session() as session:
            runs = (
                session.query(InferenceRun)
                .filter(InferenceRun.experiment_id == "test-exp-001")
                .all()
            )
            assert len(runs) == 1
            assert runs[0].model_name == "llama-3.1-8b"

    @pytest.mark.integration
    def test_query_by_model_name(self, metrics_db):
        """Test querying metrics by model name."""
        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )

        batches = [
            BatchMetrics(
                total_requests=10,
                successful_requests=10,
                failed_requests=0,
                total_prompt_tokens=100,
                total_completion_tokens=200,
                wall_clock_time_s=1.0,
                output_tokens_per_second=200.0,
                mean_latency_s=0.1,
                experiment_id=f"exp-{i}",
                model_name=model,
                timestamp=datetime.now(UTC),
            )
            for i, model in enumerate(["llama-3.1-8b", "llama-3.1-8b", "llama-3.1-70b"])
        ]

        try:
            for batch in batches:
                reporter.report_batch(batch)
            reporter.flush()
        finally:
            reporter.shutdown()

        with metrics_db.session() as session:
            llama_8b_runs = (
                session.query(InferenceRun).filter(InferenceRun.model_name == "llama-3.1-8b").all()
            )
            assert len(llama_8b_runs) == 2

            llama_70b_runs = (
                session.query(InferenceRun).filter(InferenceRun.model_name == "llama-3.1-70b").all()
            )
            assert len(llama_70b_runs) == 1

    @pytest.mark.integration
    def test_query_by_time_range(self, metrics_db):
        """Test querying metrics by timestamp range."""
        from datetime import timedelta

        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )

        now = datetime.now(UTC)
        timestamps = [
            now - timedelta(hours=2),
            now - timedelta(hours=1),
            now,
        ]

        try:
            for i, ts in enumerate(timestamps):
                batch = BatchMetrics(
                    total_requests=1,
                    successful_requests=1,
                    failed_requests=0,
                    total_prompt_tokens=10,
                    total_completion_tokens=20,
                    wall_clock_time_s=0.1,
                    output_tokens_per_second=200.0,
                    mean_latency_s=0.1,
                    experiment_id=f"time-exp-{i}",
                    timestamp=ts,
                )
                reporter.report_batch(batch)
            reporter.flush()
        finally:
            reporter.shutdown()

        with metrics_db.session() as session:
            # Query last 90 minutes
            cutoff = now - timedelta(minutes=90)
            recent_runs = session.query(InferenceRun).filter(InferenceRun.timestamp >= cutoff).all()
            assert len(recent_runs) == 2  # Last two batches

    @pytest.mark.integration
    def test_cascade_delete(self, metrics_db, sample_batch_metrics):
        """Test that deleting an InferenceRun cascades to request metrics."""
        reporter = PostgresReporter(
            host="localhost",
            port=5433,
            database="olmo_eval_test",
            user="test",
            password="test",
            sslmode="disable",
        )
        reporter.configure(include_requests=True)

        try:
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
        finally:
            reporter.shutdown()

        with metrics_db.session() as session:
            # Verify data exists
            run = session.query(InferenceRun).first()
            assert run is not None
            run_id = run.id

            request_count = (
                session.query(InferenceRequestMetric)
                .filter(InferenceRequestMetric.inference_run_id == run_id)
                .count()
            )
            assert request_count == 3

            # Delete the run
            session.delete(run)
            session.commit()

        with metrics_db.session() as session:
            # Verify cascade delete
            remaining_runs = session.query(InferenceRun).count()
            assert remaining_runs == 0

            remaining_requests = session.query(InferenceRequestMetric).count()
            assert remaining_requests == 0


class TestRegistryIntegration:
    """Test that postgres reporter works through the registry."""

    @pytest.mark.integration
    def test_create_via_registry(self, metrics_db, sample_batch_metrics):
        """Test creating postgres reporter via registry."""
        from olmo_eval.inference.metrics.core.registry import reporter_registry

        reporter = reporter_registry.create(
            {
                "name": "postgres",
                "host": "localhost",
                "port": 5433,
                "database": "olmo_eval_test",
                "user": "test",
                "password": "test",
                "sslmode": "disable",
            }
        )

        try:
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
        finally:
            reporter.shutdown()

        with metrics_db.session() as session:
            runs = session.query(InferenceRun).all()
            assert len(runs) == 1
            assert runs[0].experiment_id == "test-exp-001"


# =============================================================================
# Console Reporter Tests
# =============================================================================


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_report_batch(self, sample_batch_metrics, capsys):
        """Test that console reporter prints batch metrics."""
        reporter = ConsoleReporter()

        reporter.report_batch(sample_batch_metrics)
        reporter.flush()
        reporter.shutdown()

        captured = capsys.readouterr()
        output = captured.out

        # Check key metrics are present in output
        assert "Requests" in output or "requests" in output.lower()
        assert "3" in output  # total_requests
        assert "llama-3.1-8b" in output or "model" in output.lower()

    def test_report_request_verbose(self, capsys):
        """Test that console reporter prints request metrics in verbose mode."""
        reporter = ConsoleReporter()
        reporter.configure(verbose=True)

        request = RequestMetrics(
            request_id="test-req-001",
            prompt_tokens=50,
            completion_tokens=100,
            end_to_end_latency_s=0.5,
            tokens_per_second=200.0,
            model="test-model",
            timestamp=datetime.now(UTC),
        )

        reporter.report_request(request)
        reporter.flush()
        reporter.shutdown()

        captured = capsys.readouterr()
        output = captured.out

        # Request should be printed in verbose mode
        assert "test-req" in output
        assert "50" in output  # prompt_tokens

    def test_create_via_registry(self, sample_batch_metrics, capsys):
        """Test creating console reporter via registry."""
        from olmo_eval.inference.metrics.core.registry import reporter_registry

        reporter = reporter_registry.create("console")

        reporter.report_batch(sample_batch_metrics)
        reporter.flush()
        reporter.shutdown()

        captured = capsys.readouterr()
        assert len(captured.out) > 0


# =============================================================================
# JSONL Reporter Tests
# =============================================================================


class TestJSONLReporter:
    """Tests for JSONLReporter."""

    def test_report_batch_creates_file(self, sample_batch_metrics):
        """Test that JSONL reporter creates file and writes batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics" / "test-metrics.jsonl"

            reporter = JSONLReporter(path=path)
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            # File should exist
            assert path.exists()

            # Read and parse content
            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 1
            data = json.loads(lines[0])

            assert data["type"] == "batch"
            assert data["data"]["total_requests"] == 3
            assert data["data"]["successful_requests"] == 3
            assert data["data"]["model_name"] == "llama-3.1-8b"
            assert data["data"]["experiment_id"] == "test-exp-001"

    def test_report_batch_without_requests(self, sample_batch_metrics):
        """Test that JSONL reporter excludes requests by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = JSONLReporter(path=path)
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                data = json.loads(f.readline())

            # Requests should not be included by default
            assert "requests" not in data["data"]

    def test_report_batch_with_requests(self, sample_batch_metrics):
        """Test that JSONL reporter includes requests when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = JSONLReporter(path=path)
            reporter.configure(include_requests=True)
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                data = json.loads(f.readline())

            # Requests should be included
            assert len(data["data"]["requests"]) == 3
            request_ids = {r["request_id"] for r in data["data"]["requests"]}
            assert request_ids == {"req-001", "req-002", "req-003"}

    def test_report_request(self):
        """Test that JSONL reporter writes individual requests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = JSONLReporter(path=path)

            request = RequestMetrics(
                request_id="test-req-001",
                prompt_tokens=50,
                completion_tokens=100,
                end_to_end_latency_s=0.5,
                tokens_per_second=200.0,
                model="test-model",
                timestamp=datetime.now(UTC),
            )

            reporter.report_request(request)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                data = json.loads(f.readline())

            assert data["type"] == "request"
            assert data["data"]["request_id"] == "test-req-001"
            assert data["data"]["prompt_tokens"] == 50

    def test_multiple_batches_appends(self, sample_batch_metrics):
        """Test that multiple batches are appended to the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = JSONLReporter(path=path)

            # Write first batch
            reporter.report_batch(sample_batch_metrics)

            # Create and write second batch
            batch2 = BatchMetrics(
                total_requests=5,
                successful_requests=5,
                failed_requests=0,
                total_prompt_tokens=200,
                total_completion_tokens=400,
                wall_clock_time_s=2.0,
                output_tokens_per_second=200.0,
                mean_latency_s=0.4,
                experiment_id="test-exp-002",
                timestamp=datetime.now(UTC),
            )
            reporter.report_batch(batch2)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                lines = f.readlines()

            assert len(lines) == 2

            data1 = json.loads(lines[0])
            data2 = json.loads(lines[1])

            assert data1["data"]["experiment_id"] == "test-exp-001"
            assert data2["data"]["experiment_id"] == "test-exp-002"

    def test_creates_parent_directories(self, sample_batch_metrics):
        """Test that JSONL reporter creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "metrics.jsonl"

            reporter = JSONLReporter(path=path)
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            assert path.exists()
            assert path.parent.exists()

    def test_create_via_registry(self, sample_batch_metrics):
        """Test creating JSONL reporter via registry."""
        from olmo_eval.inference.metrics.core.registry import reporter_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = reporter_registry.create({"name": "jsonl", "path": str(path)})
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            assert path.exists()

            with open(path) as f:
                data = json.loads(f.readline())

            assert data["data"]["total_requests"] == 3

    def test_gpu_snapshots_serialized(self):
        """Test that GPU snapshots are properly serialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            batch = BatchMetrics(
                total_requests=1,
                successful_requests=1,
                failed_requests=0,
                total_prompt_tokens=10,
                total_completion_tokens=20,
                wall_clock_time_s=0.5,
                output_tokens_per_second=40.0,
                mean_latency_s=0.5,
                gpu_snapshots=(
                    GPUSnapshot(
                        device_id=0,
                        name="NVIDIA A100",
                        utilization_pct=85.0,
                        memory_used_mb=40000,
                        memory_total_mb=80000,
                    ),
                ),
                timestamp=datetime.now(UTC),
            )

            reporter = JSONLReporter(path=path)
            reporter.report_batch(batch)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                data = json.loads(f.readline())

            assert len(data["data"]["gpu_snapshots"]) == 1
            assert data["data"]["gpu_snapshots"][0]["name"] == "NVIDIA A100"
            assert data["data"]["gpu_snapshots"][0]["utilization_pct"] == 85.0

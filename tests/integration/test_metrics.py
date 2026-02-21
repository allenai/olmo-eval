"""Integration tests for inference metrics reporters."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from olmo_eval.inference.metrics.core.schema import BatchMetrics, GPUSnapshot, RequestMetrics
from olmo_eval.inference.metrics.reporters.console import ConsoleReporter
from olmo_eval.inference.metrics.reporters.db import DbReporter
from olmo_eval.inference.metrics.reporters.file import FileReporter
from olmo_eval.storage.backends.postgres.metrics_models import InferenceSample


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
        provider_kind="vllm_server",
        tags={"environment": "test", "version": "1.0"},
        requests=requests,
        timestamp=datetime.now(UTC),
    )


class TestDbReporter:
    """Integration tests for DbReporter."""

    @pytest.mark.integration
    def test_report_batch(self, metrics_db, sample_batch_metrics):
        """Test storing batch metrics."""
        reporter = DbReporter(
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
            samples = session.query(InferenceSample).all()
            assert len(samples) == 1

            sample = samples[0]
            assert sample.experiment_id == "test-exp-001"
            assert sample.experiment_name == "metrics-integration-test"
            assert sample.experiment_group == "integration-tests"
            assert sample.model_name == "llama-3.1-8b"
            assert sample.model_hash == "abc123def456"
            assert sample.task_name == "mmlu"
            assert sample.task_hash == "mmlu-v1-hash"
            assert sample.workspace == "ai2/olmo-test"
            assert sample.author == "test-runner"
            assert sample.provider_kind == "vllm_server"
            assert sample.total_requests == 3
            assert sample.successful_requests == 3
            assert sample.failed_requests == 0
            assert sample.total_prompt_tokens == 155
            assert sample.total_completion_tokens == 300
            assert abs(sample.wall_clock_time_s - 1.6) < 0.01
            assert abs(sample.mean_latency_s - 0.533) < 0.01
            assert sample.tags is not None
            assert "environment:test" in sample.tags

    @pytest.mark.integration
    def test_multiple_batches(self, metrics_db, sample_batch_metrics):
        """Test storing multiple batches."""
        reporter = DbReporter(
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
                provider_kind="litellm",
                timestamp=datetime.now(UTC),
            )
            reporter.report_batch(batch2)
            reporter.flush()
        finally:
            reporter.shutdown()

        # Verify both batches were stored
        with metrics_db.session() as session:
            samples = session.query(InferenceSample).order_by(InferenceSample.id).all()
            assert len(samples) == 2

            assert samples[0].experiment_id == "test-exp-001"
            assert samples[0].model_name == "llama-3.1-8b"
            assert samples[0].total_requests == 3
            assert samples[0].provider_kind == "vllm_server"

            assert samples[1].experiment_id == "test-exp-002"
            assert samples[1].model_name == "llama-3.1-70b"
            assert samples[1].total_requests == 5
            assert samples[1].failed_requests == 1
            assert samples[1].provider_kind == "litellm"

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

        reporter = DbReporter(
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

        # Verify GPU devices are in metadata (grouped by device)
        with metrics_db.session() as session:
            sample = session.query(InferenceSample).filter_by(experiment_id="gpu-test").first()
            assert sample is not None
            assert sample.metadata_ is not None
            assert "gpu_devices" in sample.metadata_
            # Two devices, each with one sample
            assert len(sample.metadata_["gpu_devices"]) == 2
            assert sample.metadata_["gpu_devices"][0]["name"] == "NVIDIA A100"
            assert sample.metadata_["gpu_devices"][0]["samples"][0]["utilization_pct"] == 85.0

            # Verify GPU summary is computed and stored
            assert "gpu_summary" in sample.metadata_
            summary = sample.metadata_["gpu_summary"]
            assert summary["device_count"] == 2
            assert summary["sample_count"] == 2
            assert summary["avg_utilization_pct"] == (85.0 + 78.0) / 2
            assert summary["max_utilization_pct"] == 85.0
            assert summary["avg_memory_used_mb"] == (40000 + 35000) / 2
            assert summary["max_memory_used_mb"] == 40000
            assert summary["avg_power_watts"] == (250.0 + 230.0) / 2


class TestMetricsQueryPatterns:
    """Integration tests for querying stored metrics."""

    @pytest.mark.integration
    def test_query_by_experiment_id(self, metrics_db, sample_batch_metrics):
        """Test querying metrics by experiment_id."""
        reporter = DbReporter(
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
            samples = (
                session.query(InferenceSample)
                .filter(InferenceSample.experiment_id == "test-exp-001")
                .all()
            )
            assert len(samples) == 1
            assert samples[0].model_name == "llama-3.1-8b"

    @pytest.mark.integration
    def test_query_by_model_name(self, metrics_db):
        """Test querying metrics by model name."""
        reporter = DbReporter(
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
            llama_8b = (
                session.query(InferenceSample)
                .filter(InferenceSample.model_name == "llama-3.1-8b")
                .all()
            )
            assert len(llama_8b) == 2

            llama_70b = (
                session.query(InferenceSample)
                .filter(InferenceSample.model_name == "llama-3.1-70b")
                .all()
            )
            assert len(llama_70b) == 1

    @pytest.mark.integration
    def test_query_by_time_range(self, metrics_db):
        """Test querying metrics by timestamp range."""
        from datetime import timedelta

        reporter = DbReporter(
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
            recent = (
                session.query(InferenceSample).filter(InferenceSample.timestamp >= cutoff).all()
            )
            assert len(recent) == 2  # Last two batches


class TestRegistryIntegration:
    """Test that db reporter works through the registry."""

    @pytest.mark.integration
    def test_create_via_registry(self, metrics_db, sample_batch_metrics):
        """Test creating db reporter via registry."""
        from olmo_eval.inference.metrics.core.registry import reporter_registry

        reporter = reporter_registry.create(
            {
                "name": "db",
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
            samples = session.query(InferenceSample).all()
            assert len(samples) == 1
            assert samples[0].experiment_id == "test-exp-001"


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
# File Reporter Tests
# =============================================================================


class TestFileReporter:
    """Tests for FileReporter."""

    def test_report_batch_creates_file(self, sample_batch_metrics):
        """Test that file reporter creates file and writes batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics" / "test-metrics.jsonl"

            reporter = FileReporter(path=path)
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
        """Test that file reporter excludes requests by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = FileReporter(path=path)
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                data = json.loads(f.readline())

            # Requests should not be included by default
            assert "requests" not in data["data"]

    def test_report_batch_with_requests(self, sample_batch_metrics):
        """Test that file reporter includes requests when configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = FileReporter(path=path)
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
        """Test that file reporter writes individual requests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = FileReporter(path=path)

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

            reporter = FileReporter(path=path)

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
        """Test that file reporter creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "metrics.jsonl"

            reporter = FileReporter(path=path)
            reporter.report_batch(sample_batch_metrics)
            reporter.flush()
            reporter.shutdown()

            assert path.exists()
            assert path.parent.exists()

    def test_create_via_registry(self, sample_batch_metrics):
        """Test creating file reporter via registry."""
        from olmo_eval.inference.metrics.core.registry import reporter_registry

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"

            reporter = reporter_registry.create({"name": "file", "path": str(path)})
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

            reporter = FileReporter(path=path)
            reporter.report_batch(batch)
            reporter.flush()
            reporter.shutdown()

            with open(path) as f:
                data = json.loads(f.readline())

            assert len(data["data"]["gpu_devices"]) == 1
            assert data["data"]["gpu_devices"][0]["name"] == "NVIDIA A100"
            assert data["data"]["gpu_devices"][0]["samples"][0]["utilization_pct"] == 85.0

            # Verify GPU summary is included
            assert "gpu_summary" in data["data"]
            assert data["data"]["gpu_summary"]["device_count"] == 1
            assert data["data"]["gpu_summary"]["avg_utilization_pct"] == 85.0

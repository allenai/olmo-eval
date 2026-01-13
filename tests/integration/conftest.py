"""Pytest configuration and fixtures for integration tests."""

import subprocess
import time
from pathlib import Path

import pytest

DOCKER_COMPOSE_FILE = Path(__file__).parent / "docker-compose.vllm.yml"
VLLM_CONTAINER_NAME = "olmo-eval-vllm-test"
VLLM_STARTUP_TIMEOUT = 300  # 5 minutes for model loading


def pytest_addoption(parser):
    """Add custom command line options for vLLM tests."""
    parser.addoption(
        "--vllm-model",
        action="store",
        default="Qwen/Qwen2-0.5B",
        help="Model to use for vLLM integration tests",
    )


def _is_container_running(container_name: str) -> bool:
    """Check if a Docker container is running."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _is_container_healthy(container_name: str) -> bool:
    """Check if a Docker container is healthy."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Health.Status}}", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "healthy"


def _wait_for_vllm(timeout: int = VLLM_STARTUP_TIMEOUT) -> bool:
    """Wait for vLLM container to be healthy."""
    import httpx

    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(5)
    return False


@pytest.fixture(scope="session")
def vllm_model(request) -> str:
    """Get the model name for vLLM tests."""
    return request.config.getoption("--vllm-model")


@pytest.fixture(scope="session")
def vllm_service(request):
    """Start vLLM Docker container for the test session.

    This fixture manages the lifecycle of the vLLM container:
    - Starts the container if not already running
    - Waits for it to be healthy
    - Yields control to tests
    - Stops the container after tests complete (unless --no-docker)
    """
    no_docker = request.config.getoption("--no-docker")

    if no_docker:
        # Assume vLLM is already running
        yield "http://localhost:8000"
        return

    # Check if container is already running
    already_running = _is_container_running(VLLM_CONTAINER_NAME)

    if not already_running:
        print("\nStarting vLLM container with docker-compose...")
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "up", "-d", "vllm"],
            check=True,
        )

    # Wait for vLLM to be ready
    print("Waiting for vLLM to be ready (this may take a few minutes)...")
    if not _wait_for_vllm():
        # Get logs for debugging
        logs = subprocess.run(
            ["docker", "logs", VLLM_CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        print(f"vLLM logs:\n{logs.stdout}\n{logs.stderr}")
        pytest.fail("vLLM container failed to become healthy")

    print("vLLM is ready!")
    yield "http://localhost:8000"

    # Cleanup: stop container if we started it
    if not already_running and not no_docker:
        print("\nStopping vLLM container...")
        subprocess.run(
            ["docker", "compose", "-f", str(DOCKER_COMPOSE_FILE), "down"],
            check=False,
        )


@pytest.fixture(scope="session")
def vllm_backend(vllm_service, vllm_model):
    """Create a VLLMBackend instance connected to the test container.

    Note: This creates an in-process vLLM instance, not using the Docker
    container's API. For true integration testing with the container,
    use the OpenAI-compatible API directly.
    """
    # Skip if vLLM not installed
    pytest.importorskip("vllm")

    from olmo_eval.backends.vllm import VLLMBackend

    # Create backend with small memory footprint for testing
    backend = VLLMBackend(
        vllm_model,
        max_model_len=512,
        gpu_memory_utilization=0.5,
        dtype="half",
    )

    yield backend


@pytest.fixture
def small_test_prompts() -> list[str]:
    """Provide a small set of test prompts."""
    return [
        "The capital of France is",
        "2 + 2 equals",
        "The color of the sky is",
    ]


# Storage backend integration test fixtures
STORAGE_DOCKER_COMPOSE_FILE = Path(__file__).parent / "docker-compose.yml"


@pytest.fixture(scope="session")
def storage_docker_services():
    """Start docker compose services for storage backend tests.

    This fixture starts postgres and localstack containers, waits for them
    to be healthy, and tears them down after all tests complete.

    If CI_SERVICES_AVAILABLE environment variable is set, assumes services
    are already running (e.g., managed by GitHub Actions services block).
    """
    import os

    # In CI, services are managed externally (e.g., GitHub Actions services block)
    if os.environ.get("CI_SERVICES_AVAILABLE"):
        yield
        return

    # Start services locally
    result = subprocess.run(
        ["docker", "compose", "-f", str(STORAGE_DOCKER_COMPOSE_FILE), "up", "-d", "--wait"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Docker compose up failed: {result.stderr}")
        pytest.skip("Could not start storage docker services")

    # Give services a moment to fully initialize
    time.sleep(2)

    yield

    # Tear down services
    subprocess.run(
        ["docker", "compose", "-f", str(STORAGE_DOCKER_COMPOSE_FILE), "down", "-v"],
        capture_output=True,
    )


@pytest.fixture
def postgres_backend(storage_docker_services):
    """Provide a PostgresBackend connected to the test database.

    Creates tables before yielding, drops them after.
    """
    pytest.importorskip("psycopg")

    from olmo_eval.storage.postgres import PostgresBackend

    backend = PostgresBackend(
        host="localhost",
        port=5433,
        database="olmo_eval_test",
        user="test",
        password="test",
    )

    backend.initialize()
    yield backend
    backend.cleanup()


@pytest.fixture
def s3_backend(storage_docker_services):
    """Provide an S3Backend connected to LocalStack.

    Creates bucket before yielding, cleans up after.
    """
    pytest.importorskip("boto3")

    from olmo_eval.storage.s3 import S3Backend

    backend = S3Backend(
        bucket="test-eval-bucket",
        prefix="results/",
        endpoint_url="http://localhost:4566",
        region="us-east-1",
    )

    backend.initialize()
    yield backend
    backend.cleanup()


@pytest.fixture
def sample_eval_result():
    """Create a sample EvalResult for storage testing."""
    from datetime import datetime

    from olmo_eval.storage import EvalResult, TaskResult

    return EvalResult(
        run_id="test-integration-001",
        model_name="llama3.1-8b",
        backend_name="vllm",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        tasks=[
            TaskResult(
                task_name="mmlu",
                metrics={"accuracy": 0.65},
                num_instances=100,
                task_hash="mmlu-hash-001",
                primary_metric="accuracy",
                primary_score=0.65,
            ),
            TaskResult(
                task_name="gsm8k",
                metrics={"exact_match": 0.58},
                num_instances=50,
                task_hash="gsm8k-hash-001",
                primary_metric="exact_match",
                primary_score=0.58,
            ),
        ],
        experiment_name="integration-test",
        workspace="ai2/olmo-test",
        author="test-runner",
        tags=["test", "integration"],
        git_ref="abc123",
        model_hash="model-hash-test",
        s3_location="s3://test-bucket/results/test-integration-001/",
        config={"batch_size": 32},
        metadata={"test": True},
    )


@pytest.fixture
def multiple_eval_results():
    """Create multiple EvalResults for query testing."""
    from datetime import datetime

    from olmo_eval.storage import EvalResult, TaskResult

    results = []
    models = ["llama3.1-8b", "llama3.1-70b", "olmo-2-7b"]
    tasks_data = [
        ("mmlu", {"accuracy": 0.65}, "accuracy", 0.65),
        ("gsm8k", {"exact_match": 0.58}, "exact_match", 0.58),
        ("arc_challenge", {"accuracy": 0.52}, "accuracy", 0.52),
    ]

    for i, model in enumerate(models):
        for j, (task_name, metrics, primary_metric, primary_score) in enumerate(tasks_data):
            results.append(
                EvalResult(
                    run_id=f"run-{i}-{j}",
                    model_name=model,
                    backend_name="vllm",
                    timestamp=datetime(2024, 1, 15, 10 + i, j, 0),
                    tasks=[
                        TaskResult(
                            task_name=task_name,
                            metrics=metrics,
                            primary_metric=primary_metric,
                            primary_score=primary_score,
                        )
                    ],
                    workspace="ai2/olmo-test",
                    author="test-runner",
                )
            )

    return results

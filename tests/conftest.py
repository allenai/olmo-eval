"""Root pytest configuration."""

import pytest

# Register anyio plugin for async tests
pytest_plugins = ["anyio"]


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU tests (requires GPU and vLLM)",
    )
    parser.addoption(
        "--no-docker",
        action="store_true",
        default=False,
        help="Skip Docker-based integration tests (postgres, localstack)",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires Docker services)"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on flags."""
    # Skip GPU tests unless --gpu flag is passed
    if not config.getoption("--gpu"):
        skip_gpu = pytest.mark.skip(reason="GPU tests require --gpu flag")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Skip integration tests if --no-docker flag is passed
    if config.getoption("--no-docker"):
        skip_integration = pytest.mark.skip(reason="Integration tests skipped with --no-docker")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

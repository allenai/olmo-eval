#!/usr/bin/env bash
set -euo pipefail

# Parse arguments
SKIP_COVERAGE=false
NO_DOCKER=false
RUN_GPU=false
for arg in "$@"; do
    case $arg in
        --skip-coverage)
            SKIP_COVERAGE=true
            shift
            ;;
        --no-docker)
            NO_DOCKER=true
            shift
            ;;
        --gpu)
            RUN_GPU=true
            shift
            ;;
    esac
done

echo "Running verification checks..."

echo ""
echo "==> Syncing dependencies..."
uv sync --dev --extra beaker --extra hf --extra postgres --extra analysis

echo ""
echo "==> Checking formatting with ruff..."
uv run ruff format --check src/ tests/

echo ""
echo "==> Running ruff linter..."
uv run ruff check src/ tests/

echo ""
echo "==> Running ty type checker..."
uv run ty check src/ alembic/

echo ""
echo "==> Running tests with coverage..."

# Build pytest arguments
PYTEST_ARGS="-v"
if [ "$SKIP_COVERAGE" = false ]; then
    PYTEST_ARGS="$PYTEST_ARGS --cov=src/olmo_eval --cov-report=term-missing --cov-report=html"
fi

if [ "$RUN_GPU" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --gpu"
fi

if [ "$NO_DOCKER" = true ]; then
    # Skip docker-based integration tests
    uv run pytest tests/ $PYTEST_ARGS --no-docker
else
    # Start docker containers for integration tests
    echo ""
    echo "==> Starting integration test containers..."
    docker compose -f tests/integration/docker-compose.yml up -d --wait

    # Run tests
    uv run pytest tests/ $PYTEST_ARGS
    TEST_EXIT_CODE=$?

    echo ""
    echo "==> Stopping integration test containers..."
    docker compose -f tests/integration/docker-compose.yml down -v

    if [ $TEST_EXIT_CODE -ne 0 ]; then
        exit $TEST_EXIT_CODE
    fi
fi

if [ "$SKIP_COVERAGE" = false ]; then
    echo ""
    echo "==> Coverage report generated in htmlcov/"
fi

echo ""
echo "All checks passed!"

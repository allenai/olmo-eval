#!/usr/bin/env bash
set -euo pipefail

# Parse arguments
SKIP_COVERAGE=false
for arg in "$@"; do
    case $arg in
        --skip-coverage)
            SKIP_COVERAGE=true
            shift
            ;;
    esac
done

echo "Running verification checks..."

echo ""
echo "==> Checking formatting with ruff..."
uv run ruff format --check src/ tests/

echo ""
echo "==> Running ruff linter..."
uv run ruff check src/ tests/

echo ""
echo "==> Running ty type checker..."
uv run ty check src/

echo ""
echo "==> Running tests with coverage..."
if [ "$SKIP_COVERAGE" = true ]; then
    uv run pytest tests/ --ignore=tests/integration/ -v
else
    uv run pytest tests/ --ignore=tests/integration/ -v --cov=src/olmo_eval --cov-report=term-missing --cov-report=html
    echo ""
    echo "==> Coverage report generated in htmlcov/"
fi

echo ""
echo "All checks passed!"

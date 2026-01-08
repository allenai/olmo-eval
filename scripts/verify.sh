#!/usr/bin/env bash
set -euo pipefail

echo "Running verification checks..."

echo ""
echo "==> Checking formatting with black..."
uv run black --check src/ tests/

echo ""
echo "==> Checking import sorting with isort..."
uv run isort --check-only src/ tests/

echo ""
echo "==> Running ruff linter..."
uv run ruff check src/ tests/

echo ""
echo "==> Running ty type checker..."
uv run ty check src/

echo ""
echo "All checks passed!"

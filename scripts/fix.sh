#!/usr/bin/env bash
set -euo pipefail

echo "Running auto-fix..."

echo ""
echo "==> Formatting with black..."
uv run black src/ tests/

echo ""
echo "==> Sorting imports with isort..."
uv run isort src/ tests/

echo ""
echo "==> Running ruff with --fix..."
uv run ruff check --fix src/ tests/

echo ""
echo "Done! Run ./scripts/verify.sh to confirm all checks pass."

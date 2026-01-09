#!/usr/bin/env bash
set -euo pipefail

echo "Running auto-fix..."

echo ""
echo "==> Formatting with ruff..."
uv run ruff format src/ tests/

echo ""
echo "==> Running ruff check with --fix..."
uv run ruff check --fix src/ tests/

echo ""
echo "Done! Run ./scripts/verify.sh to confirm all checks pass."

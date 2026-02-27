.PHONY: setup setup-all fix verify test lint type-check clean

# Development setup
# Usage: make setup [EXTRAS="beaker storage"]
EXTRAS ?=
EXTRA_FLAGS := $(foreach extra,$(EXTRAS),--extra $(extra))

setup:
	uv sync --dev $(EXTRA_FLAGS)
	uv run pre-commit install

# Full setup with beaker and storage (for launching jobs and fetching results)
setup-all:
	uv sync --dev --extra beaker --extra storage
	uv run pre-commit install

# Auto-fix formatting and lint issues
fix:
	./scripts/fix.sh

# Run all verification checks (lint, type-check, tests)
verify:
	./scripts/verify.sh

# Run unit tests only (no integration)
test:
	uv run pytest tests/ --ignore=tests/integration -v

# Run linter
lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

# Run type checker
type-check:
	uv run ty check src/

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info htmlcov/ .coverage coverage.xml .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Database migrations
db-upgrade:
	uv run scripts/internal/db-migrate upgrade head

db-downgrade:
	uv run scripts/internal/db-migrate downgrade -1

db-status:
	uv run scripts/internal/db-migrate current

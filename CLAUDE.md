# Development Commands

- Use `uv run` for Python commands
- Use `uv run ruff check src/ tests/` and `uv run ruff format --check src/ tests/` for linting
- Use `uv run ty check src/ alembic/` and `uv run pytest tests/ --ignore=tests/integration -v` for local verification

# Code Style

- Keep docstrings general; avoid implementation details that become stale
- Avoid comments that explain temporary or in-progress changes
- Design classes with stable interfaces; avoid coupling methods to specific fields

# Testing

- Adapt tests to match source code, not the reverse
- Ask before adding or modifying tests alongside functional changes

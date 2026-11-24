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
- Always add unit tests for new code

# Pull request descriptions

Write for a capable programmer who has not followed the development session.
Begin with the normal behavior, relevant background, and why the change matters.
Explain specialized terms and system boundaries before relying on them. For a bug
fix, describe the triggering input, the failure, and its user-visible impact;
include a small concrete example where useful. Then explain the implementation,
validation, and material limitations. Distinguish measured results from expected
behavior, and keep draft/incomplete validation explicit. Scale detail to the
change; do not assume experiment names or internal implementation shorthand are
familiar to the reviewer.

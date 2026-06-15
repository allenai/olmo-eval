# Development

## Local Installation

```bash
# Clone the repository
git clone https://github.com/allenai/olmo-eval.git
cd olmo-eval

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python 3.12 if your machine does not already have it
uv python install 3.12

# Install dependencies and the package in editable mode from the checked-in
# lockfile so builds are reproducible. Run `uv lock` to update the lockfile.
uv sync --frozen

# Install git hooks
uv run pre-commit install

# Browse a few suites
uv run olmo-eval suite inspect mmlu
uv run olmo-eval suite inspect gpqa
uv run olmo-eval suite inspect olmobase:code

# Preview a run without loading a model
uv run olmo-eval run -m mock -t gsm8k --dry-run
```

## Common Commands

```bash
# Run lint, type checks, and unit tests directly
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/ alembic/
uv run pytest tests/ --ignore=tests/integration -v

# Optional helper scripts
./scripts/fix.sh
./scripts/verify.sh
```

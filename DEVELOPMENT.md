# Development

## Local Installation

```bash
# Clone the repository
git clone https://github.com/allenai/olmo-eval-internal.git
cd olmo-eval-internal

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and the package in editable mode
uv sync

# Run the CLI
uv run olmo-eval tasks
```

## Publishing to PyPI

```bash
# Set your PyPI token
export PYPI_TOKEN="your-token-here"

# Run the publish script
./scripts/publish.sh
```

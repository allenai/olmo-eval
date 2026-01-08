#!/usr/bin/env bash
set -euo pipefail

# Clean previous builds
rm -rf dist/

# Build the package
uv build

# Publish to PyPI (requires PYPI_TOKEN environment variable)
uv publish --token "${PYPI_TOKEN}"

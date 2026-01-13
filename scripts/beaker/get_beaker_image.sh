#!/usr/bin/env bash
set -euo pipefail

# Get the full Beaker image name for olmo-eval
#
# Usage:
#   ./scripts/get_beaker_image.sh                    # Returns ai2/oe-data/olmo-eval-latest
#   ./scripts/get_beaker_image.sh ai2/other          # Custom workspace

WORKSPACE="${1:-ai2/oe-data}"
IMAGE_NAME="olmo-eval-latest"

echo "${WORKSPACE}/${IMAGE_NAME}"

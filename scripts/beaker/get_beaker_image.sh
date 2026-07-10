#!/usr/bin/env bash
set -euo pipefail

# Get the full Beaker image name for olmo-eval
#
# Usage:
#   ./scripts/beaker/get_beaker_image.sh
#   ./scripts/beaker/get_beaker_image.sh ai2-other olmo-eval-custom

IMAGE_OWNER="${1:-ai2-tylerm}"
IMAGE_NAME="${2:-olmo-eval-cu1281-trc2100-amd64}"

echo "${IMAGE_OWNER}/${IMAGE_NAME}"

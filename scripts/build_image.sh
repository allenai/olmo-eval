#!/usr/bin/env bash
set -euo pipefail

# Build the Docker image locally
#
# Usage:
#   ./scripts/build_image.sh                    # Build with default tag
#   ./scripts/build_image.sh --tag my-tag       # Build with custom tag
#   ./scripts/build_image.sh --no-cache         # Force rebuild without cache
#
# Environment Variables:
#   VLLM_VERSION - vLLM version to use (default: 0.13.0, matches pyproject.toml)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Defaults
IMAGE_NAME="olmo-eval"
TAG="latest"
VLLM_VERSION="${VLLM_VERSION:-0.13.0}"
NO_CACHE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --vllm-version)
            VLLM_VERSION="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tag TAG           Image tag (default: latest)"
            echo "  --no-cache          Force rebuild without cache"
            echo "  --vllm-version VER  vLLM version (default: 0.13.0)"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

echo "Building Docker image..."
echo "  Image: ${IMAGE_NAME}:${TAG}"
echo "  vLLM version: ${VLLM_VERSION}"
echo ""

docker build \
    ${NO_CACHE} \
    --build-arg VLLM_VERSION="${VLLM_VERSION}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "To test locally:"
echo "  docker run --rm ${IMAGE_NAME}:${TAG} --help"
echo "  docker run --rm --gpus all ${IMAGE_NAME}:${TAG} models"

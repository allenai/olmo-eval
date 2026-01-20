#!/usr/bin/env bash
set -euo pipefail

# Build the Docker image locally
#
# Usage:
#   ./scripts/build_image.sh                    # Build with default tag
#   ./scripts/build_image.sh --tag my-tag       # Build with custom tag
#   ./scripts/build_image.sh --no-cache         # Force rebuild without cache
#
# The image includes:
#   - CUDA runtime
#   - Python 3.12 with PyTorch
#   - Flash Attention 2 (pre-installed)
#   - Flash Attention 3 (pre-built wheel, installed on-demand)
#
# Backend dependencies (vllm, transformers, etc.) are installed at runtime.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Load build configuration
source "${SCRIPT_DIR}/build_config.sh"

# Defaults
IMAGE_NAME="olmo-eval"
TAG=""
CUDA_VERSION="${CUDA_VERSION:-${DEFAULT_CUDA_VERSION}}"
TORCH_VERSION="${TORCH_VERSION:-${DEFAULT_TORCH_VERSION}}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
FA2_VERSION="${FA2_VERSION:-2.8.3}"
FA3_COMMIT="${FA3_COMMIT:-92ca9da8d66f7b34ff50dc080ec0fef9661260d6}"
PLATFORM=""
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
        --cuda-version)
            CUDA_VERSION="$2"
            if ! validate_cuda_version "$CUDA_VERSION"; then
                exit 1
            fi
            shift 2
            ;;
        --torch-version)
            TORCH_VERSION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --fa2-version)
            FA2_VERSION="$2"
            shift 2
            ;;
        --fa3-commit)
            FA3_COMMIT="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --tag TAG             Image tag (default: auto-generated)"
            echo "  --cuda-version VER    CUDA version (default: ${DEFAULT_CUDA_VERSION})"
            echo "                        Supported: ${SUPPORTED_CUDA_VERSIONS[*]}"
            echo "  --torch-version VER   PyTorch version (default: ${TORCH_VERSION})"
            echo "  --python-version VER  Python version (default: ${PYTHON_VERSION})"
            echo "  --fa2-version VER     Flash Attention 2 version (default: ${FA2_VERSION})"
            echo "  --fa3-commit SHA      Flash Attention 3 commit (default: ${FA3_COMMIT:0:12}...)"
            echo "  --platform PLATFORM   Target platform (default: auto-detect)"
            echo "                        Options: linux/amd64, linux/arm64"
            echo "  --no-cache            Force rebuild without cache"
            echo "  --help                Show this help"
            echo ""
            echo "The image includes PyTorch, Flash Attention 2 (pre-installed), and"
            echo "Flash Attention 3 (pre-built wheel). Backend deps installed at runtime."
            echo ""
            echo "Examples:"
            echo "  $0 --cuda-version 12.8.0"
            echo "  $0 --torch-version 2.6.0"
            echo "  $0 --platform linux/amd64"
            echo "  FORCE_AMD64=1 $0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Auto-detect platform if not specified
if [[ -z "$PLATFORM" ]]; then
    if [[ "$(uname -m)" == "arm64" ]] && [[ -z "${FORCE_AMD64:-}" ]]; then
        PLATFORM="linux/arm64"
        echo "Detected ARM Mac - building for linux/arm64 (local testing)"
        echo "To build for production amd64, set: FORCE_AMD64=1 or --platform linux/amd64"
    else
        PLATFORM="linux/amd64"
    fi
fi

# Extract platform arch for tagging (amd64 or arm64)
PLATFORM_ARCH=$(echo "$PLATFORM" | cut -d'/' -f2)

# Auto-generate tag if not specified
if [[ -z "$TAG" ]]; then
    # Format: cuda{concatenated}-{arch}
    # Example: cuda128-amd64 (for CUDA 12.8.0)
    CUDA_SHORT=$(cuda_short "$CUDA_VERSION")
    TAG="cuda${CUDA_SHORT}-${PLATFORM_ARCH}"
    echo "Auto-generated tag: ${TAG}"
fi

echo "Building Docker image..."
echo "  Image:          ${IMAGE_NAME}:${TAG}"
echo "  CUDA version:   ${CUDA_VERSION}"
echo "  Python version: ${PYTHON_VERSION}"
echo "  Torch version:  ${TORCH_VERSION}"
echo "  FA2 version:    ${FA2_VERSION}"
echo "  FA3 commit:     ${FA3_COMMIT:0:12}..."
echo "  Platform:       ${PLATFORM}"
echo ""

docker build \
    --platform "${PLATFORM}" \
    ${NO_CACHE} \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg TORCH_VERSION="${TORCH_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg FA2_VERSION="${FA2_VERSION}" \
    --build-arg FA3_COMMIT="${FA3_COMMIT}" \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${REPO_ROOT}/Dockerfile" \
    "${REPO_ROOT}"

echo ""
echo "Build complete: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Image size:"
docker images "${IMAGE_NAME}:${TAG}" --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'
echo ""
echo "Image includes:"
echo "  - Python ${PYTHON_VERSION}"
echo "  - PyTorch ${TORCH_VERSION}"
echo "  - Flash Attention 2 v${FA2_VERSION} (pre-installed from wheel)"
echo "  - Flash Attention 3 (pre-built wheel at /opt/flash-attn-3/)"
echo ""
echo "To test locally:"
echo "  docker run --rm -v \$(pwd):/workspace ${IMAGE_NAME}:${TAG} python -c 'import flash_attn; print(flash_attn.__version__)'"
echo ""

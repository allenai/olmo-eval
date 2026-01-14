#!/usr/bin/env bash
set -euo pipefail

# Build multiple Docker images for different CUDA+PyTorch combinations
#
# This script builds valid CUDA+PyTorch pairs defined in build_config.sh
#
# Usage:
#   ./scripts/build_matrix.sh                    # Build default pair
#   ./scripts/build_matrix.sh --all              # Build all valid pairs
#   ./scripts/build_matrix.sh --push             # Build and push to Beaker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load build configuration
source "${SCRIPT_DIR}/build_config.sh"

# Defaults
BUILD_ALL=false
PUSH_TO_BEAKER=false
PLATFORM="linux/amd64"
NO_CACHE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            BUILD_ALL=true
            shift
            ;;
        --push)
            PUSH_TO_BEAKER=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all         Build all valid CUDA+PyTorch pairs"
            echo "  --push        Push to Beaker after building"
            echo "  --platform P  Target platform (default: linux/amd64)"
            echo "  --no-cache    Force rebuild without cache"
            echo "  --help        Show this help"
            echo ""
            echo "Valid CUDA+PyTorch pairs:"
            for pair in "${VALID_CUDA_TORCH_PAIRS[@]}"; do
                cuda="${pair%%:*}"
                torch="${pair##*:}"
                echo "  - CUDA ${cuda} + PyTorch ${torch}"
            done
            echo ""
            echo "Examples:"
            echo "  $0                    # Build default (CUDA ${DEFAULT_CUDA_VERSION} + PyTorch ${DEFAULT_TORCH_VERSION})"
            echo "  $0 --all              # Build all valid pairs"
            echo "  $0 --all --push       # Build all and push to Beaker"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Build pairs list
if [[ "$BUILD_ALL" == "true" ]]; then
    PAIRS=("${VALID_CUDA_TORCH_PAIRS[@]}")
else
    PAIRS=("${DEFAULT_CUDA_VERSION}:${DEFAULT_TORCH_VERSION}")
fi

TOTAL_BUILDS=${#PAIRS[@]}

echo "========================================="
echo "Docker Image Matrix Build"
echo "========================================="
echo "Platform:       ${PLATFORM}"
echo "Total builds:   ${TOTAL_BUILDS}"
echo "Push to Beaker: ${PUSH_TO_BEAKER}"
echo "========================================="
echo ""

# Track results
BUILD_COUNT=0
SUCCESS_COUNT=0
FAILED_BUILDS=()

# Build each pair
for pair in "${PAIRS[@]}"; do
    cuda="${pair%%:*}"
    torch="${pair##*:}"

    BUILD_COUNT=$((BUILD_COUNT + 1))

    echo ""
    echo "========================================="
    echo "Build ${BUILD_COUNT}/${TOTAL_BUILDS}"
    echo "CUDA ${cuda} + PyTorch ${torch}"
    echo "========================================="

    # Build the image
    if "${SCRIPT_DIR}/build_image.sh" \
        --cuda-version "${cuda}" \
        --torch-version "${torch}" \
        --platform "${PLATFORM}" \
        ${NO_CACHE}; then

        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo "✓ Build successful"

        # Push to Beaker if requested
        if [[ "$PUSH_TO_BEAKER" == "true" ]]; then
            echo "Pushing to Beaker..."
            if "${SCRIPT_DIR}/beaker/push_beaker_image.sh"; then
                echo "✓ Push successful"
            else
                echo "✗ Push failed"
                FAILED_BUILDS+=("CUDA ${cuda} + PyTorch ${torch} (push failed)")
            fi
        fi
    else
        echo "✗ Build failed"
        FAILED_BUILDS+=("CUDA ${cuda} + PyTorch ${torch}")
    fi
done

# Summary
echo ""
echo "========================================="
echo "Build Matrix Summary"
echo "========================================="
echo "Total builds:      ${TOTAL_BUILDS}"
echo "Successful builds: ${SUCCESS_COUNT}"
echo "Failed builds:     $((TOTAL_BUILDS - SUCCESS_COUNT))"

if [[ ${#FAILED_BUILDS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed builds:"
    for failed in "${FAILED_BUILDS[@]}"; do
        echo "  - ${failed}"
    done
    exit 1
else
    echo ""
    echo "✓ All builds completed successfully!"

    # List built images
    echo ""
    echo "Built images:"
    docker images olmo-eval --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | head -n $((TOTAL_BUILDS + 1))
fi

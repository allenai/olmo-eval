#!/usr/bin/env bash
# Docker build configuration
# This file contains shared configuration for building olmo-eval Docker images
#
# Note: PyTorch is NOT included in the base image. It's installed at runtime
# as a transitive dependency of the backend (vllm, transformers, etc.)

# Supported CUDA versions (full patch versions required by NVIDIA images)
# Format: "MAJOR.MINOR.PATCH"
SUPPORTED_CUDA_VERSIONS=(
    "12.6.1"
    "12.8.0"
    "12.9.1"
)

# Default CUDA version
DEFAULT_CUDA_VERSION="12.8.0"

# Supported platforms
SUPPORTED_PLATFORMS=(
    "linux/amd64"
    "linux/arm64"
)

# Beaker workspace
BEAKER_WORKSPACE="ai2/oe-data"

# Helper function: Convert CUDA version to short format (12.6 -> 126)
cuda_short() {
    local version=$1
    echo "${version}" | sed 's/\.//g' | cut -c1-3
}

# Helper function: Validate CUDA version
validate_cuda_version() {
    local version=$1
    for supported in "${SUPPORTED_CUDA_VERSIONS[@]}"; do
        if [[ "$version" == "$supported" ]]; then
            return 0
        fi
    done
    echo "Error: Unsupported CUDA version '${version}'"
    echo "Supported versions: ${SUPPORTED_CUDA_VERSIONS[*]}"
    return 1
}

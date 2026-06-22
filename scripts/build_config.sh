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
    "12.8.1"
    "12.9.1"
)

# Default CUDA version
DEFAULT_CUDA_VERSION="12.8.1"

# Default PyTorch version
DEFAULT_TORCH_VERSION="2.9.0"

# Supported platforms
SUPPORTED_PLATFORMS=(
    "linux/amd64"
    "linux/arm64"
)

# Beaker workspace
BEAKER_WORKSPACE="ai2/oe-data"

# Helper function: Convert CUDA version to short format
# Examples: 12.8.0 -> 128, 12.8.1 -> 1281, 12.9.1 -> 1291
cuda_short() {
    local version=$1
    local no_dots
    no_dots=$(echo "${version}" | sed 's/\.//g')
    # Strip trailing 0 for .0 patch versions (1280 -> 128)
    if [[ "${no_dots}" == *0 ]] && [[ "${#no_dots}" -eq 4 ]]; then
        echo "${no_dots:0:3}"
    else
        echo "${no_dots}"
    fi
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

#!/usr/bin/env bash
# Docker build configuration
# This file contains shared configuration for building olmo-eval Docker images

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

# Supported PyTorch versions
# Format: "MAJOR.MINOR.PATCH"
SUPPORTED_TORCH_VERSIONS=(
    "2.7.1"
    "2.8.0"
    "2.9.1"
)

# Default PyTorch version
DEFAULT_TORCH_VERSION="2.8.0"

# Valid CUDA + PyTorch pairs
# Format: "CUDA_VERSION:TORCH_VERSION"
VALID_CUDA_TORCH_PAIRS=(
    "12.6.1:2.7.1"
    "12.6.1:2.8.0"
    "12.8.0:2.8.0"
    "12.8.0:2.9.1"
    "12.9.1:2.9.1"
)

# Beaker workspace
BEAKER_WORKSPACE="ai2/oe-data"

# Helper function: Convert CUDA version to short format (12.6 -> 126)
cuda_short() {
    local version=$1
    echo "${version}" | sed 's/\.//g' | cut -c1-3
}

# Helper function: Convert torch version to short format (2.8.0 -> 280)
torch_short() {
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

# Helper function: Validate torch version
validate_torch_version() {
    local version=$1
    for supported in "${SUPPORTED_TORCH_VERSIONS[@]}"; do
        if [[ "$version" == "$supported" ]]; then
            return 0
        fi
    done
    echo "Error: Unsupported torch version '${version}'"
    echo "Supported versions: ${SUPPORTED_TORCH_VERSIONS[*]}"
    return 1
}

# Helper function: Validate CUDA+torch pair
validate_cuda_torch_pair() {
    local cuda=$1
    local torch=$2
    local pair="${cuda}:${torch}"

    for valid_pair in "${VALID_CUDA_TORCH_PAIRS[@]}"; do
        if [[ "$pair" == "$valid_pair" ]]; then
            return 0
        fi
    done

    echo "Error: Invalid CUDA+torch combination '${cuda}' with '${torch}'"
    echo "Valid pairs:"
    for valid_pair in "${VALID_CUDA_TORCH_PAIRS[@]}"; do
        echo "  - CUDA ${valid_pair%%:*} + torch ${valid_pair##*:}"
    done
    return 1
}

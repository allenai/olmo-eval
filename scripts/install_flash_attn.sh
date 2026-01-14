#!/bin/bash
set -euo pipefail

# Install Flash Attention 2 or 3 at runtime
#
# Usage:
#   ./scripts/install_flash_attn.sh              # Install FA2 (default)
#   ./scripts/install_flash_attn.sh --fa2        # Install FA2 explicitly
#   ./scripts/install_flash_attn.sh --fa3        # Install FA3 (requires Hopper GPU)
#   ./scripts/install_flash_attn.sh --version    # Show version info
#   FA_VERSION=3 ./scripts/install_flash_attn.sh # Install FA3 via env var
#
# Environment Variables:
#   FA_VERSION      - 2 or 3 (default: 2)
#   FA2_VERSION     - FA2 release version (default: 2.7.3)
#   FA3_COMMIT      - FA3 git commit hash (default: latest known working)
#   FA3_MAX_JOBS    - Parallel build jobs for FA3 (default: 8)
#   SKIP_TEST       - Skip import test after install (default: false)

# Defaults
FA_VERSION="${FA_VERSION:-2}"
FA2_VERSION="${FA2_VERSION:-2.7.3}"
FA3_COMMIT="${FA3_COMMIT:-92ca9da8d66f7b34ff50dc080ec0fef9661260d6}"
FA3_MAX_JOBS="${FA3_MAX_JOBS:-8}"
SKIP_TEST="${SKIP_TEST:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fa2|--FA2)
            FA_VERSION=2
            shift
            ;;
        --fa3|--FA3)
            FA_VERSION=3
            shift
            ;;
        --fa2-version)
            FA2_VERSION="$2"
            shift 2
            ;;
        --fa3-commit)
            FA3_COMMIT="$2"
            shift 2
            ;;
        --max-jobs)
            FA3_MAX_JOBS="$2"
            shift 2
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --version)
            echo "Flash Attention installer"
            echo "  FA2 default version: ${FA2_VERSION}"
            echo "  FA3 default commit:  ${FA3_COMMIT}"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fa2              Install Flash Attention 2 (default)"
            echo "  --fa3              Install Flash Attention 3 (Hopper GPUs only)"
            echo "  --fa2-version VER  FA2 version (default: ${FA2_VERSION})"
            echo "  --fa3-commit SHA   FA3 git commit (default: ${FA3_COMMIT})"
            echo "  --max-jobs N       Parallel build jobs for FA3 (default: ${FA3_MAX_JOBS})"
            echo "  --skip-test        Skip import test after install"
            echo "  --version          Show version info"
            echo "  --help             Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  FA_VERSION=2|3     Select FA version"
            echo "  FA2_VERSION        FA2 release version"
            echo "  FA3_COMMIT         FA3 git commit hash"
            echo "  FA3_MAX_JOBS       Parallel build jobs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Detect environment
echo "=== Detecting environment ==="
TORCH_VERSION=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:3] if torch.version.cuda else 'none')")
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
CXX11_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")

echo "  PyTorch:    ${TORCH_VERSION}"
echo "  CUDA:       ${CUDA_VERSION}"
echo "  Python:     ${PYTHON_VERSION}"
echo "  CXX11 ABI:  ${CXX11_ABI}"

if [[ "${CUDA_VERSION}" == "none" ]]; then
    echo "Error: CUDA not available. Flash Attention requires CUDA."
    exit 1
fi

install_fa2() {
    echo ""
    echo "=== Installing Flash Attention 2 v${FA2_VERSION} ==="

    # Construct wheel URL
    # Format: flash_attn-{version}+cu{cuda}torch{torch}cxx11abi{abi}-cp{py}-cp{py}-linux_x86_64.whl
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA2_VERSION}/flash_attn-${FA2_VERSION}+cu${CUDA_VERSION}torch${TORCH_VERSION}cxx11abi${CXX11_ABI}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-linux_x86_64.whl"

    echo "  Wheel URL: ${WHEEL_URL}"
    echo ""

    # Try to install from pre-built wheel
    if uv pip install "${WHEEL_URL}" --no-cache 2>/dev/null; then
        echo "Successfully installed FA2 from pre-built wheel"
    else
        echo "Pre-built wheel not available, falling back to source build..."
        echo "This may take 10-30 minutes depending on your system."
        uv pip install "flash-attn==${FA2_VERSION}" --no-build-isolation --no-cache
    fi
}

install_fa3() {
    echo ""
    echo "=== Installing Flash Attention 3 (commit: ${FA3_COMMIT}) ==="
    echo "Note: FA3 requires Hopper architecture GPUs (H100, H200)"
    echo ""

    # Check for Hopper GPU
    GPU_ARCH=$(python -c "import torch; print(torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
    if [[ "${GPU_ARCH}" != "9" && "${GPU_ARCH}" != "0" ]]; then
        echo "Warning: Detected GPU architecture ${GPU_ARCH}.x, FA3 requires 9.x (Hopper)"
        echo "FA3 may not work correctly on this GPU."
    fi

    # Create temp directory for build
    BUILD_DIR=$(mktemp -d)
    echo "  Build directory: ${BUILD_DIR}"

    pushd "${BUILD_DIR}" > /dev/null

    # Clone flash-attention repo
    echo "  Cloning flash-attention repository..."
    git clone --depth 1 --recurse-submodules --shallow-submodules \
        https://github.com/Dao-AILab/flash-attention.git

    cd flash-attention

    # Checkout specific commit
    echo "  Checking out commit ${FA3_COMMIT}..."
    git fetch --depth 1 origin "${FA3_COMMIT}"
    git checkout "${FA3_COMMIT}"
    git submodule update --init --recursive

    # Build FA3 from hopper directory
    echo "  Building FA3 (this may take 15-45 minutes)..."
    cd hopper

    # Build with FP16 disabled (as per OLMo-core)
    FLASH_ATTENTION_DISABLE_FP16=TRUE MAX_JOBS="${FA3_MAX_JOBS}" python setup.py install

    popd > /dev/null

    # Cleanup
    echo "  Cleaning up build directory..."
    rm -rf "${BUILD_DIR}"

    echo "Successfully installed FA3 from source"
}

# Install selected version
if [[ "${FA_VERSION}" == "2" ]]; then
    install_fa2
elif [[ "${FA_VERSION}" == "3" ]]; then
    install_fa3
else
    echo "Error: Invalid FA_VERSION '${FA_VERSION}'. Use 2 or 3."
    exit 1
fi

# Test import
if [[ "${SKIP_TEST}" != "true" ]]; then
    echo ""
    echo "=== Testing Flash Attention import ==="
    if python -c "import flash_attn; print(f'flash_attn version: {flash_attn.__version__}')" 2>/dev/null; then
        echo "Flash Attention installed successfully!"
    else
        echo "Warning: flash_attn import failed. Installation may be incomplete."
        exit 1
    fi
fi

echo ""
echo "=== Installation complete ==="

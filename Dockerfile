# OLMo Evaluation Framework Docker Image
#
# Base image with CUDA, Python, PyTorch, and Flash Attention 2 & 3.
# Backend dependencies (vllm, transformers, etc.) installed at runtime via gantry/uv.
#
# Build:
#   ./scripts/build_image.sh
#   ./scripts/build_image.sh --cuda-version 12.8.0
#   ./scripts/build_image.sh --platform linux/amd64
#
# Tags: cuda{ver}-{arch}
# Example: cuda128-amd64

# ============================================================================
# Build arguments
# ============================================================================
ARG CUDA_VERSION=12.8.1
ARG TORCH_VERSION=2.7.1
ARG PYTHON_VERSION=3.12
ARG FA2_VERSION=2.8.3
ARG FA3_COMMIT=92ca9da8d66f7b34ff50dc080ec0fef9661260d6

# ============================================================================
# Stage 1: Base builder with CUDA, Python and PyTorch
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Create virtual environment with specified Python version
RUN uv python install ${PYTHON_VERSION} && \
    uv venv /opt/venv --python ${PYTHON_VERSION}

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Install PyTorch with CUDA support
RUN CUDA_SHORT=$(echo "${CUDA_VERSION}" | sed 's/\.//g' | cut -c1-3) && \
    uv pip install "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}"

# ============================================================================
# Stage 2: Install Flash Attention 2 from pre-built wheel
# ============================================================================
FROM builder AS fa2-builder

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION
ARG FA2_VERSION

# Install FA2 from pre-built wheel (much faster than building from source)
# Format: flash_attn-{ver}+cu{cuda}torch{torch}cxx11abi{abi}-cp{py}-cp{py}-linux_x86_64.whl
RUN TORCH_SHORT=$(python -c "import torch; print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))") && \
    CUDA_SHORT=$(python -c "import torch; print(torch.version.cuda.replace('.', '')[:2])") && \
    PYTHON_V=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')") && \
    echo "Installing flash-attn ${FA2_VERSION} for CUDA ${CUDA_SHORT} PyTorch ${TORCH_SHORT} ABI ${ABI}" && \
    uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v${FA2_VERSION}/flash_attn-${FA2_VERSION}+cu${CUDA_SHORT}torch${TORCH_SHORT}cxx11abi${ABI}-cp${PYTHON_V}-cp${PYTHON_V}-linux_x86_64.whl" --no-cache

# Verify FA2 installation
RUN python -c "import flash_attn; print(f'flash_attn {flash_attn.__version__} installed successfully')"

# ============================================================================
# Stage 3: Build Flash Attention 3 wheel (Hopper GPUs)
# TEMPORARILY DISABLED - FA3 build has issues, re-enable when fixed
# ============================================================================
# FROM builder AS fa3-builder
#
# ARG FA3_COMMIT
#
# ENV FA3_MAX_JOBS=4
#
# WORKDIR /build
#
# # Clone and build FA3 wheel (no pre-built wheels available)
# # FLASH_ATTENTION_FORCE_BUILD=TRUE skips attempting to download pre-built wheels (which don't exist for this commit)
# RUN git clone --depth 1 --recurse-submodules --shallow-submodules \
#         https://github.com/Dao-AILab/flash-attention.git && \
#     cd flash-attention && \
#     git fetch --depth 1 origin ${FA3_COMMIT} && \
#     git checkout ${FA3_COMMIT} && \
#     git submodule update --init --recursive && \
#     cd hopper && \
#     uv pip install packaging ninja wheel setuptools && \
#     FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE MAX_JOBS=${FA3_MAX_JOBS} python setup.py bdist_wheel && \
#     cp dist/*.whl /build/

# ============================================================================
# Stage 4: Runtime image
# ============================================================================
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION
ARG FA2_VERSION
ARG FA3_COMMIT

LABEL org.opencontainers.image.source="https://github.com/allenai/olmo-eval-internal"
LABEL org.opencontainers.image.description="OLMo evaluation framework with Flash Attention 2"
LABEL cuda_version="${CUDA_VERSION}"
LABEL torch_version="${TORCH_VERSION}"
LABEL python_version="${PYTHON_VERSION}"
LABEL flash_attn_2_version="${FA2_VERSION}"
LABEL flash_attn_3_commit="${FA3_COMMIT}"

# Install runtime dependencies
# Clean up first to free space in the runtime image
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Copy virtual environment from fa2-builder (includes PyTorch + Flash Attention 2)
COPY --from=fa2-builder /opt/venv /opt/venv

# Copy uv resources
COPY --from=fa2-builder /root/.local/share/uv /root/.local/share/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy FA3 wheel for on-demand installation (TEMPORARILY DISABLED)
# COPY --from=fa3-builder /build/*.whl /opt/flash-attn-3/

# Set up environment
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
ENV VLLM_LOGGING_LEVEL=WARNING
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1
# ENV FA3_WHEEL_DIR=/opt/flash-attn-3  # TEMPORARILY DISABLED

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    python -c "import flash_attn; print(f'Flash Attention {flash_attn.__version__}')"

WORKDIR /workspace
CMD ["bash"]

# OLMo Evaluation Framework Docker Image
#
# Base image with CUDA, Python, and PyTorch.
# Backend dependencies (vllm, transformers, etc.) installed at runtime via gantry/uv.
#
# Build:
#   ./scripts/build_image.sh
#   ./scripts/build_image.sh --cuda-version 12.8.0
#   ./scripts/build_image.sh --platform linux/amd64
#
# Tags: cu{cuda}-trc{torch}-{arch}
# Example: cu128-trc291-amd64

# ============================================================================
# Build arguments
# ============================================================================
ARG CUDA_VERSION=12.8.1
ARG TORCH_VERSION=2.9.0
ARG PYTHON_VERSION=3.12

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

RUN uv pip install numpy packaging ninja wheel setuptools

# Install PyTorch with CUDA support
ARG INSTALL_CHANNEL=whl
RUN CUDA_SHORT=$(echo "${CUDA_VERSION}" | sed 's/\.//g' | cut -c1-3) && \
    uv pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/cu${CUDA_SHORT}/ \
    torch==${TORCH_VERSION}

# ============================================================================
# Stage 2: Runtime image
# ============================================================================
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION

LABEL org.opencontainers.image.source="https://github.com/allenai/olmo-eval-internal"
LABEL org.opencontainers.image.description="OLMo evaluation framework"
LABEL cuda_version="${CUDA_VERSION}"
LABEL torch_version="${TORCH_VERSION}"
LABEL python_version="${PYTHON_VERSION}"

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

# Copy virtual environment from builder (includes PyTorch)
COPY --from=builder /opt/venv /opt/venv

# Copy uv resources
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up environment
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
ENV VLLM_LOGGING_LEVEL=WARNING
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')"

WORKDIR /workspace
CMD ["bash"]

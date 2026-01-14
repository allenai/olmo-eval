# OLMo Evaluation Framework Docker Image
#
# Multi-stage build with configurable CUDA + PyTorch versions.
# Contains ONLY: Python, PyTorch, CUDA, and system dependencies.
# Everything else (source code, dependencies, backends) installed at runtime via gantry/uv.
#
# Build:
#   ./scripts/build_image.sh
#   ./scripts/build_image.sh --cuda-version 12.8.0 --torch-version 2.8.0
#   ./scripts/build_image.sh --platform linux/amd64
#
# Tags: cuda{ver}-torch{ver}-{arch}
# Example: cuda128-torch280-amd64

# ============================================================================
# Stage 1: Builder - Install PyTorch with CUDA support
# ============================================================================
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

# Install uv for fast package and Python management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Use uv to install Python 3.12
RUN uv python install 3.12

# Build arguments
ARG TORCH_VERSION=2.8.0
ARG CUDA_VERSION=12.8.0

# Create virtual environment
RUN echo "Installing Python 3.12 and creating virtual environment..." && \
    uv venv /opt/venv --python 3.12

# Install torch with CUDA support (only dependency in the image)
RUN echo "Installing PyTorch ${TORCH_VERSION} with CUDA ${CUDA_VERSION} support..." && \
    CUDA_SHORT=$(echo "${CUDA_VERSION}" | sed 's/\.//g' | cut -c1-3) && \
    uv pip install --python /opt/venv/bin/python \
        "torch==${TORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}"

# ============================================================================
# Stage 2: Runtime - Minimal production image
# ============================================================================
ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/allenai/olmo-eval-internal"
LABEL org.opencontainers.image.description="OLMo evaluation framework - minimal runtime"
ARG CUDA_VERSION=12.8.0
ARG TORCH_VERSION=2.8.0
LABEL cuda_version="${CUDA_VERSION}"
LABEL torch_version="${TORCH_VERSION}"

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment and uv-managed Python from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set Python path to use venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Set environment defaults
ENV VLLM_LOGGING_LEVEL=WARNING
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Working directory where gantry will mount source code
WORKDIR /workspace

# No ENTRYPOINT - gantry will handle command execution
CMD ["bash"]

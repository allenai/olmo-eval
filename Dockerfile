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
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

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
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04

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

# ============================================================================
# Stage 3: Runtime with Podman for sandboxed execution
# ============================================================================
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime-sandbox

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION

LABEL org.opencontainers.image.source="https://github.com/allenai/olmo-eval-internal"
LABEL org.opencontainers.image.description="OLMo evaluation framework with Podman sandbox support"
LABEL cuda_version="${CUDA_VERSION}"
LABEL torch_version="${TORCH_VERSION}"
LABEL python_version="${PYTHON_VERSION}"
LABEL sandbox_enabled="true"

# Install runtime dependencies + Podman build dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    # Podman build dependencies
    gcc \
    golang-go \
    go-md2man \
    iptables \
    libassuan-dev \
    libbtrfs-dev \
    libc6-dev \
    libdevmapper-dev \
    libglib2.0-dev \
    libgpgme-dev \
    libgpg-error-dev \
    libprotobuf-dev \
    libprotobuf-c-dev \
    libseccomp-dev \
    libselinux1-dev \
    libsystemd-dev \
    netavark \
    passt \
    pkg-config \
    uidmap \
    conmon \
    golang-github-containers-common \
    autoconf \
    automake \
    libtool \
    libcap-dev \
    libyajl-dev \
    systemd \
    python3-sphinx \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

# Configure container registries and policies
RUN mkdir -p /etc/containers/registries.conf.d/
COPY src/olmo_eval/launch/beaker/podman/containers.conf /etc/containers/containers.conf
COPY src/olmo_eval/launch/beaker/podman/policy.json /etc/containers/policy.json
COPY src/olmo_eval/launch/beaker/podman/10-unqualified-search-registries.conf /etc/containers/registries.conf.d/10-unqualified-search-registries.conf

# Build and install Podman from source
RUN wget -qO- https://github.com/containers/podman/archive/refs/tags/v5.6.2.tar.gz \
    | tar xz -C /tmp \
    && cd /tmp/podman-5.6.2 \
    && make BUILDTAGS="selinux seccomp" PREFIX=/usr \
    && make install PREFIX=/usr \
    && rm -rf /tmp/podman-5.6.2

# Build and install crun
RUN git clone --depth 1 -b 1.14.3 https://github.com/containers/crun.git /tmp/crun \
    && cd /tmp/crun \
    && ./autogen.sh \
    && ./configure --prefix=/usr --sysconfdir=/etc \
    && make \
    && make install \
    && rm -rf /tmp/crun

# Symlink so docker commands are translated to podman
RUN ln -sf $(which podman) /usr/local/bin/docker

# Set user namespace ranges
RUN echo "root:10000:11165536" >> /etc/subuid \
    && echo "root:10000:11165536" >> /etc/subgid

# Copy virtual environment from builder (includes PyTorch)
COPY --from=builder /opt/venv /opt/venv

# Copy uv resources
COPY --from=builder /root/.local/share/uv /root/.local/share/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set up environment
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

# Install podman-compose into venv
RUN uv pip install --no-cache-dir podman-compose

ENV VLLM_LOGGING_LEVEL=WARNING
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

# Verify installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}')" && \
    podman --version

WORKDIR /workspace
CMD ["bash"]

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

ARG CUDA_VERSION=12.8.1
ARG TORCH_VERSION=2.9.0
ARG PYTHON_VERSION=3.12
ARG INSTALL_CHANNEL=whl
ARG GIT_COMMIT=""
ARG GIT_BRANCH=""

# ============================================================================
# Stage 1: Builder — venv with PyTorch + lockfile-pinned project deps
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS builder

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION
ARG INSTALL_CHANNEL

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"

RUN uv python install ${PYTHON_VERSION} \
    && uv venv /opt/venv --python ${PYTHON_VERSION}

WORKDIR /opt/project
COPY pyproject.toml uv.lock README.md ./

# Base deps from lockfile (no extras, no default groups, no project — extras install at runtime).
RUN uv sync --frozen --active --no-default-groups --no-install-project

# PyTorch from the CUDA-specific PyTorch index (overrides any transitive CPU torch).
RUN CUDA_SHORT=$(echo "${CUDA_VERSION}" | sed 's/\.//g' | cut -c1-3) \
    && uv pip install --no-cache-dir \
        --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/cu${CUDA_SHORT}/ \
        torch==${TORCH_VERSION}

# ============================================================================
# Stage 2: Runtime — minimal image with venv + lockfile
# ============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS runtime

ARG CUDA_VERSION
ARG TORCH_VERSION
ARG PYTHON_VERSION
ARG GIT_COMMIT
ARG GIT_BRANCH

LABEL org.opencontainers.image.source="https://github.com/allenai/olmo-eval-internal"
LABEL org.opencontainers.image.description="OLMo evaluation framework"
LABEL cuda_version="${CUDA_VERSION}"
LABEL torch_version="${TORCH_VERSION}"
LABEL python_version="${PYTHON_VERSION}"
LABEL git_commit="${GIT_COMMIT}"
LABEL git_branch="${GIT_BRANCH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/project /opt/project
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uv

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
ENV VLLM_LOGGING_LEVEL=WARNING
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV GIT_COMMIT=${GIT_COMMIT}
ENV GIT_BRANCH=${GIT_BRANCH}

WORKDIR /workspace
CMD ["bash"]

# ============================================================================
# Stage 3: Runtime + Podman for sandboxed execution
# ============================================================================
FROM runtime AS runtime-sandbox

ARG GIT_COMMIT
ARG GIT_BRANCH

LABEL org.opencontainers.image.description="OLMo evaluation framework with Podman sandbox support"
LABEL git_commit="${GIT_COMMIT}"
LABEL git_branch="${GIT_BRANCH}"
LABEL sandbox_enabled="true"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gcc golang-go go-md2man iptables \
    libassuan-dev libbtrfs-dev libc6-dev libdevmapper-dev libglib2.0-dev \
    libgpgme-dev libgpg-error-dev libprotobuf-dev libprotobuf-c-dev \
    libseccomp-dev libselinux1-dev libsystemd-dev \
    netavark pkg-config uidmap conmon golang-github-containers-common \
    autoconf automake libtool libcap-dev libyajl-dev systemd python3-sphinx \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && apt-get clean

RUN mkdir -p /etc/containers/registries.conf.d/
COPY src/olmo_eval/launch/beaker/podman/containers.conf /etc/containers/containers.conf
COPY src/olmo_eval/launch/beaker/podman/policy.json /etc/containers/policy.json
COPY src/olmo_eval/launch/beaker/podman/10-unqualified-search-registries.conf /etc/containers/registries.conf.d/10-unqualified-search-registries.conf

# Podman from source
RUN wget -qO- https://github.com/containers/podman/archive/refs/tags/v5.6.2.tar.gz \
        | tar xz -C /tmp \
    && make -C /tmp/podman-5.6.2 BUILDTAGS="selinux seccomp" PREFIX=/usr \
    && make -C /tmp/podman-5.6.2 install PREFIX=/usr \
    && rm -rf /tmp/podman-5.6.2

# crun
RUN git clone --depth 1 -b 1.14.3 https://github.com/containers/crun.git /tmp/crun \
    && cd /tmp/crun && ./autogen.sh && ./configure --prefix=/usr --sysconfdir=/etc \
    && make && make install \
    && rm -rf /tmp/crun

# pasta (/dev/net/tun is created at runtime by the sandbox executor)
RUN wget -qO /usr/bin/passt https://passt.top/builds/latest/x86_64/passt \
    && chmod +x /usr/bin/passt \
    && ln -sf /usr/bin/passt /usr/bin/pasta

RUN ln -sf "$(which podman)" /usr/local/bin/docker \
    && echo "root:10000:11165536" >> /etc/subuid \
    && echo "root:10000:11165536" >> /etc/subgid

CMD ["bash"]

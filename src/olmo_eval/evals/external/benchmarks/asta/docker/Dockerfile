# ASTA-bench evaluation container image
# Build: docker build -t ghcr.io/allenai/olmo-eval-asta:latest -f src/olmo_eval/evals/external/benchmarks/asta/docker/Dockerfile .
# Run: docker run --rm -it ghcr.io/allenai/olmo-eval-asta:latest

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Clone ASTA-bench at specific version
# Using v0.3.1 for stability
RUN git clone --recursive --branch v0.3.1 \
    https://github.com/allenai/asta-bench.git

WORKDIR /workspace/asta-bench

# Install dependencies with uv
RUN uv sync

# Install swe-rex for sandbox communication
RUN uv pip install swe-rex

# Create results directory
RUN mkdir -p /workspace/asta-bench/results

# Add venv to PATH
ENV PATH="/workspace/asta-bench/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Default sandbox mode for Beaker compatibility
ENV INSPECT_SANDBOX=local
ENV INSPECT_EVAL_SANDBOX=local

# Default entrypoint
ENTRYPOINT ["/bin/bash", "-c"]
CMD ["echo 'ASTA-bench container ready. Use: uv run astabench eval --help'"]

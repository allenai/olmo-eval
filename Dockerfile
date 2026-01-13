# OLMo Evaluation Framework Docker Image
#
# Build:
#   ./scripts/build_image.sh
#
# Run:
#   docker run --rm olmo-eval:latest --help
#   docker run --rm --gpus all olmo-eval:latest models

ARG VLLM_VERSION=0.13.0
FROM vllm/vllm-openai:v${VLLM_VERSION}

LABEL org.opencontainers.image.source="https://github.com/allenai/olmo-eval-internal"
LABEL org.opencontainers.image.description="OLMo evaluation framework"

# Install uv for fast package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy package files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install olmo-eval with all optional dependencies
RUN uv pip install --system -e ".[all]"

# Set environment defaults for Beaker
ENV VLLM_LOGGING_LEVEL=WARNING
ENV HF_HOME=/root/.cache/huggingface

# Entry point - allows running olmo-eval commands directly
ENTRYPOINT ["olmo-eval"]
CMD ["--help"]

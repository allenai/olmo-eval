"""Utilities for building derived sandbox images with swe-rex."""

from __future__ import annotations

import hashlib
import logging
import subprocess

from olmo_eval.common.config import get_infra_config

logger = logging.getLogger(__name__)

# Python standalone URL for building derived images
PYTHON_STANDALONE_URL = (
    "https://github.com/indygreg/python-build-standalone/releases/download/"
    "20240107/cpython-3.11.7+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz"
)

# Version bump this when changing the Dockerfile to invalidate cached images
SWEREX_IMAGE_VERSION = "20260304.1"


def get_swerex_image(
    base_image: str,
    container_runtime: str = "docker",
    dockerfile_extra: tuple[str, ...] = (),
) -> str:
    """Build a derived image with Python and swe-rex pre-installed.

    Checks local cache first, then registry (if configured), then builds and pushes.

    Args:
        base_image: The base container image.
        container_runtime: Container runtime (docker or podman).
        dockerfile_extra: Additional Dockerfile commands to inject.

    Returns:
        The derived image name with swe-rex installed.
    """
    config = get_infra_config()
    registry = config.swerex_registry

    # Create a deterministic tag based on base image, Python URL, version, and extra commands
    extra_hash = ":".join(dockerfile_extra) if dockerfile_extra else ""
    hash_input = f"{base_image}:{PYTHON_STANDALONE_URL}:{SWEREX_IMAGE_VERSION}:{extra_hash}"
    tag_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # Local image name
    local_image = f"swerex-{tag_hash}:latest"

    # Check if image exists locally
    result = subprocess.run(
        [container_runtime, "image", "inspect", local_image],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.info(f"Using cached swerex image: {local_image}")
        return local_image

    logger.debug(f"Local image {local_image} not found, checking registry...")

    # Try to pull from registry (if configured)
    if registry:
        registry_image = f"{registry}/swerex-{tag_hash}:latest"
        result = subprocess.run(
            [container_runtime, "pull", registry_image],
            capture_output=True,
        )
        if result.returncode == 0:
            # Tag locally for consistency
            subprocess.run(
                [container_runtime, "tag", registry_image, local_image],
                capture_output=True,
            )
            logger.info(f"Pulled swerex image from registry: {local_image}")
            return local_image
        logger.debug("Registry pull failed, will build locally")

    # Build the image with Python, swe-rex, curl, git, and uv
    logger.info(f"Building swerex image from {base_image}...")

    extra_lines = "\n".join(dockerfile_extra) if dockerfile_extra else ""

    dockerfile = f"""\
FROM {base_image}
USER root
# Disable apt sandboxing to avoid setgroups/setegid errors in rootless containers
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/99-disable-sandbox
RUN apt-get update && \\
    apt-get install -y --no-install-recommends curl git ca-certificates && \\
    rm -rf /var/lib/apt/lists/*
ADD {PYTHON_STANDALONE_URL} /tmp/python.tar.gz
RUN tar xzf /tmp/python.tar.gz -C /root && rm /tmp/python.tar.gz && \\
    /root/python/bin/pip install --no-cache-dir swe-rex uv
{extra_lines}
ENV PATH="/root/python/bin:$PATH"
"""

    result = subprocess.run(
        [container_runtime, "build", "-t", local_image, "-"],
        input=dockerfile.encode(),
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode() if result.stderr else ""
        raise RuntimeError(f"Failed to build swerex image: {stderr}")

    logger.info(f"Built swerex image: {local_image}")

    if registry:
        registry_image = f"{registry}/swerex-{tag_hash}:latest"
        logger.info(f"Pushing swerex image to registry: {registry_image}")
        subprocess.run([container_runtime, "tag", local_image, registry_image], capture_output=True)
        push_result = subprocess.run(
            [container_runtime, "push", registry_image], capture_output=True
        )
        if push_result.returncode == 0:
            logger.info(f"Pushed swerex image to registry: {registry_image}")
        else:
            stderr = push_result.stderr.decode() if push_result.stderr else ""
            logger.warning(f"Failed to push to registry (using local image): {stderr}")

    return local_image

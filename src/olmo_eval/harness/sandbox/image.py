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
    require_registry: bool = False,
) -> str:
    """Build a derived image with Python and swe-rex pre-installed.

    Checks local cache first, then registry (if configured), then builds and pushes.

    Args:
        base_image: The base container image.
        container_runtime: Container runtime (docker or podman).
        dockerfile_extra: Additional Dockerfile commands to inject.
        require_registry: If True, requires SWEREX_REGISTRY and returns registry URL.
            Use for Modal which needs remote-accessible images.

    Returns:
        The derived image name with swe-rex installed. If require_registry=True,
        returns the registry URL.

    Raises:
        ValueError: If require_registry=True but SWEREX_REGISTRY is not set.
        RuntimeError: If image build or push fails.
    """
    config = get_infra_config()
    registry = config.swerex_registry

    if require_registry and not registry:
        raise ValueError(
            "SWEREX_REGISTRY required for Modal deployments with inject_swerex=True. "
            "Set to your registry URL (e.g., 'us-docker.pkg.dev/project/repo')."
        )

    # Create a deterministic tag based on base image, Python URL, version, and extra commands
    extra_hash = ":".join(dockerfile_extra) if dockerfile_extra else ""
    hash_input = f"{base_image}:{PYTHON_STANDALONE_URL}:{SWEREX_IMAGE_VERSION}:{extra_hash}"
    tag_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]

    # Local image name
    local_image = f"swerex-{tag_hash}:latest"

    # Registry image name (if registry configured)
    registry_image = f"{registry}/swerex-{tag_hash}:latest" if registry else None

    # Check if image exists locally
    result = subprocess.run(
        [container_runtime, "image", "inspect", local_image],
        capture_output=True,
    )
    if result.returncode == 0:
        logger.debug(f"Using cached swerex image: {local_image}")
        if require_registry:
            # Ensure image is pushed to registry for remote access
            assert registry_image is not None  # Guaranteed by earlier check
            subprocess.run(
                [container_runtime, "tag", local_image, registry_image],
                capture_output=True,
            )
            push_result = subprocess.run(
                [container_runtime, "push", registry_image], capture_output=True
            )
            if push_result.returncode != 0:
                stderr = push_result.stderr.decode() if push_result.stderr else ""
                raise RuntimeError(f"Failed to push to registry: {stderr}")
            logger.debug(f"Pushed swerex image to registry: {registry_image}")
            return registry_image
        return local_image

    logger.debug(f"Local image {local_image} not found, checking registry...")

    # Try to pull from registry (if configured)
    if registry and registry_image:
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
            logger.debug(f"Pulled swerex image from registry: {local_image}")
            return registry_image if require_registry else local_image
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

    if registry and registry_image:
        logger.debug(f"Pushing swerex image to registry: {registry_image}")
        subprocess.run([container_runtime, "tag", local_image, registry_image], capture_output=True)
        push_result = subprocess.run(
            [container_runtime, "push", registry_image], capture_output=True
        )
        if push_result.returncode == 0:
            logger.debug(f"Pushed swerex image to registry: {registry_image}")
            if require_registry:
                return registry_image
        else:
            stderr = push_result.stderr.decode() if push_result.stderr else ""
            if require_registry:
                raise RuntimeError(f"Failed to push to registry: {stderr}")
            logger.warning(f"Failed to push to registry (using local image): {stderr}")

    # When require_registry=True, we've already returned above on success or raised on failure
    # So reaching here with require_registry=True means something unexpected happened
    if require_registry:
        raise RuntimeError("Failed to push image to registry (unexpected state)")

    return local_image

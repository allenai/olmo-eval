"""Network utilities for external evaluations."""

from __future__ import annotations

import os
import re
import subprocess
from urllib.parse import urlparse, urlunparse

from olmo_eval.common.config import get_infra_config

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_PASTA_HOST_IP = "169.254.1.2"
DEFAULT_DOCKER0_GATEWAY = "172.17.0.1"


def get_docker_network_args(runtime: str | None = None) -> tuple[str, ...]:
    """Get Docker/Podman args for network configuration.

    Args:
        runtime: Container runtime to use. If None, uses config default.

    Returns:
        Tuple of docker args for network configuration.
    """
    config = get_infra_config()
    runtime = runtime or config.container_runtime

    if runtime == "docker":
        # Docker needs explicit host gateway mapping
        return ("--add-host=host.docker.internal:host-gateway",)

    # Podman: use pasta with --map-guest-addr for fixed host IP access
    # Also pass --dns to use Google DNS, avoiding pasta's TCP DNS timeout issues
    return (
        f"--network=pasta:--map-guest-addr,{config.pasta_host_ip}",
        "--dns=8.8.8.8",
    )


def resolve_container_runtime(runtime: str | None = None) -> str:
    """Resolve docker vs podman, defaulting to podman to match ExternalEval."""
    value = runtime or os.environ.get("OLMO_CONTAINER_RUNTIME") or "podman"
    return value.strip().lower()


def get_workspace_gateway_ip(runtime: str | None = None) -> str:
    """Host IP reachable from an inner workspace container.

    Podman pasta maps the host at ``OLMO_PASTA_HOST_IP`` (default
    ``169.254.1.2``). Docker uses the docker0 bridge (typically
    ``172.17.0.1``).
    """
    if resolve_container_runtime(runtime) == "docker":
        return _detect_docker0_gateway() or DEFAULT_DOCKER0_GATEWAY
    return os.environ.get("OLMO_PASTA_HOST_IP") or get_infra_config().pasta_host_ip


def rewrite_loopback_url(
    url: str,
    runtime: str | None = None,
    gateway_ip: str | None = None,
) -> str:
    """Rewrite loopback hosts so a workspace container can reach the host.

    Preserves scheme, port, and path. Non-loopback URLs are returned unchanged.
    """
    if not url:
        return url
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").strip("[]")
    if hostname not in LOOPBACK_HOSTS:
        return url
    gateway = gateway_ip or get_workspace_gateway_ip(runtime)
    new_netloc = f"{gateway}:{parsed.port}" if parsed.port else gateway
    return urlunparse(
        (parsed.scheme, new_netloc, parsed.path, parsed.params, parsed.query, parsed.fragment)
    )


def _detect_docker0_gateway() -> str | None:
    """Return the IPv4 address on docker0, if the interface exists."""
    try:
        result = subprocess.run(
            ["ip", "-4", "addr", "show", "docker0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
    return match.group(1) if match else None

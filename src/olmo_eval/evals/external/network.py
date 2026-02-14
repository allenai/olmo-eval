"""Network utilities for external evaluations."""

from __future__ import annotations

# slirp4netns gateway IP - this is the fixed gateway for podman's slirp4netns networking
SLIRP4NETNS_GATEWAY = "10.0.2.2"


def get_docker_network_args(runtime: str = "podman") -> tuple[str, ...]:
    """Get Docker/Podman args for network configuration.

    Args:
        runtime: Container runtime to use.

    Returns:
        Tuple of docker args for network configuration.
    """
    if runtime == "docker":
        # Docker needs explicit host gateway mapping
        return ("--add-host=host.docker.internal:host-gateway",)

    # Podman: use slirp4netns with host loopback access
    # This allows the container to reach the host at 10.0.2.2
    return ("--network=slirp4netns:allow_host_loopback=true",)

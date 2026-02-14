"""Network utilities for external evaluations."""

from __future__ import annotations

# slirp4netns gateway IP - fixed address to reach the host from inside container
SLIRP4NETNS_HOST_IP = "10.0.2.2"


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

    # Podman: use slirp4netns with allow_host_loopback for host access
    # The host is reachable at 10.0.2.2 from inside the container
    return ("--network=slirp4netns:allow_host_loopback=true",)

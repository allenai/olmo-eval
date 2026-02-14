"""Network utilities for external evaluations."""

from __future__ import annotations

# Pasta uses --map-guest-addr to provide a fixed IP for host access
# This is the default in Podman 5.3+ for host.containers.internal
PASTA_HOST_IP = "169.254.1.2"


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

    # Podman: use pasta networking with --map-guest-addr for host access
    # This maps host.containers.internal to a fixed IP (169.254.1.2)
    return (f"--network=pasta:--map-guest-addr,{PASTA_HOST_IP}",)

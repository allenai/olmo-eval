"""Network utilities for external evaluations."""

from __future__ import annotations

# Pasta host IP - fixed address to reach the host from inside container
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

    # Podman: use pasta with --map-guest-addr for fixed host IP access
    return (f"--network=pasta:--map-guest-addr,{PASTA_HOST_IP}",)

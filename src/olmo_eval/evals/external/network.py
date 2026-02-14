"""Network utilities for external evaluations."""

from __future__ import annotations


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

    # Podman: no special args needed, gateway discovered dynamically
    return ()

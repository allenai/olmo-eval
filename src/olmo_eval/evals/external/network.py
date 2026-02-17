"""Network utilities for external evaluations."""

from __future__ import annotations

from olmo_eval.common.config import get_infra_config


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
    return (f"--network=pasta:--map-guest-addr,{config.pasta_host_ip}",)

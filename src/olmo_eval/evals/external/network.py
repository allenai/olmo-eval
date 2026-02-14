"""Network utilities for external evaluations.

Handles translation of provider URLs between host and container contexts.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse


def get_host_ip_for_slirp4netns() -> str:
    """Get the host IP for podman's slirp4netns networking.

    When running podman in rootless mode or inside a container, it uses
    slirp4netns for networking. The gateway to the host is at 10.0.2.2.

    Returns:
        Gateway IP for slirp4netns (10.0.2.2).
    """
    return "10.0.2.2"


def should_use_host_network() -> bool:
    """Check if containers should use host networking.

    Sandboxes are intended to be isolated and should not have access to host
    networking. We always use bridge networking and connect via IP address.
    """
    return False


def translate_url_for_container(
    url: str,
    runtime: str = "podman",
    use_host_network: bool = False,
) -> str:
    """Translate a localhost URL for access from within a container.

    Args:
        url: Original URL (e.g., "http://localhost:8000").
        runtime: Container runtime ("docker" or "podman").
        use_host_network: If True, container uses --network=host and
            localhost works directly.

    Returns:
        Translated URL accessible from within the container.
    """
    parsed = urlparse(url)

    # If using network=host, localhost works directly
    if use_host_network:
        return url

    # Only translate localhost URLs
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return url

    # Determine the host to use based on runtime
    if runtime == "docker":
        # Docker supports host.docker.internal
        new_host = "host.docker.internal"
    else:
        # Podman uses slirp4netns networking where the gateway is 10.0.2.2
        # On macOS, podman also supports host.docker.internal
        import platform

        new_host = (
            "host.docker.internal"
            if platform.system() == "Darwin"
            else get_host_ip_for_slirp4netns()
        )

    # Reconstruct the URL with the new host
    new_netloc = new_host
    if parsed.port:
        new_netloc = f"{new_host}:{parsed.port}"

    return urlunparse(
        (
            parsed.scheme,
            new_netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def get_provider_url_for_container(
    base_url: str,
    runtime: str = "podman",
    use_host_network: bool | None = None,
) -> str:
    """Get the provider URL translated for container access.

    Args:
        base_url: Base URL of the inference provider.
        runtime: Container runtime to use.
        use_host_network: Whether to use host networking. Auto-detected if None.

    Returns:
        URL accessible from within the container.
    """
    if use_host_network is None:
        use_host_network = should_use_host_network()

    return translate_url_for_container(
        base_url,
        runtime=runtime,
        use_host_network=use_host_network,
    )


def get_docker_network_args(
    runtime: str = "podman",
    use_host_network: bool = False,
) -> tuple[str, ...]:
    """Get Docker/Podman args for network configuration.

    Args:
        runtime: Container runtime to use.
        use_host_network: Whether to use host networking.

    Returns:
        Tuple of docker args for network configuration.
    """
    if use_host_network:
        return ("--network=host",)

    if runtime == "docker":
        # Docker needs explicit host gateway mapping
        return ("--add-host=host.docker.internal:host-gateway",)

    # Podman typically handles this automatically on macOS
    # On Linux, we connect via host IP so no special args needed
    return ()

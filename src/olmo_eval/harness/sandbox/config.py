"""Configuration for sandboxed tool execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

ContainerRuntime = Literal["docker", "podman"]


class SandboxMode(StrEnum):
    """Sandbox deployment modes."""

    LOCAL = "local"
    DOCKER = "docker"
    MODAL = "modal"


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for sandboxed tool execution via SWE-ReX.

    Attributes:
        image: Container image for the sandbox environment.
        mode: How to run the sandbox.
        startup_timeout: Timeout for container startup in seconds.
        command_timeout: Default timeout for command execution in seconds.
        remove_container: Whether to remove container after use.
        working_dir: Working directory inside the container.
        environment: Environment variables as tuple of (name, value) pairs.
        volumes: Volume mounts as tuple of (host_path, container_path) pairs.
        modal_sandbox_kwargs: Additional kwargs for Modal sandbox configuration.
        runtime_timeout: Timeout for Modal runtime in seconds.
        required_secrets: Environment variable names that must be set.
    """

    image: str
    mode: SandboxMode
    container_runtime: ContainerRuntime = "podman"
    startup_timeout: float = 60.0
    command_timeout: float = 30.0
    remove_container: bool = True
    working_dir: str = "/workspace"
    environment: tuple[tuple[str, str], ...] = ()
    volumes: tuple[tuple[str, str], ...] = ()
    modal_sandbox_kwargs: dict[str, Any] | None = None
    runtime_timeout: float = 3600.0
    required_secrets: tuple[str, ...] = ()
    docker_args: tuple[str, ...] = ()
    image_cache_dir: str | None = None  # Custom container storage location for image caching

    @property
    def is_local(self) -> bool:
        """True if sandbox runs locally (docker/local), False if remote (modal)."""
        return self.mode in (SandboxMode.LOCAL, SandboxMode.DOCKER)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "image": self.image,
            "mode": self.mode.value,
            "container_runtime": self.container_runtime,
            "startup_timeout": self.startup_timeout,
            "command_timeout": self.command_timeout,
            "remove_container": self.remove_container,
            "working_dir": self.working_dir,
            "environment": list(self.environment),
            "volumes": list(self.volumes),
            "runtime_timeout": self.runtime_timeout,
            "docker_args": list(self.docker_args),
        }
        if self.image_cache_dir is not None:
            result["image_cache_dir"] = self.image_cache_dir
        if self.modal_sandbox_kwargs is not None:
            result["modal_sandbox_kwargs"] = self.modal_sandbox_kwargs
        if self.required_secrets:
            result["required_secrets"] = list(self.required_secrets)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxConfig:
        """Create from dictionary."""
        if "image" not in data:
            raise ValueError("SandboxConfig requires 'image' to be specified")
        if "mode" not in data:
            raise ValueError("SandboxConfig requires 'mode' to be specified")
        return cls(
            image=data["image"],
            mode=SandboxMode(data["mode"]),
            container_runtime=data.get("container_runtime", "podman"),
            startup_timeout=data.get("startup_timeout", 60.0),
            command_timeout=data.get("command_timeout", 30.0),
            remove_container=data.get("remove_container", True),
            working_dir=data.get("working_dir", "/workspace"),
            environment=tuple(tuple(e) for e in data.get("environment", [])),
            volumes=tuple(tuple(v) for v in data.get("volumes", [])),
            modal_sandbox_kwargs=data.get("modal_sandbox_kwargs"),
            runtime_timeout=data.get("runtime_timeout", 3600.0),
            required_secrets=tuple(data.get("required_secrets", [])),
            docker_args=tuple(data.get("docker_args", [])),
            image_cache_dir=data.get("image_cache_dir"),
        )


def build_storage_conf(graphroot: Path, runroot: Path) -> str:
    """Build containers storage.conf content.

    Args:
        graphroot: Directory for container image storage.
        runroot: Directory for runtime state (locks, temp files).

    Returns:
        Storage configuration file content.

    Environment variables:
        SANDBOX_OVERLAY_MOUNT_PROGRAM: Path to fuse-overlayfs binary for
            network filesystem compatibility (e.g., /usr/bin/fuse-overlayfs).
    """
    lines = [
        "[storage]",
        'driver = "overlay"',
        f'graphroot = "{graphroot}"',
        f'runroot = "{runroot}"',
    ]

    mount_program = os.environ.get("SANDBOX_OVERLAY_MOUNT_PROGRAM")
    if mount_program:
        lines.extend(
            [
                "",
                "[storage.options.overlay]",
                f'mount_program = "{mount_program}"',
            ]
        )

    return "\n".join(lines) + "\n"

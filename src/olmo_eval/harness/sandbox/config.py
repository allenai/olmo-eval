"""Configuration for sandboxed tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


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
    """

    image: str
    mode: SandboxMode
    startup_timeout: float = 60.0
    command_timeout: float = 30.0
    remove_container: bool = True
    working_dir: str = "/workspace"
    environment: tuple[tuple[str, str], ...] = ()
    volumes: tuple[tuple[str, str], ...] = ()
    modal_sandbox_kwargs: dict[str, Any] | None = None
    runtime_timeout: float = 3600.0

    @property
    def is_local(self) -> bool:
        """True if sandbox runs locally (docker/local), False if remote (modal)."""
        return self.mode in (SandboxMode.LOCAL, SandboxMode.DOCKER)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "image": self.image,
            "mode": self.mode.value,
            "startup_timeout": self.startup_timeout,
            "command_timeout": self.command_timeout,
            "remove_container": self.remove_container,
            "working_dir": self.working_dir,
            "environment": list(self.environment),
            "volumes": list(self.volumes),
            "runtime_timeout": self.runtime_timeout,
        }
        if self.modal_sandbox_kwargs is not None:
            result["modal_sandbox_kwargs"] = self.modal_sandbox_kwargs
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
            startup_timeout=data.get("startup_timeout", 60.0),
            command_timeout=data.get("command_timeout", 30.0),
            remove_container=data.get("remove_container", True),
            working_dir=data.get("working_dir", "/workspace"),
            environment=tuple(tuple(e) for e in data.get("environment", [])),
            volumes=tuple(tuple(v) for v in data.get("volumes", [])),
            modal_sandbox_kwargs=data.get("modal_sandbox_kwargs"),
            runtime_timeout=data.get("runtime_timeout", 3600.0),
        )

"""Configuration for sandboxed tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

ContainerRuntime = Literal["docker", "podman"]


class SandboxMode(StrEnum):
    """Sandbox deployment modes."""

    LOCAL = "local"
    DOCKER = "docker"
    MODAL = "modal"


class Capability:
    """Standard sandbox capability identifiers.

    Example:
        @registered_tool(sandbox=Capability.BASH)
        async def execute_bash(command: str) -> str:
            ...
    """

    BASH: frozenset[str] = frozenset({"bash"})
    PYTHON: frozenset[str] = frozenset({"python"})
    DEFAULT: frozenset[str] = BASH


@dataclass(frozen=True)
class SandboxConfig:
    """Configuration for sandboxed tool execution via SWE-ReX.

    Attributes:
        image: Container image for the sandbox environment.
        mode: How to run the sandbox.
        capabilities: Capabilities this sandbox provides (e.g., {"bash"}).
        instances: Number of executor instances to create from this config.
            Multiple instances enable higher throughput via round-robin.
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
    capabilities: frozenset[str] = Capability.DEFAULT
    instances: int = 1
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
    log_dir: str | None = None

    @property
    def is_local(self) -> bool:
        """True if sandbox runs locally (docker/local), False if remote (modal)."""
        return self.mode in (SandboxMode.LOCAL, SandboxMode.DOCKER)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        from dataclasses import asdict

        result = asdict(self)
        result["mode"] = self.mode.value
        result["capabilities"] = sorted(self.capabilities)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SandboxConfig:
        """Create from dictionary."""
        if "image" not in data:
            raise ValueError("SandboxConfig requires 'image' to be specified")
        if "mode" not in data:
            raise ValueError("SandboxConfig requires 'mode' to be specified")
        capabilities = data.get("capabilities")
        return cls(
            image=data["image"],
            mode=SandboxMode(data["mode"]),
            capabilities=frozenset(capabilities) if capabilities else Capability.DEFAULT,
            instances=data.get("instances", 1),
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
            log_dir=data.get("log_dir"),
        )

"""Configuration for external black-box evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExternalEvalConfig:
    """Configuration for an external evaluation that runs in a sandbox container.

    External evaluations are "black box" benchmarks that install themselves
    (clone repo, pip install, etc.) and communicate with a model provider
    via an OpenAI-compatible API.

    Attributes:
        name: Unique identifier for this evaluation.
        sandbox_image: Container image to use for the sandbox.
        setup_commands: Commands to run during setup (e.g., clone repo, pip install).
        run_command: CLI command to execute the evaluation.
        timeout: Maximum execution time in seconds.
        api_base_env_var: Environment variable name for the API base URL.
        model_env_var: Environment variable name for the model identifier.
        required_secrets: Environment variable names that must be forwarded.
        working_dir: Working directory inside the container.
        environment: Additional environment variables as (name, value) pairs.
    """

    name: str
    sandbox_image: str = "python:3.11"
    setup_commands: tuple[str, ...] = ()
    run_command: str = ""
    timeout: float = 3600.0
    api_base_env_var: str = "OPENAI_API_BASE"
    model_env_var: str = "OPENAI_MODEL"
    required_secrets: tuple[str, ...] = ()
    working_dir: str = "/workspace"
    environment: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "sandbox_image": self.sandbox_image,
            "setup_commands": list(self.setup_commands),
            "run_command": self.run_command,
            "timeout": self.timeout,
            "api_base_env_var": self.api_base_env_var,
            "model_env_var": self.model_env_var,
            "required_secrets": list(self.required_secrets),
            "working_dir": self.working_dir,
            "environment": [list(e) for e in self.environment],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalEvalConfig:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            sandbox_image=data.get("sandbox_image", "python:3.11"),
            setup_commands=tuple(data.get("setup_commands", [])),
            run_command=data.get("run_command", ""),
            timeout=data.get("timeout", 3600.0),
            api_base_env_var=data.get("api_base_env_var", "OPENAI_API_BASE"),
            model_env_var=data.get("model_env_var", "OPENAI_MODEL"),
            required_secrets=tuple(data.get("required_secrets", [])),
            working_dir=data.get("working_dir", "/workspace"),
            environment=tuple(tuple(e) for e in data.get("environment", [])),
        )


@dataclass
class ExternalEvalRunConfig:
    """Runtime configuration for executing external evaluations.

    Attributes:
        eval_names: List of external evaluation names to run.
        output_dir: Directory to write results.
        provider_port: Port for the inference provider server.
        use_network_host: Whether to use --network=host for container.
        container_runtime: Container runtime to use (docker or podman).
    """

    eval_names: list[str] = field(default_factory=list)
    output_dir: str = "/results"
    provider_port: int = 8000
    use_network_host: bool = False
    container_runtime: str = "podman"

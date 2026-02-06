"""HarnessConfig: Immutable configuration describing model capabilities.

The configuration uses tool names (not Tool objects) for serialization.
Tools are resolved from the global registry at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from olmo_eval.core.types import ToolSchema

    from .tool import Tool


class HarnessBackend(StrEnum):
    """Type of backend for harness execution."""

    INTERNAL = "internal"
    OPENAI_AGENTS = "openai_agents"


@dataclass(frozen=True)
class HarnessConfig:
    """Immutable configuration describing model capabilities.

    This configuration determines how a Harness wraps a provider:
    - Which tools are available (by name, resolved from registry)
    - System prompt to prepend to requests
    - Tool choice behavior (auto, none, required)
    - Multi-turn settings (max_turns, max_concurrency)
    - Backend selection (internal, openai_agents)

    The configuration is serializable (uses tool names, not Tool objects)
    for passing across process boundaries.

    Attributes:
        name: Human-readable name for this harness configuration.
        tool_names: Names of tools from the registry to enable.
        system_prompt: System prompt to prepend to requests.
        tool_choice: How the model should use tools ("auto", "none", "required", or tool name).
        max_turns: Maximum turns in multi-turn execution.
        max_concurrency: Maximum concurrent tool executions.
        backend: Backend for agent execution ("internal", "openai_agents").
        model_url: API endpoint URL (for external backends).
        api_key: API key (for external backends).
        required_secrets: Environment variable names that must be set.
    """

    name: str
    tool_names: tuple[str, ...] = ()
    system_prompt: str | None = None
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    max_turns: int = 10
    max_concurrency: int = 8
    backend: HarnessBackend = HarnessBackend.INTERNAL
    # For API-based backends
    model_url: str | None = None
    api_key: str | None = None
    # Secrets validation
    required_secrets: tuple[str, ...] = ()

    @property
    def tools(self) -> tuple[Tool, ...]:
        """Resolve tools from the registry.

        Returns:
            Tuple of Tool instances corresponding to tool_names.

        Raises:
            ValueError: If any tool name is not registered.
        """
        from .tools import get_tools

        return get_tools(self.tool_names)

    @property
    def tool_schemas(self) -> tuple[ToolSchema, ...]:
        """Get just the schemas for LLM requests.

        Returns:
            Tuple of ToolSchema instances for all configured tools.
        """
        return tuple(t.schema for t in self.tools)

    @property
    def has_tools(self) -> bool:
        """Check if this configuration has any tools enabled.

        Returns:
            True if at least one tool is configured.
        """
        return len(self.tool_names) > 0

    def validate_secrets(self) -> list[str]:
        """Check that all required secrets are available.

        Returns:
            List of missing secret names (empty if all present).
        """
        import os

        missing = []
        for secret in self.required_secrets:
            if not os.getenv(secret):
                missing.append(secret)
        return missing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "name": self.name,
            "tool_names": list(self.tool_names),
            "system_prompt": self.system_prompt,
            "tool_choice": self.tool_choice,
            "max_turns": self.max_turns,
            "max_concurrency": self.max_concurrency,
            "backend": self.backend.value
            if isinstance(self.backend, HarnessBackend)
            else self.backend,
            "model_url": self.model_url,
            "api_key": self.api_key,
            "required_secrets": list(self.required_secrets),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarnessConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with HarnessConfig data.

        Returns:
            A new HarnessConfig instance.
        """
        return cls(
            name=data.get("name", "default"),
            tool_names=tuple(data.get("tool_names", [])),
            system_prompt=data.get("system_prompt"),
            tool_choice=data.get("tool_choice", "auto"),
            max_turns=data.get("max_turns", 10),
            max_concurrency=data.get("max_concurrency", 8),
            backend=HarnessBackend(data.get("backend", "internal")),
            model_url=data.get("model_url"),
            api_key=data.get("api_key"),
            required_secrets=tuple(data.get("required_secrets", [])),
        )

    def with_tools(self, *tools: Tool | str) -> HarnessConfig:
        """Create a new config with additional tools.

        Args:
            *tools: Tool instances or names to add.

        Returns:
            New HarnessConfig with the additional tools.
        """
        from .tools import register_tool

        new_names = list(self.tool_names)
        for t in tools:
            if isinstance(t, str):
                new_names.append(t)
            else:
                register_tool(t)
                new_names.append(t.name)

        return HarnessConfig(
            name=self.name,
            tool_names=tuple(new_names),
            system_prompt=self.system_prompt,
            tool_choice=self.tool_choice,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
            backend=self.backend,
            model_url=self.model_url,
            api_key=self.api_key,
            required_secrets=self.required_secrets,
        )

    def with_system_prompt(self, system_prompt: str) -> HarnessConfig:
        """Create a new config with a different system prompt.

        Args:
            system_prompt: The new system prompt to use.

        Returns:
            New HarnessConfig with the updated system prompt.
        """
        return HarnessConfig(
            name=self.name,
            tool_names=self.tool_names,
            system_prompt=system_prompt,
            tool_choice=self.tool_choice,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
            backend=self.backend,
            model_url=self.model_url,
            api_key=self.api_key,
            required_secrets=self.required_secrets,
        )


def harness_config(
    name: str,
    tools: Sequence[Tool | str] = (),
    system_prompt: str | None = None,
    tool_choice: Literal["auto", "none", "required"] | str = "auto",
    max_turns: int = 10,
    max_concurrency: int = 8,
    backend: HarnessBackend = HarnessBackend.INTERNAL,
    model_url: str | None = None,
    api_key: str | None = None,
    required_secrets: Sequence[str] = (),
) -> HarnessConfig:
    """Create a HarnessConfig, registering any Tool objects passed.

    This is a convenience function that accepts either Tool instances
    or tool names. Tool instances are automatically registered.

    Args:
        name: Human-readable name for this configuration.
        tools: Sequence of Tool instances or tool names.
        system_prompt: System prompt to prepend to requests.
        tool_choice: How the model should use tools.
        max_turns: Maximum turns in multi-turn execution.
        max_concurrency: Maximum concurrent tool executions.
        backend: Backend for agent execution.
        model_url: API endpoint URL.
        api_key: API key.
        required_secrets: Environment variable names that must be set.

    Returns:
        A new HarnessConfig instance with tools registered.
    """
    from .tool import Tool
    from .tools import register_tool

    tool_names: list[str] = []
    for t in tools:
        if isinstance(t, Tool):
            register_tool(t)
            tool_names.append(t.name)
        else:
            tool_names.append(t)

    return HarnessConfig(
        name=name,
        tool_names=tuple(tool_names),
        system_prompt=system_prompt,
        tool_choice=tool_choice,
        max_turns=max_turns,
        max_concurrency=max_concurrency,
        backend=backend,
        model_url=model_url,
        api_key=api_key,
        required_secrets=tuple(required_secrets),
    )

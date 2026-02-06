"""HarnessConfig: Immutable configuration describing model capabilities.

The configuration uses tool names (not Tool objects) for serialization.
Tools are resolved from the global registry at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from olmo_eval.core.types import ToolSchema

    from .tools import Tool


@dataclass(frozen=True)
class HarnessConfig:
    """Immutable configuration for a Harness.

    This configuration determines how a Harness wraps a provider:
    - Which tools are available (by name, resolved from registry)
    - System prompt to prepend to requests
    - Tool choice behavior (auto, none, required)
    - Backend selection (default, openai_agents)
    """

    name: str
    tool_names: tuple[str, ...] = ()
    system_prompt: str | None = None
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    backend: str = "default"
    required_secrets: tuple[str, ...] = ()
    max_turns: int | None = None
    max_concurrency: int | None = None

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
        d: dict[str, Any] = {
            "name": self.name,
            "tool_names": list(self.tool_names),
            "system_prompt": self.system_prompt,
            "tool_choice": self.tool_choice,
            "backend": self.backend,
            "required_secrets": list(self.required_secrets),
        }
        # Only include agent-specific fields if set
        if self.max_turns is not None:
            d["max_turns"] = self.max_turns
        if self.max_concurrency is not None:
            d["max_concurrency"] = self.max_concurrency
        return d

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
            backend=data.get("backend", "default"),
            required_secrets=tuple(data.get("required_secrets", [])),
            max_turns=data.get("max_turns"),
            max_concurrency=data.get("max_concurrency"),
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
            backend=self.backend,
            required_secrets=self.required_secrets,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
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
            backend=self.backend,
            required_secrets=self.required_secrets,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
        )


def harness_config(
    name: str,
    tools: Sequence[Tool | str] = (),
    system_prompt: str | None = None,
    tool_choice: Literal["auto", "none", "required"] | str = "auto",
    backend: str = "default",
    required_secrets: Sequence[str] = (),
    max_turns: int | None = None,
    max_concurrency: int | None = None,
) -> HarnessConfig:
    """Create a HarnessConfig, registering any Tool objects passed.

    This is a convenience function that accepts either Tool instances
    or tool names. Tool instances are automatically registered.

    Args:
        name: Human-readable name for this configuration.
        tools: Sequence of Tool instances or tool names.
        system_prompt: System prompt to prepend to requests.
        tool_choice: How the model should use tools.
        backend: Backend name.
        required_secrets: Environment variable names that must be set.
        max_turns: Maximum turns for agent backends (None = backend default).
        max_concurrency: Maximum concurrent tool executions for agent backends.

    Returns:
        A new HarnessConfig instance with tools registered.
    """
    from .tools import Tool, register_tool

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
        backend=backend,
        required_secrets=tuple(required_secrets),
        max_turns=max_turns,
        max_concurrency=max_concurrency,
    )

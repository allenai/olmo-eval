from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from olmo_eval.inference.providers.config import ProviderConfig

if TYPE_CHECKING:
    from olmo_eval.common.types import ToolSchema

    from .tools import Tool


@dataclass(frozen=True)
class HarnessConfig:
    """Immutable configuration for a Harness.

    This configuration determines how a Harness wraps a provider:
    - Provider configuration (via ProviderConfig)
    - Which tools are available (Tool objects or names resolved from registry)
    - System prompt to prepend to requests
    - Tool choice behavior (auto, none, required)
    - Backend selection (default, openai_agents)
    """

    name: str
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    tools: tuple[Tool | str, ...] = ()
    system_prompt: str | None = None
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    backend: str | None = None
    required_secrets: tuple[str, ...] = ()
    max_turns: int | None = None
    max_concurrency: int | None = None

    # Cache for resolved tools
    _resolved_tools_cache: tuple[Tool, ...] | None = field(
        default=None, repr=False, compare=False, hash=False
    )

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Get tool names (for serialization)."""
        return tuple(t if isinstance(t, str) else t.name for t in self.tools)

    @property
    def resolved_tools(self) -> tuple[Tool, ...]:
        """Resolve all tools to Tool instances.

        Results are cached since config is immutable.
        """
        if self._resolved_tools_cache is not None:
            return self._resolved_tools_cache

        from .tools import get_tool

        resolved = tuple(t if not isinstance(t, str) else get_tool(t) for t in self.tools)
        # Use object.__setattr__ to bypass frozen dataclass
        object.__setattr__(self, "_resolved_tools_cache", resolved)
        return resolved

    @property
    def tool_schemas(self) -> tuple[ToolSchema, ...]:
        """Get just the schemas for LLM requests.

        Returns:
            Tuple of ToolSchema instances for all configured tools.
        """
        return tuple(t.schema for t in self.resolved_tools)

    @property
    def has_tools(self) -> bool:
        """Check if this configuration has any tools enabled.

        Returns:
            True if at least one tool is configured.
        """
        return len(self.tools) > 0

    def validate_secrets(self) -> list[str]:
        """Check that all required secrets are available.

        Returns:
            List of missing secret names (empty if all present).
        """
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
            "provider": self.provider.to_dict(),
            # Tool configuration
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
        provider_data = data.get("provider", {})
        return cls(
            name=data.get("name", "default"),
            provider=ProviderConfig.from_dict(provider_data),
            # Tool configuration (serialized as "tool_names" for backward compat)
            tools=tuple(data.get("tool_names", [])),
            system_prompt=data.get("system_prompt"),
            tool_choice=data.get("tool_choice", "auto"),
            backend=data.get("backend"),
            required_secrets=tuple(data.get("required_secrets", [])),
            max_turns=data.get("max_turns"),
            max_concurrency=data.get("max_concurrency"),
        )

    def with_tools(self, *new_tools: Tool | str) -> HarnessConfig:
        """Create a new config with additional tools.

        Args:
            *new_tools: Tool instances or names to add.

        Returns:
            New HarnessConfig with the additional tools.
        """
        return HarnessConfig(
            name=self.name,
            provider=self.provider,
            tools=self.tools + new_tools,
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
            provider=self.provider,
            tools=self.tools,
            system_prompt=system_prompt,
            tool_choice=self.tool_choice,
            backend=self.backend,
            required_secrets=self.required_secrets,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
        )

    def with_provider(self, provider: ProviderConfig) -> HarnessConfig:
        """Create a new config with a different provider configuration.

        Args:
            provider: The new provider configuration to use.

        Returns:
            New HarnessConfig with the updated provider.
        """
        return HarnessConfig(
            name=self.name,
            provider=provider,
            tools=self.tools,
            system_prompt=self.system_prompt,
            tool_choice=self.tool_choice,
            backend=self.backend,
            required_secrets=self.required_secrets,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
        )

    def merge_provider(self, provider: ProviderConfig) -> HarnessConfig:
        """Create a new config merging model info from provider while preserving harness settings.

        This is useful when a harness preset specifies a provider kind (e.g., VLLM_SERVER)
        but the model name comes from user input. The harness's provider kind takes precedence
        if explicitly set (non-default), while model-specific fields come from the new provider.

        Args:
            provider: Provider configuration with model information.

        Returns:
            New HarnessConfig with merged provider settings.
        """
        from dataclasses import fields, replace

        # Start with incoming provider, overlay non-default harness values
        defaults = ProviderConfig()
        overrides = {
            f.name: getattr(self.provider, f.name)
            for f in fields(self.provider)
            if getattr(self.provider, f.name) != getattr(defaults, f.name)
        }
        # Special case: kwargs should merge, not replace
        if self.provider.kwargs:
            overrides["kwargs"] = {**provider.kwargs, **self.provider.kwargs}

        return self.with_provider(replace(provider, **overrides))

    def with_provider_overrides(self, **overrides: Any) -> HarnessConfig:
        """Create a new config with provider overrides applied.

        Convenience method that applies overrides to the provider config.
        Known provider field names are set directly; unknown names go to kwargs.
        None values are ignored.

        Args:
            **overrides: Provider field overrides.

        Returns:
            New HarnessConfig with updated provider.

        Example:
            config = harness_config.with_provider_overrides(
                tensor_parallel_size=4,
                max_model_len=8192,
                enable_auto_tool_choice=True,
            )
        """
        return self.with_provider(self.provider.with_overrides(**overrides))


def harness_config(
    name: str,
    provider: ProviderConfig | None = None,
    tools: Sequence[Tool | str] = (),
    system_prompt: str | None = None,
    tool_choice: Literal["auto", "none", "required"] | str = "auto",
    backend: str | None = None,
    required_secrets: Sequence[str] = (),
    max_turns: int | None = None,
    max_concurrency: int | None = None,
) -> HarnessConfig:
    """Create a HarnessConfig.

    Args:
        name: Human-readable name for this configuration.
        provider: Provider configuration (defaults to empty ProviderConfig).
        tools: Sequence of Tool instances or tool names.
        system_prompt: System prompt to prepend to requests.
        tool_choice: How the model should use tools.
        backend: Backend name (None = no multi-turn support via run()).
        required_secrets: Environment variable names for tools.
        max_turns: Maximum turns for agent backends (None = backend default).
        max_concurrency: Maximum concurrent tool executions for agent backends.

    Returns:
        A new HarnessConfig instance.
    """
    return HarnessConfig(
        name=name,
        provider=provider or ProviderConfig(),
        tools=tuple(tools),
        system_prompt=system_prompt,
        tool_choice=tool_choice,
        backend=backend,
        required_secrets=tuple(required_secrets),
        max_turns=max_turns,
        max_concurrency=max_concurrency,
    )

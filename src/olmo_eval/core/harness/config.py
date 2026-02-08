"""HarnessConfig: Immutable configuration describing model capabilities.

The configuration uses tool names (not Tool objects) for serialization.
Tools are resolved from the global registry at runtime.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from olmo_eval.core.types import ProviderKind

if TYPE_CHECKING:
    from olmo_eval.core.types import ToolSchema
    from olmo_eval.inference.base import InferenceProvider

    from .tools import Tool


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for creating an InferenceProvider.

    This configuration contains all the information needed to instantiate
    a provider via the create_provider factory function.

    Attributes:
        kind: Provider type (vllm, vllm_server, hf, litellm, mock).
        model_name: Model identifier or path (HuggingFace ID or local path).
        alias: Short display name for the model (used in DB and S3 paths).
        base_url: Base URL for API-based providers (vllm_server, litellm).
        tokenizer: Tokenizer path/identifier (defaults to model_name if None).
        revision: Model revision/commit hash for HuggingFace models.
        trust_remote_code: Whether to trust remote code for HuggingFace models.
        dtype: Data type for model weights (auto, float16, bfloat16, float32).
        max_model_len: Maximum sequence length (overrides model default).
        max_concurrency: Maximum concurrent requests.
        required_secrets: Environment variable names that must be set.
        package: Optional custom package specifier for runtime installation.
        kwargs: Additional arguments passed to the provider constructor.
    """

    kind: str = ProviderKind.VLLM
    model_name: str = ""
    alias: str | None = None
    base_url: str | None = None
    tokenizer: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: int | None = None
    max_concurrency: int | None = None
    required_secrets: tuple[str, ...] = ()
    package: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    def create_provider(self) -> InferenceProvider:
        """Create an InferenceProvider from this configuration.

        Returns:
            Configured InferenceProvider instance.

        Raises:
            ValueError: If required secrets are missing or provider type is unknown.
        """
        from olmo_eval.inference import create_provider

        # Validate secrets
        missing = self.validate_secrets()
        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")

        # Build kwargs from config fields
        provider_kwargs: dict[str, Any] = dict(self.kwargs)
        if self.base_url is not None:
            provider_kwargs["base_url"] = self.base_url
        if self.tokenizer is not None:
            provider_kwargs["tokenizer"] = self.tokenizer
        if self.revision is not None:
            provider_kwargs["revision"] = self.revision
        if self.trust_remote_code:
            provider_kwargs["trust_remote_code"] = self.trust_remote_code
        if self.dtype != "auto":
            provider_kwargs["dtype"] = self.dtype
        if self.max_model_len is not None:
            provider_kwargs["max_model_len"] = self.max_model_len
        if self.max_concurrency is not None:
            provider_kwargs["max_concurrency"] = self.max_concurrency

        return create_provider(self.kind, self.model_name, **provider_kwargs)

    def get_provider_name(self, override: str | None = None) -> str:
        """Get the effective provider name as a string.

        Args:
            override: Optional provider name override.

        Returns:
            Provider name string (e.g., "vllm", "litellm", "hf").
        """
        if override:
            return override
        kind = self.kind
        return str(kind.value) if hasattr(kind, "value") else str(kind)

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
            "kind": self.kind,
            "model_name": self.model_name,
        }
        if self.base_url is not None:
            d["base_url"] = self.base_url
        if self.tokenizer is not None:
            d["tokenizer"] = self.tokenizer
        if self.revision is not None:
            d["revision"] = self.revision
        if self.trust_remote_code:
            d["trust_remote_code"] = self.trust_remote_code
        if self.dtype != "auto":
            d["dtype"] = self.dtype
        if self.max_model_len is not None:
            d["max_model_len"] = self.max_model_len
        if self.max_concurrency is not None:
            d["max_concurrency"] = self.max_concurrency
        if self.required_secrets:
            d["required_secrets"] = list(self.required_secrets)
        if self.package is not None:
            d["package"] = self.package
        if self.kwargs:
            d["kwargs"] = dict(self.kwargs)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProviderConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with ProviderConfig data.

        Returns:
            A new ProviderConfig instance.
        """
        return cls(
            kind=data.get("kind", ProviderKind.VLLM),
            model_name=data.get("model_name", ""),
            base_url=data.get("base_url"),
            tokenizer=data.get("tokenizer"),
            revision=data.get("revision"),
            trust_remote_code=data.get("trust_remote_code", False),
            dtype=data.get("dtype", "auto"),
            max_model_len=data.get("max_model_len"),
            max_concurrency=data.get("max_concurrency"),
            required_secrets=tuple(data.get("required_secrets", [])),
            package=data.get("package"),
            kwargs=data.get("kwargs", {}),
        )


@dataclass(frozen=True)
class HarnessConfig:
    """Immutable configuration for a Harness.

    This configuration determines how a Harness wraps a provider:
    - Provider configuration (via ProviderConfig)
    - Which tools are available (by name, resolved from registry)
    - System prompt to prepend to requests
    - Tool choice behavior (auto, none, required)
    - Backend selection (default, openai_agents)
    """

    name: str
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    # Tool configuration
    tool_names: tuple[str, ...] = ()
    system_prompt: str | None = None
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    backend: str = "default"
    required_secrets: tuple[str, ...] = ()  # For tools
    max_turns: int | None = None
    max_concurrency: int | None = None  # For agent execution

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
            # Tool configuration
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
            provider=self.provider,
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
            provider=self.provider,
            tool_names=self.tool_names,
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
            tool_names=self.tool_names,
            system_prompt=self.system_prompt,
            tool_choice=self.tool_choice,
            backend=self.backend,
            required_secrets=self.required_secrets,
            max_turns=self.max_turns,
            max_concurrency=self.max_concurrency,
        )


def harness_config(
    name: str,
    provider: ProviderConfig | None = None,
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
        provider: Provider configuration (defaults to empty ProviderConfig).
        tools: Sequence of Tool instances or tool names.
        system_prompt: System prompt to prepend to requests.
        tool_choice: How the model should use tools.
        backend: Backend name.
        required_secrets: Environment variable names for tools.
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
        provider=provider or ProviderConfig(),
        tool_names=tuple(tool_names),
        system_prompt=system_prompt,
        tool_choice=tool_choice,
        backend=backend,
        required_secrets=tuple(required_secrets),
        max_turns=max_turns,
        max_concurrency=max_concurrency,
    )

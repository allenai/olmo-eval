from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import TYPE_CHECKING, Any, Literal

from olmo_eval.common.repr import hide_unset
from olmo_eval.inference.providers.config import ProviderConfig

if TYPE_CHECKING:
    from olmo_eval.common.types import ToolSchema
    from olmo_eval.inference.metrics import MetricsConfig
    from olmo_eval.runners.asynq.batching import BatchConfig

    from .sandbox import SandboxConfig
    from .tools import Tool


@hide_unset(skip=frozenset({"_resolved_tools_cache"}))
@dataclass(frozen=True)
class HarnessConfig:
    """Immutable configuration for a Harness.

    This configuration determines how a Harness wraps a provider:
    - Provider configuration (via ProviderConfig)
    - Which tools are available (Tool objects or names resolved from registry)
    - System prompt to prepend to requests
    - Tool choice behavior (auto, none, required)
    - Backend selection (default, openai_agents)
    - Sandbox configurations for isolated tool execution
    """

    name: str
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    auxiliary_providers: Mapping[str, ProviderConfig] = field(default_factory=dict)
    tools: tuple[Tool | str, ...] = ()
    system_prompt: str | None = None
    tool_choice: Literal["auto", "none", "required"] | str = "auto"
    backend: str | None = None
    required_secrets: tuple[str, ...] = ()
    max_turns: int | None = None
    max_concurrency: int | None = None
    scoring_concurrency: int | None = None
    sandboxes: tuple[SandboxConfig, ...] = ()
    sandbox_pool_instances: int | None = None
    backend_kwargs: dict[str, Any] = field(default_factory=dict)
    metrics: MetricsConfig | None = None
    batching: BatchConfig | None = None
    scorer_startup_timeout: float | None = None

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

    @property
    def has_sandbox_tools(self) -> bool:
        """Check if any configured tools require sandbox execution.

        Returns:
            True if at least one tool has non-empty capabilities.
        """
        return any(t.sandbox for t in self.resolved_tools)

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
        if self.auxiliary_providers:
            d["auxiliary_providers"] = {
                name: config.to_dict() for name, config in self.auxiliary_providers.items()
            }
        if self.max_turns is not None:
            d["max_turns"] = self.max_turns
        if self.max_concurrency is not None:
            d["max_concurrency"] = self.max_concurrency
        if self.scoring_concurrency is not None:
            d["scoring_concurrency"] = self.scoring_concurrency
        if self.sandboxes:
            d["sandboxes"] = [s.to_dict() for s in self.sandboxes]
        if self.sandbox_pool_instances is not None:
            d["sandbox_pool_instances"] = self.sandbox_pool_instances
        if self.backend_kwargs:
            d["backend_kwargs"] = self.backend_kwargs
        if self.metrics is not None:
            d["metrics"] = self.metrics.to_dict()
        if self.batching is not None:
            d["batching"] = self.batching.to_dict()
        if self.scorer_startup_timeout is not None:
            d["scorer_startup_timeout"] = self.scorer_startup_timeout
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HarnessConfig:
        """Create from dictionary.

        Args:
            data: Dictionary with HarnessConfig data.

        Returns:
            A new HarnessConfig instance.
        """
        from olmo_eval.inference.metrics import MetricsConfig
        from olmo_eval.runners.asynq.batching import BatchConfig

        from .sandbox import SandboxConfig

        provider_data = data.get("provider", {})
        sandboxes_data = data.get("sandboxes", [])
        sandboxes = tuple(SandboxConfig.from_dict(s) for s in sandboxes_data)

        metrics_data = data.get("metrics")
        metrics = MetricsConfig.from_dict(metrics_data) if metrics_data else None

        batching_data = data.get("batching")
        batching = BatchConfig.from_dict(batching_data) if batching_data else None

        auxiliary_data = data.get("auxiliary_providers", {})
        auxiliary_providers = {
            name: ProviderConfig.from_dict(config) for name, config in auxiliary_data.items()
        }

        return cls(
            name=data.get("name", "default"),
            provider=ProviderConfig.from_dict(provider_data),
            auxiliary_providers=auxiliary_providers,
            tools=tuple(data.get("tool_names", [])),
            system_prompt=data.get("system_prompt"),
            tool_choice=data.get("tool_choice", "auto"),
            backend=data.get("backend"),
            required_secrets=tuple(data.get("required_secrets", [])),
            max_turns=data.get("max_turns"),
            max_concurrency=data.get("max_concurrency"),
            scoring_concurrency=data.get("scoring_concurrency"),
            sandboxes=sandboxes,
            sandbox_pool_instances=data.get("sandbox_pool_instances"),
            backend_kwargs=data.get("backend_kwargs", {}),
            metrics=metrics,
            batching=batching,
            scorer_startup_timeout=data.get("scorer_startup_timeout"),
        )

    def with_tools(self, *new_tools: Tool | str) -> HarnessConfig:
        """Create a new config with additional tools."""
        return replace(self, tools=self.tools + new_tools)

    def with_system_prompt(self, system_prompt: str) -> HarnessConfig:
        """Create a new config with a different system prompt."""
        return replace(self, system_prompt=system_prompt)

    def with_provider(self, provider: ProviderConfig) -> HarnessConfig:
        """Create a new config with a different provider configuration."""
        return replace(self, provider=provider)

    def merge_provider(self, provider: ProviderConfig) -> HarnessConfig:
        """Merge model info from provider while preserving harness provider settings.

        The harness's provider kind takes precedence if explicitly set (non-default),
        while model-specific fields come from the new provider.
        """
        defaults = ProviderConfig()
        overrides = {
            f.name: getattr(self.provider, f.name)
            for f in fields(self.provider)
            if getattr(self.provider, f.name) != getattr(defaults, f.name)
        }
        # kwargs should merge, not replace
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

    def with_metrics(self, metrics: MetricsConfig) -> HarnessConfig:
        """Create a new config with updated metrics configuration."""
        return replace(self, metrics=metrics)

    def with_batching(self, batching: BatchConfig) -> HarnessConfig:
        """Create a new config with updated batching configuration."""
        return replace(self, batching=batching)


def harness_config(
    name: str,
    provider: ProviderConfig | None = None,
    auxiliary_providers: Mapping[str, ProviderConfig] | None = None,
    tools: Sequence[Tool | str] = (),
    system_prompt: str | None = None,
    tool_choice: Literal["auto", "none", "required"] | str = "auto",
    backend: str | None = None,
    required_secrets: Sequence[str] = (),
    max_turns: int | None = None,
    max_concurrency: int | None = None,
    scoring_concurrency: int | None = None,
    sandboxes: Sequence[SandboxConfig] = (),
    sandbox_pool_instances: int | None = None,
    backend_kwargs: dict[str, Any] | None = None,
    metrics: MetricsConfig | None = None,
    batching: BatchConfig | None = None,
    scorer_startup_timeout: float | None = None,
) -> HarnessConfig:
    """Create a HarnessConfig.

    Args:
        name: Human-readable name for this configuration.
        provider: Provider configuration (defaults to empty ProviderConfig).
        auxiliary_providers: Named auxiliary providers for scoring, sub-agents, etc.
        tools: Sequence of Tool instances or tool names.
        system_prompt: System prompt to prepend to requests.
        tool_choice: How the model should use tools.
        backend: Backend name (None = no multi-turn support via run()).
        required_secrets: Environment variable names for tools.
        max_turns: Maximum turns for agent backends (None = backend default).
        max_concurrency: Maximum concurrent tool executions for agent backends.
        scoring_concurrency: Maximum concurrent scoring operations (default 8).
        sandboxes: Sandbox configurations for isolated tool execution.
        sandbox_pool_instances: Shared executor budget for auto-allocated sandboxes.
        backend_kwargs: Backend-specific kwargs (e.g., enable_compaction for openai_agents).
        metrics: Metrics collection configuration (None = no metrics).
        batching: Batching strategy configuration (None = sequential).
        scorer_startup_timeout: Timeout for scorer worker startup. If None, derived from
            sandbox configs (max startup_timeout + 60s buffer) or defaults to 60s.

    Returns:
        A new HarnessConfig instance.
    """
    return HarnessConfig(
        name=name,
        provider=provider or ProviderConfig(),
        auxiliary_providers=auxiliary_providers or {},
        tools=tuple(tools),
        system_prompt=system_prompt,
        tool_choice=tool_choice,
        backend=backend,
        required_secrets=tuple(required_secrets),
        max_turns=max_turns,
        max_concurrency=max_concurrency,
        scoring_concurrency=scoring_concurrency,
        sandboxes=tuple(sandboxes),
        sandbox_pool_instances=sandbox_pool_instances,
        backend_kwargs=backend_kwargs or {},
        metrics=metrics,
        batching=batching,
        scorer_startup_timeout=scorer_startup_timeout,
    )

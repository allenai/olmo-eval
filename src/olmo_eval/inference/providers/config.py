"""Provider configuration for inference providers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from olmo_eval.common.types import ProviderKind

if TYPE_CHECKING:
    from olmo_eval.inference.base import InferenceProvider


@dataclass(frozen=True)
class ProviderConfig:
    """Immutable configuration for creating an InferenceProvider.

    This configuration contains all the information needed to instantiate
    a provider via the create_provider factory function.

    Attributes:
        kind: Provider type (vllm, vllm_server, hf, litellm, mock).
        model: Model name or path (HuggingFace ID, API model name, or local path).
        alias: Short display name for the model (used in DB and S3 paths).
        base_url: Base URL for API-based providers (vllm_server, litellm).
        tokenizer: Tokenizer path/identifier (defaults to model if None).
        revision: Model revision/commit hash for HuggingFace models.
        trust_remote_code: Whether to trust remote code for HuggingFace models.
        dtype: Data type for model weights (auto, float16, bfloat16, float32).
        max_model_len: Maximum sequence length (overrides model default).
        max_concurrency: Maximum concurrent requests.
        tool_call_parser: Tool call parser for vLLM server (llama3_json, mistral,
            hermes). Auto-detected from model name if not specified.
        required_secrets: Environment variable names that must be set.
        package: Optional custom package specifier for runtime installation.
        kwargs: Additional arguments passed to the provider constructor.
    """

    kind: str = ProviderKind.VLLM
    model: str = ""
    alias: str | None = None
    base_url: str | None = None
    tokenizer: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    dtype: str = "auto"
    max_model_len: int | None = None
    max_concurrency: int | None = None
    tool_call_parser: str | None = None
    required_secrets: tuple[str, ...] = ()
    package: str | None = None
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    # Providers that require GPU resources for local inference
    _GPU_PROVIDERS: ClassVar[frozenset[str]] = frozenset({"vllm", "vllm_server", "hf"})

    @property
    def requires_gpu(self) -> bool:
        """Whether this provider requires GPU resources.

        Returns True for local inference providers (vllm, vllm_server, hf).
        Returns False for API-based providers (litellm, mock).
        """
        return str(self.kind) in self._GPU_PROVIDERS

    # Config fields accepted by each provider kind (fields with non-default values are passed)
    _PROVIDER_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "vllm": ("tokenizer", "revision", "trust_remote_code", "dtype", "max_model_len"),
        "vllm_server": ("base_url", "tokenizer", "max_concurrency", "tool_call_parser"),
        "litellm": ("base_url", "max_concurrency"),
        "hf": ("tokenizer", "trust_remote_code", "dtype"),
        "mock": (),
    }

    # Extra kwargs accepted by each provider (from self.kwargs)
    # vllm and vllm_server share most engine kwargs since vllm_server can spawn a vLLM process
    _VLLM_ENGINE_KWARGS: ClassVar[tuple[str, ...]] = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "attention_backend",
        "load_format",
        "model_loader_extra_config",
        "enforce_eager",
        "enable_chunked_prefill",
        "quantization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "guided_decoding_backend",
        "enable_prefix_caching",
        "disable_sliding_window",
        "seed",
    )

    _PROVIDER_KWARGS: ClassVar[dict[str, tuple[str, ...]]] = {
        "vllm": _VLLM_ENGINE_KWARGS,
        "vllm_server": (
            *_VLLM_ENGINE_KWARGS,
            "enable_auto_tool_choice",
            "timeout",
            "max_retries",
            "retry_delay",
            "log_dir",
            "chat_template_kwargs",
        ),
        "litellm": (),
        "hf": ("torch_dtype", "device_map", "attn_implementation"),
        "mock": (),
    }

    def create_provider(self) -> InferenceProvider:
        """Create an InferenceProvider from this configuration."""
        from olmo_eval.inference import create_provider

        missing = self.validate_secrets()
        if missing:
            raise ValueError(f"Missing required secrets: {', '.join(missing)}")

        kind_str = str(self.kind)
        provider_kwargs: dict[str, Any] = {}

        # Filter self.kwargs to only include kwargs this provider accepts
        allowed_kwargs = set(self._PROVIDER_KWARGS.get(kind_str, ()))
        for key, value in self.kwargs.items():
            if key in allowed_kwargs:
                provider_kwargs[key] = value

        # Add config fields this provider accepts (skip None/default values)
        for field_name in self._PROVIDER_FIELDS.get(kind_str, ()):
            value = getattr(self, field_name)
            if value is not None and value is not False and value != "auto":
                provider_kwargs[field_name] = value

        return create_provider(self.kind, self.model, **provider_kwargs)

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
            "model": self.model,
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
        if self.tool_call_parser is not None:
            d["tool_call_parser"] = self.tool_call_parser
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
            model=data.get("model", data.get("model_name", "")),
            base_url=data.get("base_url"),
            tokenizer=data.get("tokenizer"),
            revision=data.get("revision"),
            trust_remote_code=data.get("trust_remote_code", False),
            dtype=data.get("dtype", "auto"),
            max_model_len=data.get("max_model_len"),
            max_concurrency=data.get("max_concurrency"),
            tool_call_parser=data.get("tool_call_parser"),
            required_secrets=tuple(data.get("required_secrets", [])),
            package=data.get("package"),
            kwargs=data.get("kwargs", {}),
        )

    def with_overrides(self, **overrides: Any) -> ProviderConfig:
        """Create a new config with overrides applied.

        Known field names are set directly on the config. Unknown names
        are merged into kwargs. None values are ignored.

        Args:
            **overrides: Field overrides (e.g., max_model_len=4096).

        Returns:
            New ProviderConfig with overrides applied.

        Example:
            config = provider_config.with_overrides(
                tensor_parallel_size=4,
                max_model_len=8192,
                tokenizer="/path/to/tokenizer",
            )
        """
        from dataclasses import fields, replace

        # Separate known fields from kwargs
        field_names = {f.name for f in fields(self)}
        field_overrides = {}
        extra_kwargs = dict(self.kwargs)

        for key, value in overrides.items():
            if value is None:
                continue
            if key in field_names and key != "kwargs":
                field_overrides[key] = value
            else:
                extra_kwargs[key] = value

        if extra_kwargs != dict(self.kwargs):
            field_overrides["kwargs"] = extra_kwargs

        return replace(self, **field_overrides) if field_overrides else self

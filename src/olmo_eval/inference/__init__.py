"""Language model inference providers."""

from olmo_eval.common.types import ProviderKind

from .base import InferenceProvider
from .providers.mock import MockProvider
from .tokenizer_utils import (
    encode_context_and_continuation,
    get_bos_token_ids,
    get_context_token_ids,
    has_bos_token,
)

__all__ = [
    "InferenceProvider",
    "ProviderKind",
    "MockProvider",
    "HuggingFaceProvider",
    "VLLMProvider",
    "VLLMServerProvider",
    "LiteLLMProvider",
    "create_provider",
    # Tokenizer utilities
    "encode_context_and_continuation",
    "get_bos_token_ids",
    "get_context_token_ids",
    "has_bos_token",
]


def create_provider(
    provider_kind: ProviderKind | str,
    model_name: str,
    worker_id: str | None = None,
    **kwargs,
) -> InferenceProvider:
    """Create a provider instance.

    Args:
        provider_kind: Kind of provider to create (e.g., "vllm", "vllm_server", "litellm").
        model_name: Model identifier or path.
        worker_id: Optional worker identifier for logging (only used by vLLM).
        **kwargs: Additional arguments passed to provider constructor.

    Returns:
        Initialized provider instance.

    Raises:
        ValueError: If provider kind is unknown.
    """
    # Normalize to string for comparison (StrEnum compares equal to its value)
    kind_str = str(provider_kind)

    match kind_str:
        case "mock":
            return MockProvider(model_name)
        case "hf":
            from .providers.huggingface import HuggingFaceProvider

            return HuggingFaceProvider(model_name, **kwargs)
        case "vllm":
            from .providers.vllm import VLLMProvider

            return VLLMProvider(model_name, worker_id=worker_id, **kwargs)
        case "vllm_server":
            from .providers.vllm_server import VLLMServerProvider

            return VLLMServerProvider(model_name, **kwargs)
        case "litellm":
            from .providers.litellm import LiteLLMProvider

            return LiteLLMProvider(model_name, **kwargs)
        case _:
            raise ValueError(f"Unknown provider kind: {provider_kind}")


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "HuggingFaceProvider":
        from .providers.huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    if name == "VLLMProvider":
        from .providers.vllm import VLLMProvider

        return VLLMProvider
    if name == "VLLMServerProvider":
        from .providers.vllm_server import VLLMServerProvider

        return VLLMServerProvider
    if name == "LiteLLMProvider":
        from .providers.litellm import LiteLLMProvider

        return LiteLLMProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

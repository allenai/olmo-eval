"""Language model inference providers."""

from enum import Enum

from .base import InferenceProvider
from .mock import MockProvider

__all__ = [
    "InferenceProvider",
    "ProviderType",
    "MockProvider",
    "HuggingFaceProvider",
    "VLLMProvider",
    "LiteLLMProvider",
    "create_provider",
]


class ProviderType(str, Enum):
    """Supported provider types."""

    MOCK = "mock"
    HUGGINGFACE = "hf"
    VLLM = "vllm"
    LITELLM = "litellm"


def create_provider(
    provider_type: ProviderType | str, model_name: str, **kwargs
) -> InferenceProvider:
    """Create a provider instance.

    Args:
        provider_type: Type of provider to create.
        model_name: Model identifier or path.
        **kwargs: Additional arguments passed to provider constructor.

    Returns:
        Initialized provider instance.

    Raises:
        ValueError: If provider type is unknown.
    """
    provider_type = ProviderType(provider_type) if isinstance(provider_type, str) else provider_type

    match provider_type:
        case ProviderType.MOCK:
            return MockProvider(model_name)
        case ProviderType.HUGGINGFACE:
            from .huggingface import HuggingFaceProvider

            return HuggingFaceProvider(model_name, **kwargs)
        case ProviderType.VLLM:
            from .vllm import VLLMProvider

            return VLLMProvider(model_name, **kwargs)
        case ProviderType.LITELLM:
            from .litellm import LiteLLMProvider

            return LiteLLMProvider(model_name, **kwargs)
        case _:
            raise ValueError(f"Unknown provider type: {provider_type}")


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "HuggingFaceProvider":
        from .huggingface import HuggingFaceProvider

        return HuggingFaceProvider
    if name == "VLLMProvider":
        from .vllm import VLLMProvider

        return VLLMProvider
    if name == "LiteLLMProvider":
        from .litellm import LiteLLMProvider

        return LiteLLMProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

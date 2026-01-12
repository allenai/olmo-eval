"""Language model backends."""

from enum import Enum

from .base import Backend
from .mock import MockBackend

__all__ = [
    "Backend",
    "BackendType",
    "MockBackend",
    "HuggingFaceBackend",
    "VLLMBackend",
    "LiteLLMBackend",
    "create_backend",
]


class BackendType(str, Enum):
    """Supported backend types."""

    MOCK = "mock"
    HUGGINGFACE = "hf"
    VLLM = "vllm"
    LITELLM = "litellm"


def create_backend(backend_type: BackendType | str, model_name: str, **kwargs) -> Backend:
    """Create a backend instance.

    Args:
        backend_type: Type of backend to create.
        model_name: Model identifier or path.
        **kwargs: Additional arguments passed to backend constructor.

    Returns:
        Initialized backend instance.

    Raises:
        ValueError: If backend type is unknown.
    """
    backend_type = BackendType(backend_type) if isinstance(backend_type, str) else backend_type

    match backend_type:
        case BackendType.MOCK:
            return MockBackend(model_name)
        case BackendType.HUGGINGFACE:
            from .huggingface import HuggingFaceBackend

            return HuggingFaceBackend(model_name, **kwargs)
        case BackendType.VLLM:
            from .vllm import VLLMBackend

            return VLLMBackend(model_name, **kwargs)
        case BackendType.LITELLM:
            from .litellm import LiteLLMBackend

            return LiteLLMBackend(model_name, **kwargs)
        case _:
            raise ValueError(f"Unknown backend type: {backend_type}")


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "HuggingFaceBackend":
        from .huggingface import HuggingFaceBackend

        return HuggingFaceBackend
    if name == "VLLMBackend":
        from .vllm import VLLMBackend

        return VLLMBackend
    if name == "LiteLLMBackend":
        from .litellm import LiteLLMBackend

        return LiteLLMBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

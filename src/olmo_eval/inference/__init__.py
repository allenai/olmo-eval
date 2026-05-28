"""Language model inference providers."""

import logging

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
    # Metrics (lazy import via __getattr__)
    "metrics",
]


logger = logging.getLogger(__name__)


def _print_model_config(
    model_name: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = True,
    force_download: bool = False,
) -> None:
    """Log resolved HF model config, including architecture and RoPE settings."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            force_download=force_download,
        )
        architectures = getattr(config, "architectures", ["Unknown"])
        model_type = getattr(config, "model_type", None)
        max_pos = getattr(config, "max_position_embeddings", None)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)

        logger.info(
            "Model config resolved | model=%s model_type=%s architectures=%s "
            "max_position_embeddings=%s rope_scaling=%s rope_theta=%s",
            model_name,
            model_type,
            architectures,
            max_pos,
            rope_scaling,
            rope_theta,
        )
    except Exception as e:
        logger.warning("Could not load model config for %s: %s", model_name, e)


def _print_tokenizer_config(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = True,
    force_download: bool = False,
) -> None:
    """Log resolved tokenizer details used at inference startup."""
    effective_tokenizer = tokenizer_name or model_name
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            effective_tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            force_download=force_download,
        )
        chat_template = getattr(tokenizer, "chat_template", None)
        has_chat_template = bool(chat_template)

        logger.info(
            "Tokenizer resolved | model=%s tokenizer=%s tokenizer_class=%s vocab_size=%s "
            "model_max_length=%s bos_token_id=%s eos_token_id=%s pad_token_id=%s "
            "has_chat_template=%s",
            model_name,
            effective_tokenizer,
            tokenizer.__class__.__name__,
            getattr(tokenizer, "vocab_size", None),
            getattr(tokenizer, "model_max_length", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            has_chat_template,
        )
    except Exception as e:
        logger.warning("Could not load tokenizer config for %s: %s", effective_tokenizer, e)


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
    revision = kwargs.get("revision")
    trust_remote_code = kwargs.get("trust_remote_code", True)
    force_download = bool(kwargs.get("force_download", False))
    tokenizer_name = kwargs.get("tokenizer")

    match kind_str:
        case "mock":
            return MockProvider(model_name)
        case "hf":
            from .providers.huggingface import HuggingFaceProvider

            _print_model_config(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            _print_tokenizer_config(
                model_name,
                tokenizer_name=tokenizer_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            return HuggingFaceProvider(model_name, **kwargs)
        case "vllm":
            from .providers.vllm import VLLMProvider

            _print_model_config(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            _print_tokenizer_config(
                model_name,
                tokenizer_name=tokenizer_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            return VLLMProvider(model_name, worker_id=worker_id, **kwargs)
        case "vllm_server":
            from .providers.vllm_server import VLLMServerProvider

            _print_model_config(
                model_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
            _print_tokenizer_config(
                model_name,
                tokenizer_name=tokenizer_name,
                revision=revision,
                trust_remote_code=trust_remote_code,
                force_download=force_download,
            )
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
    if name == "metrics":
        from . import metrics

        return metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

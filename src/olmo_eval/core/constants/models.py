from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.core.configs import ModelConfig


# =============================================================================
# Model Presets for Evaluation
# =============================================================================


def get_model_presets() -> dict[str, ModelConfig]:
    """Get model presets dictionary.

    Returns a dictionary mapping preset names to ModelConfig instances.
    Uses lazy import to avoid circular dependencies.

    Model presets can be either:
    1. HuggingFace models (for vLLM inference)
    2. API-based models with model_url (for agent tasks or LiteLLM)
    """
    from olmo_eval.core.configs import ModelConfig
    from olmo_eval.core.types import ProviderKind
    from olmo_eval.launch.config import ProviderConfig

    return {
        # HuggingFace models (vLLM inference)
        "llama3.1-8b": ModelConfig(
            model="meta-llama/Meta-Llama-3.1-8B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama3.1-8b-instruct": ModelConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama3.1-70b": ModelConfig(
            model="meta-llama/Meta-Llama-3.1-70B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama3.1-70b-instruct": ModelConfig(
            model="meta-llama/Llama-3.1-70B-Instruct",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-2-7b": ModelConfig(
            model="allenai/OLMo-2-1124-7B",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-2-13b": ModelConfig(
            model="allenai/OLMo-2-1124-13B",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen2.5-7b": ModelConfig(
            model="Qwen/Qwen2.5-7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mistral-7b": ModelConfig(
            model="mistralai/Mistral-7B-v0.3",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        # Mock model for testing (no dependencies required)
        "mock": ModelConfig(
            model="mock",
            provider=ProviderConfig(kind=ProviderKind.MOCK),
        ),
        # API-based models (for agent tasks - requires API keys)
        "gpt-4o": ModelConfig(
            model="gpt-4o",
            model_url="https://api.openai.com/v1",
            provider=ProviderConfig(
                kind=ProviderKind.LITELLM, required_secrets=("OPENAI_API_KEY",)
            ),
        ),
        "gpt-4o-mini": ModelConfig(
            model="gpt-4o-mini",
            model_url="https://api.openai.com/v1",
            provider=ProviderConfig(
                kind=ProviderKind.LITELLM, required_secrets=("OPENAI_API_KEY",)
            ),
        ),
        "gpt-4-turbo": ModelConfig(
            model="gpt-4-turbo",
            model_url="https://api.openai.com/v1",
            provider=ProviderConfig(
                kind=ProviderKind.LITELLM, required_secrets=("OPENAI_API_KEY",)
            ),
        ),
        "claude-3-opus": ModelConfig(
            model="claude-3-opus-20240229",
            model_url="https://api.anthropic.com",
            provider=ProviderConfig(
                kind=ProviderKind.LITELLM, required_secrets=("ANTHROPIC_API_KEY",)
            ),
        ),
        "claude-3-sonnet": ModelConfig(
            model="claude-3-sonnet-20240229",
            model_url="https://api.anthropic.com",
            provider=ProviderConfig(
                kind=ProviderKind.LITELLM, required_secrets=("ANTHROPIC_API_KEY",)
            ),
        ),
    }

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.harness.config import ProviderConfig


# =============================================================================
# Model Presets for Evaluation
# =============================================================================


def get_model_presets() -> dict[str, ProviderConfig]:
    """Get model presets dictionary.

    Returns a dictionary mapping preset names to ProviderConfig instances.
    """
    from olmo_eval.common.types import ProviderKind
    from olmo_eval.harness.config import ProviderConfig

    return {
        "llama3.1-8b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Meta-Llama-3.1-8B",
        ),
        "llama3.1-8b-instruct": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Llama-3.1-8B-Instruct",
        ),
        "llama3.1-70b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Meta-Llama-3.1-70B",
        ),
        "llama3.1-70b-instruct": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="meta-llama/Llama-3.1-70B-Instruct",
        ),
        "olmo-2-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/OLMo-2-1124-7B",
            trust_remote_code=True,
        ),
        "olmo-2-13b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="allenai/OLMo-2-1124-13B",
            trust_remote_code=True,
        ),
        "qwen2.5-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="Qwen/Qwen2.5-7B",
        ),
        "mistral-7b": ProviderConfig(
            kind=ProviderKind.VLLM,
            model="mistralai/Mistral-7B-v0.3",
        ),
        "gpt-4o": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "gpt-4o-mini": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "gpt-4-turbo": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="gpt-4-turbo",
            base_url="https://api.openai.com/v1",
            required_secrets=("OPENAI_API_KEY",),
        ),
        "claude-3-opus": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="claude-3-opus-20240229",
            base_url="https://api.anthropic.com",
            required_secrets=("ANTHROPIC_API_KEY",),
        ),
        "claude-3-sonnet": ProviderConfig(
            kind=ProviderKind.LITELLM,
            model="claude-3-sonnet-20240229",
            base_url="https://api.anthropic.com",
            required_secrets=("ANTHROPIC_API_KEY",),
        ),
        "mock": ProviderConfig(
            kind=ProviderKind.MOCK,
            model="mock",
        ),
    }

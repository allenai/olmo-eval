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

        # Llama
        "llama3-8b": ModelConfig(
            model="meta-llama/Meta-Llama-3-8B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama-3.1-8b": ModelConfig(
            model="meta-llama/Meta-Llama-3.1-8B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama-3.1-8b-instruct": ModelConfig(
            model="meta-llama/Llama-3.1-8B-Instruct",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama-3.1-70b": ModelConfig(
            model="meta-llama/Meta-Llama-3.1-70B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama-3.1-70b-instruct": ModelConfig(
            model="meta-llama/Llama-3.1-70B-Instruct",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama3.2-1b": ModelConfig(
            model="meta-llama/Llama-3.2-1B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama3.2-3b": ModelConfig(
            model="meta-llama/Llama-3.2-3B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "llama4-scout-17b-16e": ModelConfig(
            model="meta-llama/Llama-4-Scout-17B-16E",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),

        # Yi
        "yi-1.5-34b": ModelConfig(
            model="01-ai/Yi-1.5-34B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "yi-1.5-9b": ModelConfig(
            model="01-ai/Yi-1.5-9B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "yi-1.5-6b": ModelConfig(
            model="01-ai/Yi-1.5-6B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "yi-34b": ModelConfig(
            model="01-ai/Yi-34B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "yi-9b": ModelConfig(
            model="01-ai/Yi-9B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "yi-6b": ModelConfig(
            model="01-ai/Yi-6B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Olmo
        "olmo-1b-0724": ModelConfig(
            model="allenai/OLMo-1B-0724-hf",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-2-32b": ModelConfig(
            model="allenai/OLMo-2-0325-32B",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-2-1b": ModelConfig(
            model="allenai/OLMo-2-0425-1B",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-7b-0424": ModelConfig(
            model="allenai/OLMo-7B-0424-hf",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-7b-0724": ModelConfig(
            model="allenai/OLMo-7B-0724-hf",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-1b": ModelConfig(
            model="allenai/OLMo-1B-hf",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-7b": ModelConfig(
            model="allenai/OLMo-7B-hf",
            trust_remote_code=True,
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "olmo-7b-twin-2t": ModelConfig(
            model="allenai/OLMo-7B-Twin-2T-hf",
            trust_remote_code=True,
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
    
        # DeepSeek
        "deepseek-llm-67b": ModelConfig(
            model="deepseek-ai/deepseek-llm-67b-base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "deepseek-llm-7b": ModelConfig(
            model="deepseek-ai/deepseek-llm-7b-base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "deepseek-moe-16b": ModelConfig(
            model="deepseek-ai/deepseek-moe-16b-base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "deepseek-v2-lite": ModelConfig(
            model="deepseek-ai/DeepSeek-V2-Lite",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Gemma (Google)
        "gemma-2-27b": ModelConfig(
            model="google/gemma-2-27b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "gemma-2-9b": ModelConfig(
            model="google/gemma-2-9b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "gemma-2-2b": ModelConfig(
            model="google/gemma-2-2b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "gemma-2b": ModelConfig(
            model="google/gemma-2b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "gemma-7b": ModelConfig(
            model="google/gemma-7b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "gemma-3-12b-pt": ModelConfig(
            model="google/gemma-3-12b-pt",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "gemma-3-4b-pt": ModelConfig(
            model="google/gemma-3-4b-pt",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),

        # SmolLM
        "smollm-2-1.7b": ModelConfig(
            model="HuggingFaceTB/SmolLM2-1.7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "smollm-1.7b": ModelConfig(
            model="HuggingFaceTB/SmolLM-1.7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Marin
        "marin-8b": ModelConfig(
            model="marin-community/marin-8b-base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Microsoft (Orca / Phi)
        "orca-2-13b": ModelConfig(
            model="microsoft/Orca-2-13b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "orca-2-7b": ModelConfig(
            model="microsoft/Orca-2-7b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "phi-4": ModelConfig(
            model="microsoft/phi-4",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "phi-1.5": ModelConfig(
            model="microsoft/phi-1_5",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "phi-1": ModelConfig(
            model="microsoft/phi-1",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Mistral
        "codestral-22b": ModelConfig(
            model="mistralai/Codestral-22B-v0.1",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mathstral-7b": ModelConfig(
            model="mistralai/Mathstral-7B-v0.1",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mistral-7b": ModelConfig(
            model="mistralai/Mistral-7B-v0.3",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mistral-7b-v0.1": ModelConfig(
            model="mistralai/Mistral-7B-v0.1",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mixtral-8x22b": ModelConfig(
            model="mistralai/Mixtral-8x22B-v0.1",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mixtral-8x7b": ModelConfig(
            model="mistralai/Mixtral-8x7B-v0.1",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mistral-nemo-base": ModelConfig(
            model="mistralai/Mistral-Nemo-Base-2407",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mistral-small-24b": ModelConfig(
            model="mistralai/Mistral-Small-24B-Base-2501",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "mistral-small-3.1-24b": ModelConfig(
            model="mistralai/Mistral-Small-3.1-24B-Base-2503",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),

        # Qwen
        "codeqwen-1.5-7b": ModelConfig(
            model="Qwen/CodeQwen1.5-7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-0.5b": ModelConfig(
            model="Qwen/Qwen1.5-0.5B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-1.8b": ModelConfig(
            model="Qwen/Qwen1.5-1.8B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-4b": ModelConfig(
            model="Qwen/Qwen1.5-4B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-7b": ModelConfig(
            model="Qwen/Qwen1.5-7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-32b": ModelConfig(
            model="Qwen/Qwen1.5-32B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-72b": ModelConfig(
            model="Qwen/Qwen1.5-72B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-110b": ModelConfig(
            model="Qwen/Qwen1.5-110B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-1.5-moe-a2.7b": ModelConfig(
            model="Qwen/Qwen1.5-MoE-A2.7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2-0.5b": ModelConfig(
            model="Qwen/Qwen2-0.5B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2-1.5b": ModelConfig(
            model="Qwen/Qwen2-1.5B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2-7b": ModelConfig(
            model="Qwen/Qwen2-7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2-72b": ModelConfig(
            model="Qwen/Qwen2-72B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-0.5b": ModelConfig(
            model="Qwen/Qwen2.5-0.5B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-1.5b": ModelConfig(
            model="Qwen/Qwen2.5-1.5B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-3b": ModelConfig(
            model="Qwen/Qwen2.5-3B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-7b": ModelConfig(
            model="Qwen/Qwen2.5-7B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-14b": ModelConfig(
            model="Qwen/Qwen2.5-14B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-32b": ModelConfig(
            model="Qwen/Qwen2.5-32B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-2.5-72b": ModelConfig(
            model="Qwen/Qwen2.5-72B",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-3-0.6b": ModelConfig(
            model="Qwen/Qwen3-0.6B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-3-1.7b": ModelConfig(
            model="Qwen/Qwen3-1.7B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-3-4b": ModelConfig(
            model="Qwen/Qwen3-4B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-3-8b": ModelConfig(
            model="Qwen/Qwen3-8B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-3-14b": ModelConfig(
            model="Qwen/Qwen3-14B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "qwen-3-30b-a3b": ModelConfig(
            model="Qwen/Qwen3-30B-A3B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # StabilityAI
        "stablelm-2-1.6b": ModelConfig(
            model="stabilityai/stablelm-2-1_6b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "stablelm-base-alpha-7b": ModelConfig(
            model="stabilityai/stablelm-base-alpha-7b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Falcon3
        "falcon-3-10b": ModelConfig(
            model="tiiuae/Falcon3-10B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "falcon-3-7b": ModelConfig(
            model="tiiuae/Falcon3-7B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "falcon-3-3b": ModelConfig(
            model="tiiuae/Falcon3-3B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "falcon-3-1b": ModelConfig(
            model="tiiuae/Falcon3-1B-Base",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
    
        # Pythia
        "pythia-14m": ModelConfig(
            model="EleutherAI/pythia-14m",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-70m": ModelConfig(
            model="EleutherAI/pythia-70m",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-160m": ModelConfig(
            model="EleutherAI/pythia-160m",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-410m": ModelConfig(
            model="EleutherAI/pythia-410m",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-1b": ModelConfig(
            model="EleutherAI/pythia-1b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-1.4b": ModelConfig(
            model="EleutherAI/pythia-1.4b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-6.9b": ModelConfig(
            model="EleutherAI/pythia-6.9b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "pythia-12b": ModelConfig(
            model="EleutherAI/pythia-12b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        # HuggyLLaMA
        "huggyllama-7b": ModelConfig(
            model="huggyllama/llama-7b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "huggyllama-13b": ModelConfig(
            model="huggyllama/llama-13b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "huggyllama-30b": ModelConfig(
            model="huggyllama/llama-30b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        "huggyllama-65b": ModelConfig(
            model="huggyllama/llama-65b",
            provider=ProviderConfig(kind=ProviderKind.VLLM),
        ),
        # Aquila
        "aquila-7b": ModelConfig(
            model="BAAI/Aquila-7B",
            max_model_len=2048,
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

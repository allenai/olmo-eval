"""OLMo model constants for weights conversion and tokenizer configuration.

This module contains configuration for OLMo model families, including
Git repository locations, conversion scripts, default tokenizers,
and model presets for evaluation.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.core.configs import ModelConfig


class OlmoModelType(str, Enum):
    """Supported OLMo model architecture types."""

    OLMOE = "olmoe"
    OLMO2 = "olmo2"
    OLMO_CORE = "olmo-core"
    OLMO_CORE_V2 = "olmo-core-v2"


# =============================================================================
# External Repository URLs
# =============================================================================

TRANSFORMERS_GIT_URL = "https://github.com/huggingface/transformers.git"
"""HuggingFace transformers repository URL."""

TRANSFORMERS_COMMIT_HASH = "241c04d36867259cdf11dbb4e9d9a60f9cb65ebc"
"""Pinned transformers commit (v4.47.1)."""

AI2_OLMO_GIT_URL = "https://github.com/allenai/OLMo.git"
"""AI2 OLMo repository URL."""

AI2_OLMO_CORE_GIT_URL = "https://github.com/allenai/OLMo-core.git"
"""AI2 OLMo-core repository URL."""


# =============================================================================
# OLMoE Configuration
# =============================================================================

OLMOE_COMMIT_HASH = "04a2da53db172bd9a0450705592ed50888bdcaa7"
"""Pinned commit hash for OLMoE conversion."""

OLMOE_UNSHARD_SCRIPT = "scripts/unshard.py"
"""Path to OLMoE unsharding script within the OLMo repository."""

OLMOE_CONVERSION_SCRIPT = "src/transformers/models/olmoe/convert_olmoe_weights_to_hf.py"
"""Path to OLMoE HuggingFace conversion script within transformers."""

DEFAULT_OLMOE_TOKENIZER = "allenai/eleuther-ai-gpt-neox-20b-pii-special"
"""Default tokenizer for OLMoE models."""


# =============================================================================
# OLMo 2 Configuration
# =============================================================================

OLMO2_COMMIT_HASH = "69362b95c66655191d513e9c1420d54aa8477d92"
"""Pinned commit hash for OLMo 2 conversion."""

OLMO2_UNSHARD_SCRIPT = "scripts/unshard.py"
"""Path to OLMo 2 unsharding script within the OLMo repository."""

OLMO2_CONVERSION_SCRIPT = "src/transformers/models/olmo2/convert_olmo2_weights_to_hf.py"
"""Path to OLMo 2 HuggingFace conversion script within transformers."""

DEFAULT_OLMO2_TOKENIZER = "allenai/dolma2-tokenizer"
"""Default tokenizer for OLMo 2 models."""


# =============================================================================
# OLMo-Core Configuration
# =============================================================================

OLMO_CORE_COMMIT_HASH = "9bad23d9a78e62101699a585a8fde3d69dba5616"
"""Pinned commit hash for OLMo-core conversion."""

OLMO_CORE_V2_COMMIT_HASH = "1662d0d4f3e628ebb68591e311cce68737c094c4"
"""Pinned commit hash for OLMo-core v2 conversion."""

OLMO_CORE_UNSHARD_CONVERT_SCRIPT = "src/examples/huggingface/convert_checkpoint_to_hf.py"
"""Path to OLMo-core HuggingFace conversion script."""

OLMO_CORE_CONVERT_FROM_HF_SCRIPT = "src/examples/huggingface/convert_checkpoint_from_hf.py"
"""Path to script for converting HuggingFace checkpoints to OLMo-core format."""


class OlmoCoreDtype(str, Enum):
    """Supported data types for OLMo-core checkpoint conversion."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


DEFAULT_OLMO_CORE_TOKENIZER = "allenai/OLMo-2-1124-7B"
"""Default tokenizer for OLMo-core models."""


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

    return {
        # HuggingFace models (vLLM inference)
        "llama3.1-8b": ModelConfig(model="meta-llama/Meta-Llama-3.1-8B"),
        "llama3.1-8b-instruct": ModelConfig(model="meta-llama/Llama-3.1-8B-Instruct"),
        "llama3.1-70b": ModelConfig(model="meta-llama/Meta-Llama-3.1-70B"),
        "llama3.1-70b-instruct": ModelConfig(model="meta-llama/Llama-3.1-70B-Instruct"),
        "olmo-2-7b": ModelConfig(
            model="allenai/OLMo-2-1124-7B",
            tokenizer=DEFAULT_OLMO2_TOKENIZER,
            trust_remote_code=True,
        ),
        "olmo-2-13b": ModelConfig(
            model="allenai/OLMo-2-1124-13B",
            tokenizer=DEFAULT_OLMO2_TOKENIZER,
            trust_remote_code=True,
        ),
        "qwen2.5-7b": ModelConfig(model="Qwen/Qwen2.5-7B"),
        "mistral-7b": ModelConfig(model="mistralai/Mistral-7B-v0.3"),
        # Mock model for testing (no dependencies required)
        "mock": ModelConfig(model="mock", provider="mock"),
        # API-based models (for agent tasks - requires API keys)
        "gpt-4o": ModelConfig(
            model="gpt-4o",
            model_url="https://api.openai.com/v1",
            provider="litellm",
        ),
        "gpt-4o-mini": ModelConfig(
            model="gpt-4o-mini",
            model_url="https://api.openai.com/v1",
            provider="litellm",
        ),
        "gpt-4-turbo": ModelConfig(
            model="gpt-4-turbo",
            model_url="https://api.openai.com/v1",
            provider="litellm",
        ),
        "claude-3-opus": ModelConfig(
            model="claude-3-opus-20240229",
            model_url="https://api.anthropic.com",
            provider="litellm",
        ),
        "claude-3-sonnet": ModelConfig(
            model="claude-3-sonnet-20240229",
            model_url="https://api.anthropic.com",
            provider="litellm",
        ),
    }

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
    """
    from olmo_eval.core.configs import ModelConfig

    return {
        "llama3.1-8b": ModelConfig(model="meta-llama/Meta-Llama-3.1-8B"),
        "llama3.1-70b": ModelConfig(model="meta-llama/Meta-Llama-3.1-70B"),
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
        "yi-1.5-6b": ModelConfig(model="01-ai/Yi-1.5-6B"),
        "yi-1.5-9b": ModelConfig(model="01-ai/Yi-1.5-9B"),
        "yi-6b": ModelConfig(model="01-ai/Yi-6B"),
        "yi-9b": ModelConfig(model="01-ai/Yi-9B"),
        "deepseek-llm-7b-base": ModelConfig(model="deepseek-ai/deepseek-llm-7b-base"),
        "gemma2-9b": ModelConfig(model="google/gemma-2-9b"),
        "gemma-7b": ModelConfig(model="google/gemma-7b"),
        "marin-8b-base": ModelConfig(model="marin-community/marin-8b-base"),
        "llama3-8b": ModelConfig(model="meta-llama/Meta-Llama-3-8B"),
        "orca-2-7b": ModelConfig(model="microsoft/Orca-2-7b"),
        "mathstral-7b": ModelConfig(model="mistralai/Mathstral-7B-v0.1"),
        "mistral-7b-v0.1": ModelConfig(model="mistralai/Mistral-7B-v0.1"),
        "codeqwen1.5-7b": ModelConfig(model="Qwen/CodeQwen1.5-7B"),
        "qwen1.5-7b": ModelConfig(model="Qwen/Qwen1.5-7B"),
        "qwen2-7b": ModelConfig(model="Qwen/Qwen2-7B"),
        "qwen3-8b": ModelConfig(model="Qwen/Qwen3-8B-Base"),
        "stablelm-base-alpha-7b": ModelConfig(model="stabilityai/stablelm-base-alpha-7b"),
        "falcon3-7b": ModelConfig(model="tiiuae/Falcon3-7B-Base"),
        "falcon3-10b": ModelConfig(model="tiiuae/Falcon3-10B-Base"),
        "olmo-7b-0424": ModelConfig(model="allenai/OLMo-7B-0424-hf"),
        "olmo-7b-0724": ModelConfig(model="allenai/OLMo-7B-0724-hf"),
        "olmo-7b-hf": ModelConfig(model="allenai/OLMo-7B-hf", max_model_len=2048),
        "olmo-7b-twin-2t": ModelConfig(model="allenai/OLMo-7B-Twin-2T-hf", max_model_len=2048),
        "olmo-2-7b-stage1-step928646": ModelConfig(
            model="allenai/OLMo-2-1124-7B",
            revision="stage1-step928646-tokens3896B",
            tokenizer=DEFAULT_OLMO2_TOKENIZER,
            trust_remote_code=True,
        ),
        "olmo-2-7b-stage2-ingredient1": ModelConfig(
            model="allenai/OLMo-2-1124-7B",
            revision="stage2-ingredient1-step11931-tokens50B",
            tokenizer=DEFAULT_OLMO2_TOKENIZER,
            trust_remote_code=True,
        ),
        "olmo-2-7b-stage2-ingredient2": ModelConfig(
            model="allenai/OLMo-2-1124-7B",
            revision="stage2-ingredient2-step11931-tokens50B",
            tokenizer=DEFAULT_OLMO2_TOKENIZER,
            trust_remote_code=True,
        ),
        "olmo-2-7b-stage2-ingredient3": ModelConfig(
            model="allenai/OLMo-2-1124-7B",
            revision="stage2-ingredient3-step11931-tokens50B",
            tokenizer=DEFAULT_OLMO2_TOKENIZER,
            trust_remote_code=True,
        ),
        "pythia-6.9b": ModelConfig(model="EleutherAI/pythia-6.9b", max_model_len=2048),
        "llama-7b": ModelConfig(model="huggyllama/llama-7b", max_model_len=2048),
        "aquila-7b": ModelConfig(model="BAAI/Aquila-7B", max_model_len=2048),
    }

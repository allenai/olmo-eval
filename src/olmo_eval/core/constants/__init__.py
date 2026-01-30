"""Constants for olmo-eval infrastructure and model configuration.

This module provides centralized access to core infrastructure constants:

- **infrastructure**: Beaker, cluster, and storage configuration
- **models**: OLMo model types, tokenizers, and conversion scripts

For evaluation-related constants (benchmarks, tasks), see `olmo_eval.evals.constants`.
"""

# Infrastructure constants
from olmo_eval.core.constants.infrastructure import (
    BACKEND_OPTIONAL_GROUPS,
    BEAKER_DEFAULT_BUDGET,
    BEAKER_DEFAULT_PRIORITY,
    BEAKER_DEFAULT_WORKSPACE,
    BEAKER_GANTRY_MAX_VERSION,
    BEAKER_GANTRY_MIN_VERSION,
    BEAKER_KNOWN_CLUSTERS,
    BEAKER_PY_MAX_VERSION,
    BEAKER_PY_MIN_VERSION,
    NEW_CLUSTER_ALIASES,
    OE_EVAL_COMMIT_HASH,
    OE_EVAL_GIT_URL,
    OE_EVAL_LAUNCH_COMMAND,
    WEKA_MOUNTS,
    BeakerPriority,
)

# Model constants
from olmo_eval.core.constants.models import (
    AI2_OLMO_CORE_GIT_URL,
    AI2_OLMO_GIT_URL,
    DEFAULT_OLMO2_TOKENIZER,
    DEFAULT_OLMO_CORE_TOKENIZER,
    DEFAULT_OLMOE_TOKENIZER,
    OLMO2_COMMIT_HASH,
    OLMO2_CONVERSION_SCRIPT,
    OLMO2_UNSHARD_SCRIPT,
    OLMO_CORE_COMMIT_HASH,
    OLMO_CORE_CONVERT_FROM_HF_SCRIPT,
    OLMO_CORE_UNSHARD_CONVERT_SCRIPT,
    OLMO_CORE_V2_COMMIT_HASH,
    OLMOE_COMMIT_HASH,
    OLMOE_CONVERSION_SCRIPT,
    OLMOE_UNSHARD_SCRIPT,
    TRANSFORMERS_COMMIT_HASH,
    TRANSFORMERS_GIT_URL,
    OlmoCoreDtype,
    OlmoModelType,
    get_model_presets,
)

__all__ = [
    # Infrastructure
    "BACKEND_OPTIONAL_GROUPS",
    "BEAKER_DEFAULT_BUDGET",
    "BEAKER_DEFAULT_PRIORITY",
    "BEAKER_DEFAULT_WORKSPACE",
    "BEAKER_GANTRY_MAX_VERSION",
    "BEAKER_GANTRY_MIN_VERSION",
    "BEAKER_KNOWN_CLUSTERS",
    "BEAKER_PY_MAX_VERSION",
    "BEAKER_PY_MIN_VERSION",
    "BeakerPriority",
    "NEW_CLUSTER_ALIASES",
    "OE_EVAL_COMMIT_HASH",
    "OE_EVAL_GIT_URL",
    "OE_EVAL_LAUNCH_COMMAND",
    "WEKA_MOUNTS",
    # Models
    "AI2_OLMO_CORE_GIT_URL",
    "AI2_OLMO_GIT_URL",
    "DEFAULT_OLMO2_TOKENIZER",
    "DEFAULT_OLMO_CORE_TOKENIZER",
    "DEFAULT_OLMOE_TOKENIZER",
    "OLMO2_COMMIT_HASH",
    "OLMO2_CONVERSION_SCRIPT",
    "OLMO2_UNSHARD_SCRIPT",
    "OLMO_CORE_COMMIT_HASH",
    "OLMO_CORE_CONVERT_FROM_HF_SCRIPT",
    "OLMO_CORE_UNSHARD_CONVERT_SCRIPT",
    "OLMO_CORE_V2_COMMIT_HASH",
    "OLMOE_COMMIT_HASH",
    "OLMOE_CONVERSION_SCRIPT",
    "OLMOE_UNSHARD_SCRIPT",
    "OlmoCoreDtype",
    "OlmoModelType",
    "TRANSFORMERS_COMMIT_HASH",
    "TRANSFORMERS_GIT_URL",
    "get_model_presets",
]

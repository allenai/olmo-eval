"""Constants for olmo-eval infrastructure and model configuration.

For evaluation-related constants (benchmarks, tasks), see `olmo_eval.evals.constants`.
"""

# Infrastructure constants
from olmo_eval.common.constants.infrastructure import (
    BACKEND_OPTIONAL_GROUPS,
    BEAKER_DEFAULT_BUDGET,
    BEAKER_DEFAULT_PRIORITY,
    BEAKER_DEFAULT_WORKSPACE,
    BEAKER_GANTRY_MAX_VERSION,
    BEAKER_GANTRY_MIN_VERSION,
    BEAKER_KNOWN_CLUSTERS,
    BEAKER_PY_MAX_VERSION,
    BEAKER_PY_MIN_VERSION,
    BEAKER_RESULT_DIR,
    LOCAL_RESULT_DIR,
    NEW_CLUSTER_ALIASES,
    OE_EVAL_COMMIT_HASH,
    OE_EVAL_GIT_URL,
    OE_EVAL_LAUNCH_COMMAND,
    WEKA_CLUSTERS,
    WEKA_MOUNTS,
    BeakerPriority,
    cluster_has_weka,
)

# Model constants
from olmo_eval.common.constants.models import get_model_presets

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
    "BEAKER_RESULT_DIR",
    "BeakerPriority",
    "LOCAL_RESULT_DIR",
    "NEW_CLUSTER_ALIASES",
    "OE_EVAL_COMMIT_HASH",
    "OE_EVAL_GIT_URL",
    "OE_EVAL_LAUNCH_COMMAND",
    "WEKA_CLUSTERS",
    "WEKA_MOUNTS",
    "cluster_has_weka",
    "get_model_presets",
]

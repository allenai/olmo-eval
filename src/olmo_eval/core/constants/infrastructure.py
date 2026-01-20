"""Infrastructure constants for Beaker, clusters, and storage mounts.

This module contains configuration constants for AI2's compute infrastructure,
including Beaker workspace settings, cluster definitions, and Weka storage mounts.
"""

from enum import Enum

# =============================================================================
# Beaker Configuration
# =============================================================================

BEAKER_DEFAULT_WORKSPACE = "ai2/oe-data"
"""Default Beaker workspace for evaluation jobs."""

BEAKER_DEFAULT_BUDGET = "ai2/oe-base"
"""Default budget allocation for Beaker jobs."""

BEAKER_DEFAULT_PRIORITY = "normal"
"""Default job priority level."""

BEAKER_DEFAULT_IMAGE = "ai2-tylerm/olmo-eval-cu1281-trc290-amd64"
"""Default Docker image for Beaker evaluation jobs."""

BEAKER_RESULT_DIR = "/results"
"""Default directory for evaluation results in Beaker jobs."""

DEFAULT_MAX_GPUS_PER_NODE = 8
"""Default maximum GPUs available per node for experiment splitting."""


class BeakerPriority(str, Enum):
    """Valid Beaker job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


# Version constraints for Beaker tooling
BEAKER_PY_MIN_VERSION = "2.5.0"
BEAKER_PY_MAX_VERSION = "3"
BEAKER_GANTRY_MIN_VERSION = "3.2.0"
BEAKER_GANTRY_MAX_VERSION = "4"


# =============================================================================
# Cluster Definitions
# =============================================================================

BEAKER_KNOWN_CLUSTERS: dict[str, list[str]] = {
    "aus": [
        "ai2/jupiter",
        "ai2/neptune",
        "ai2/saturn",
        "ai2/ceres",
    ],
    "aus80g": [
        "ai2/jupiter",
        "ai2/saturn",
        "ai2/ceres",
    ],
    "h100": [
        "ai2/jupiter",
        "ai2/ceres",
    ],
    "a100": [
        "ai2/saturn",
    ],
    "l40": [
        "ai2/neptune",
    ],
    "80g": [
        "ai2/jupiter",
        "ai2/saturn",
        "ai2/ceres",
    ],
}
"""Mapping of cluster group aliases to their constituent Beaker clusters."""


NEW_CLUSTER_ALIASES: dict[str, str] = {
    "ai2/allennlp-elanding-a100-40g": "ai2/allennlp",
    "ai2/allennlp-dev-a100-sea": "ai2/allennlp",
    "ai2/ceres-dev-h100-aus-ib": "ai2/ceres",
    "ai2/ceres-cirrascale": "ai2/ceres",
    "ai2/jupiter-batch-h100-aus-ib": "ai2/jupiter",
    "ai2/jupiter-cirrascale-2": "ai2/jupiter",
    "ai2/neptune-dev-l40-aus": "ai2/neptune",
    "ai2/neptune-cirrascale": "ai2/neptune",
    "ai2/phobos-dev-aus": "ai2/phobos",
    "ai2/phobos-cirrascale": "ai2/phobos",
    "ai2/prior-dev-a6000-sea": "ai2/prior",
    "ai2/prior-elanding": "ai2/prior",
    "ai2/prior-elanding-rtx8000": "ai2/prior-rtx8000",
    "ai2/prior-dev-rtx8000-sea": "ai2/prior-rtx8000",
    "ai2/rhea-dev-a6000-aus": "ai2/rhea",
    "ai2/rhea-cirrascale": "ai2/rhea",
    "ai2/saturn-dev-a100-aus": "ai2/saturn",
    "ai2/saturn-cirrascale": "ai2/saturn",
    "ai2/titan-batch-b200-aus-ib": "ai2/titan",
    "ai2/titan-cirrascale": "ai2/titan",
    "ai2/triton-dev-l40-aus": "ai2/triton",
    "ai2/triton-cirrascale": "ai2/triton",
}
"""Mapping of specific cluster names to their canonical short names."""


# =============================================================================
# Weka Storage Mounts
# =============================================================================

WEKA_MOUNTS: tuple[str, ...] = (
    "ai1-default",
    "climate-default",
    "dfive-default",
    "mosaic-default",
    "nora-default",
    "nsf-uchicago-apto",
    "oe-adapt-default",
    "oe-data-default",
    "oe-eval-default",
    "oe-training-default",
    "prior-default",
    "reviz-default",
    "skylight-default",
)
"""Available Weka storage mount points for Beaker jobs."""


# =============================================================================
# OE-Eval Configuration
# =============================================================================

OE_EVAL_GIT_URL = "git@github.com:allenai/oe-eval-internal.git"
"""Git URL for the oe-eval-internal repository."""

OE_EVAL_COMMIT_HASH: str | None = None
"""Pinned commit hash for oe-eval (None means latest)."""

OE_EVAL_LAUNCH_COMMAND = "oe_eval/launch.py"
"""Entry point script for launching evaluations."""


# =============================================================================
# Backend Dependencies
# =============================================================================

BACKEND_OPTIONAL_GROUPS: dict[str, str | None] = {
    "vllm": "vllm",
    "hf": "hf",
    "litellm": "litellm",
    "mock": None,  # No dependency needed for mock backend
}
"""Mapping of backend types to their pyproject.toml optional dependency group names.

Used for auto-installing backend dependencies at runtime when launching
Beaker jobs via `pip install .[group_name]`.
"""

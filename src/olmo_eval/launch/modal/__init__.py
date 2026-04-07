"""Modal launch utilities for olmo-eval.

This module provides utilities for launching evaluation jobs on Modal,
including secret management for container registry authentication.
"""

from olmo_eval.launch.modal.secrets import (
    ensure_modal_gcp_secret,
    get_local_gcp_credentials,
)

__all__ = [
    "get_local_gcp_credentials",
    "ensure_modal_gcp_secret",
]

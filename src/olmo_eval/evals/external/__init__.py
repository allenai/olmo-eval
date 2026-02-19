"""External black-box evaluation integration.

This module provides support for running external evaluations that install
themselves in a sandbox container while communicating with a model provider
running in the parent process.
"""

from olmo_eval.evals.external.base import ExternalEval, SandboxedExternalEval
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.registry import (
    clear_registry,
    get_external_eval,
    is_external_eval_registered,
    list_external_evals,
    load_external_evals,
    register_external_eval,
)
from olmo_eval.evals.external.result import ExternalEvalResult

__all__ = [
    "ExternalEval",
    "ExternalEvalResult",
    "SandboxedExternalEval",
    "clear_registry",
    "get_docker_network_args",
    "get_external_eval",
    "is_external_eval_registered",
    "list_external_evals",
    "load_external_evals",
    "register_external_eval",
]

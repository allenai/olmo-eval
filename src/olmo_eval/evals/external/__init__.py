"""External black-box evaluation integration.

This module provides support for running external evaluations that install
themselves in a sandbox container while communicating with a model provider
running in the parent process.
"""

# Import benchmarks to trigger registration
from olmo_eval.evals.external import benchmarks as _benchmarks  # noqa: F401
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.network import get_docker_network_args
from olmo_eval.evals.external.registry import (
    get_external_eval,
    is_external_eval_registered,
    list_external_evals,
    register_external_eval,
)
from olmo_eval.evals.external.result import ExternalEvalResult

__all__ = [
    "ExternalEval",
    "ExternalEvalResult",
    "get_external_eval",
    "list_external_evals",
    "is_external_eval_registered",
    "register_external_eval",
    "get_docker_network_args",
]

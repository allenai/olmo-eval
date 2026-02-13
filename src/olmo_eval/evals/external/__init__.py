"""External black-box evaluation integration.

This module provides support for running external evaluations that install
themselves in a sandbox container while communicating with a model provider
running in the parent process.

Example usage:
    from olmo_eval.evals.external import get_external_eval

    eval = get_external_eval("tau2_bench")
    result = await eval.execute(
        provider_url="http://localhost:8000",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
"""

# Import benchmarks to trigger registration
from olmo_eval.evals.external import benchmarks as _benchmarks  # noqa: F401
from olmo_eval.evals.external.base import ExternalEval
from olmo_eval.evals.external.config import ExternalEvalConfig, ExternalEvalRunConfig
from olmo_eval.evals.external.default import BaseExternalEval
from olmo_eval.evals.external.network import (
    get_docker_network_args,
    get_provider_url_for_container,
    is_running_in_beaker,
    translate_url_for_container,
)
from olmo_eval.evals.external.registry import (
    get_external_config,
    get_external_eval,
    is_external_eval_registered,
    list_external_evals,
    register_external_config,
)
from olmo_eval.evals.external.result import ExternalEvalResult

__all__ = [
    "ExternalEval",
    "ExternalEvalConfig",
    "ExternalEvalResult",
    "ExternalEvalRunConfig",
    "BaseExternalEval",
    "get_external_eval",
    "get_external_config",
    "list_external_evals",
    "is_external_eval_registered",
    "register_external_config",
    "get_provider_url_for_container",
    "translate_url_for_container",
    "is_running_in_beaker",
    "get_docker_network_args",
]

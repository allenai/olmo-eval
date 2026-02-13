"""Registry for external evaluation configurations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.evals.external.base import ExternalEval
    from olmo_eval.evals.external.config import ExternalEvalConfig

# Global registry of external eval configs
_EXTERNAL_EVAL_CONFIGS: dict[str, ExternalEvalConfig] = {}


def register_external_config(name: str, config: ExternalEvalConfig) -> None:
    """Register an external evaluation configuration.

    Args:
        name: Unique name for the evaluation.
        config: Configuration for the evaluation.

    Raises:
        ValueError: If an evaluation with this name is already registered.
    """
    if name in _EXTERNAL_EVAL_CONFIGS:
        raise ValueError(f"External eval '{name}' is already registered")
    _EXTERNAL_EVAL_CONFIGS[name] = config


def get_external_config(name: str) -> ExternalEvalConfig:
    """Get an external evaluation configuration by name.

    Args:
        name: Name of the evaluation.

    Returns:
        The evaluation configuration.

    Raises:
        KeyError: If the evaluation is not registered.
    """
    if name not in _EXTERNAL_EVAL_CONFIGS:
        available = ", ".join(sorted(_EXTERNAL_EVAL_CONFIGS.keys()))
        raise KeyError(f"External eval '{name}' not found. Available: {available or '(none)'}")
    return _EXTERNAL_EVAL_CONFIGS[name]


def get_external_eval(name: str) -> ExternalEval:
    """Get an external evaluation instance by name.

    This creates a BaseExternalEval using the registered configuration.

    Args:
        name: Name of the evaluation.

    Returns:
        An ExternalEval instance ready to execute.

    Raises:
        KeyError: If the evaluation is not registered.
    """
    from olmo_eval.evals.external.default import BaseExternalEval

    config = get_external_config(name)
    return BaseExternalEval(config)


def list_external_evals() -> list[str]:
    """List all registered external evaluation names.

    Returns:
        Sorted list of evaluation names.
    """
    return sorted(_EXTERNAL_EVAL_CONFIGS.keys())


def is_external_eval_registered(name: str) -> bool:
    """Check if an external evaluation is registered.

    Args:
        name: Name of the evaluation.

    Returns:
        True if the evaluation is registered.
    """
    return name in _EXTERNAL_EVAL_CONFIGS

"""Registry for external evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.evals.external.base import ExternalEval

# Global registry of external evals
_EXTERNAL_EVALS: dict[str, ExternalEval] = {}


def register_external_eval(eval_instance: ExternalEval) -> None:
    """Register an external evaluation.

    Args:
        eval_instance: The evaluation instance to register.

    Raises:
        ValueError: If an evaluation with this name is already registered.
    """
    name = eval_instance.name
    if name in _EXTERNAL_EVALS:
        raise ValueError(f"External eval '{name}' is already registered")
    _EXTERNAL_EVALS[name] = eval_instance


def get_external_eval(name: str) -> ExternalEval:
    """Get an external evaluation by name.

    Args:
        name: Name of the evaluation.

    Returns:
        The evaluation instance.

    Raises:
        KeyError: If the evaluation is not registered.
    """
    if name not in _EXTERNAL_EVALS:
        available = ", ".join(sorted(_EXTERNAL_EVALS.keys()))
        raise KeyError(f"External eval '{name}' not found. Available: {available or '(none)'}")
    return _EXTERNAL_EVALS[name]


def list_external_evals() -> list[str]:
    """List all registered external evaluation names.

    Returns:
        Sorted list of evaluation names.
    """
    return sorted(_EXTERNAL_EVALS.keys())


def is_external_eval_registered(name: str) -> bool:
    """Check if an external evaluation is registered.

    Args:
        name: Name of the evaluation.

    Returns:
        True if the evaluation is registered.
    """
    return name in _EXTERNAL_EVALS

"""Registry for external evaluations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmo_eval.evals.external.base import ExternalEval

# Global registry of external evals
_EXTERNAL_EVALS: dict[str, ExternalEval] = {}
_evals_loaded = False


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


def load_external_evals() -> None:
    """Import all benchmark modules to register their evals.

    This is called automatically when accessing evals via get_external_eval(),
    but can be called explicitly to ensure all evals are registered.
    """
    global _evals_loaded
    if _evals_loaded:
        return

    import importlib
    import pkgutil

    import olmo_eval.evals.external.benchmarks as benchmarks_pkg

    for module_info in pkgutil.iter_modules(benchmarks_pkg.__path__):
        if not module_info.name.startswith("_"):
            importlib.import_module(f".{module_info.name}", benchmarks_pkg.__name__)

    _evals_loaded = True


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
        load_external_evals()

    if name not in _EXTERNAL_EVALS:
        available = ", ".join(sorted(_EXTERNAL_EVALS.keys()))
        raise KeyError(f"External eval '{name}' not found. Available: {available or '(none)'}")
    return _EXTERNAL_EVALS[name]


def list_external_evals() -> list[str]:
    """List all registered external evaluation names.

    Returns:
        Sorted list of evaluation names.
    """
    load_external_evals()
    return sorted(_EXTERNAL_EVALS.keys())


def is_external_eval_registered(name: str) -> bool:
    """Check if an external evaluation is registered.

    Args:
        name: Name of the evaluation.

    Returns:
        True if the evaluation is registered.
    """
    if name not in _EXTERNAL_EVALS:
        load_external_evals()
    return name in _EXTERNAL_EVALS


def clear_registry() -> None:
    """Clear all registered evals and reset loading state.

    Primarily useful for testing.
    """
    global _evals_loaded
    _EXTERNAL_EVALS.clear()
    _evals_loaded = False

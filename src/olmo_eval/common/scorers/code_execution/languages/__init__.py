"""Language evaluator registry with auto-discovery."""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LanguageEvaluator

_EVALUATORS: dict[str, LanguageEvaluator] = {}
_discovered = False


def register_evaluator(evaluator: LanguageEvaluator) -> None:
    """Register a language evaluator."""
    _EVALUATORS[evaluator.LANG_ID] = evaluator


def get_evaluator(lang_id: str) -> LanguageEvaluator:
    """Get evaluator for a language ID."""
    _auto_discover_evaluators()
    if lang_id not in _EVALUATORS:
        raise ValueError(f"No evaluator registered for language: {lang_id}")
    return _EVALUATORS[lang_id]


def list_languages() -> list[str]:
    """List all registered language IDs."""
    _auto_discover_evaluators()
    return sorted(_EVALUATORS.keys())


def _auto_discover_evaluators() -> None:
    """Import all language modules to trigger registration."""
    global _discovered
    if _discovered:
        return
    _discovered = True

    package = importlib.import_module(__name__.rsplit(".", 1)[0] + ".languages")
    package_path = package.__path__

    for _, module_name, _ in pkgutil.iter_modules(package_path):
        if module_name not in ("base", "__init__"):
            module = importlib.import_module(f"{package.__name__}.{module_name}")
            # Look for an 'evaluator' attribute in the module
            if hasattr(module, "evaluator"):
                register_evaluator(module.evaluator)

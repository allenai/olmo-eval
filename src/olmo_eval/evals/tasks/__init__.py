"""Task framework for evaluation."""

import importlib
import pkgutil
from pathlib import Path


def _discover_and_load_tasks() -> None:
    """Auto-discover and import all task modules to trigger registration."""
    package_dir = Path(__file__).parent

    for _finder, module_name, _is_pkg in pkgutil.iter_modules([str(package_dir)]):
        # Skip the common subpackage and private modules
        if module_name == "common" or module_name.startswith("_"):
            continue

        # Import the module (triggers @register decorators)
        importlib.import_module(f".{module_name}", package=__package__)


# Auto-discover and load all task modules
_discover_and_load_tasks()

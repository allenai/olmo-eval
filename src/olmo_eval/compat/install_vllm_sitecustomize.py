"""Install olmo-eval's opt-in startup hook in an isolated vLLM venv."""

from __future__ import annotations

import os
import site
from pathlib import Path


def main() -> None:
    pythonpath = os.environ.get("PYTHONPATH", "")
    source_root = Path(pythonpath.split(os.pathsep, 1)[0])
    source = source_root / "sitecustomize.py"
    if not source.is_file():
        raise FileNotFoundError(f"Missing olmo-eval startup hook: {source}")

    target = Path(site.getsitepackages()[0]) / "sitecustomize.py"
    target.unlink(missing_ok=True)
    target.symlink_to(source)
    print(f"Installed olmo-eval startup hook: {target} -> {source}")


if __name__ == "__main__":
    main()

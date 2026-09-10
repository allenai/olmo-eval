"""Clone helper for the OpenHands/benchmarks OpenAgentSafety runner."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_URL = "https://github.com/OpenHands/benchmarks.git"
# Pin the OpenHands/benchmarks revision that this wrapper was developed against.
DEFAULT_REF = "405bae7140d7e961a75f4910a0b2e7069731db96"
DEFAULT_CACHE_DIR = Path("/tmp") / "olmo-eval-openhands-benchmarks"


def ensure_repo(target_dir: Path, ref: str | None = None) -> Path:
    """Clone or update OpenHands/benchmarks at ``ref``."""
    ref = ref or DEFAULT_REF
    target_dir = target_dir.expanduser().resolve()

    if (target_dir / ".git").exists():
        logger.info("Updating OpenHands/benchmarks at %s", target_dir)
        _run(["git", "-C", str(target_dir), "fetch", "origin"])
        _run(["git", "-C", str(target_dir), "checkout", ref])
        return target_dir

    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info("Using existing OpenHands/benchmarks checkout at %s", target_dir)
        return target_dir

    logger.info("Cloning OpenHands/benchmarks to %s", target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", REPO_URL, str(target_dir)])
    _run(["git", "-C", str(target_dir), "checkout", ref])
    return target_dir


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Command failed ({' '.join(command)}): {stderr}")

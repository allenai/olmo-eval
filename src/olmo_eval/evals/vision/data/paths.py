"""Locations of the on-disk multimodal benchmark data.

Data is read from ``$MOLMO_DATA_DIR`` (default ``/weka/oe-training-default/mm-olmo``)
and never written -- loaders error out rather than build caches.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MOLMO_DATA_DIR = "/weka/oe-training-default/mm-olmo"


def molmo_data_dir() -> Path:
    """Root of the mm-olmo data tree (read-only)."""
    return Path(os.environ.get("MOLMO_DATA_DIR", DEFAULT_MOLMO_DATA_DIR))


def torch_datasets_dir() -> Path:
    return molmo_data_dir() / "torch_datasets"


def rebase_data_path(path: str) -> str:
    """Rebase an absolute path recorded on another machine onto the current root.

    Cached manifests (e.g. ``vqa2/molmo_val.json``) store absolute image paths
    from the machine that built them; if the stored path does not exist locally
    but contains ``torch_datasets/``, re-anchor it under the current data root.
    """
    if os.path.exists(path):
        return path
    marker = "torch_datasets/"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        return str(torch_datasets_dir() / suffix)
    return path

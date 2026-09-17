"""Locations of the on-disk multimodal benchmark data.

Data is read from ``$MOLMO_DATA_DIR`` (default ``/weka/oe-training-default/mm-olmo``)
and never written -- loaders error out rather than build caches.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_MOLMO_DATA_DIR = "/weka/oe-training-default/mm-olmo"


def molmo_data_dir() -> Path:
    """The multimodal data root, from ``$MOLMO_DATA_DIR``.

    Fails fast with the variable named when the root does not exist, since every
    vision task reads under it. The Ai2 default is tracked for removal in
    allenai/olmo-eval#381.
    """
    root = Path(os.environ.get("MOLMO_DATA_DIR", DEFAULT_MOLMO_DATA_DIR))
    if not root.is_dir():
        raise FileNotFoundError(
            f"Multimodal data root {root} does not exist; set MOLMO_DATA_DIR to the "
            "directory holding torch_datasets/ (see the README's Multimodal Evaluation section)."
        )
    return root


def torch_datasets_dir() -> Path:
    return molmo_data_dir() / "torch_datasets"

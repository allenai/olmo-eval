"""Lazily resolved benchmark image references.

An instance stores its image as an :class:`~olmo_eval.common.images.ImageReference`
(such as :class:`~olmo_eval.common.images.HfDatasetImage` or
:class:`~olmo_eval.common.images.PathImage`), as ``metadata["image_path"]`` (a
filesystem path), or as an inline PIL image in tests. Using typed
``ImageReference`` descriptors keeps ``Instance`` objects picklable across worker
processes and gives JSONL request writers a structured serialization form.
"""

from __future__ import annotations

from typing import Any

from olmo_eval.common.images import (
    CappedImageList,
    HfDatasetImage,
    HfDatasetImageList,
    resolve_image,
)
from olmo_eval.common.types import Instance

__all__ = [
    "CappedImageList",
    "HfDatasetImage",
    "HfDatasetImageList",
    "capped_image_list",
    "lazy_hf_image",
    "lazy_hf_image_list",
    "load_instance_image",
]


def lazy_hf_image(dataset: Any, index: int, column: str = "image") -> HfDatasetImage:
    """Create a typed lazy reference to one image cell of a no-decode HF dataset.

    ``dataset`` should have ``column`` cast to ``datasets.Image(decode=False)`` so
    building instances never decodes pixels; the provider decodes the cell lazily via
    ``resolve_images``.
    """
    return HfDatasetImage(dataset=dataset, index=index, column=column)


def load_instance_image(instance: Instance) -> Any:
    """Resolve an instance's image to a PIL image (or None if imageless)."""
    image = instance.metadata.get("image")
    if image is None:
        image = instance.metadata.get("image_path")
    return resolve_image(image)


def lazy_hf_image_list(dataset: Any, index: int, column: str) -> HfDatasetImageList:
    """Create a typed lazy reference to one list-of-images cell of a no-decode HF dataset.

    ``dataset`` should have ``column`` cast to a sequence of
    ``datasets.Image(decode=False)`` so building instances never decodes pixels.
    """
    return HfDatasetImageList(dataset=dataset, index=index, column=column)


def capped_image_list(images: Any, max_images: int) -> CappedImageList:
    """Create a typed lazy reference to an example's image list capped at ``max_images``."""
    return CappedImageList(images=images, max_images=max_images)

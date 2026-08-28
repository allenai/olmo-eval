"""Lazily resolved benchmark images.

An instance stores its image either as ``metadata["image_path"]`` (a filesystem
path) or ``metadata["image"]`` (a PIL image, or a zero-argument callable
returning one). Callables must be module-level ``functools.partial``s, not
closures, so the owning instance stays picklable across worker processes.
"""

from __future__ import annotations

import functools
from typing import Any

from olmo_eval.common.types import Instance


def _decode_hf_image_cell(dataset: Any, index: int, column: str):
    """Decode one image cell of a no-decode HF dataset (module-level so it is picklable)."""
    import io

    from PIL import Image

    rec = dataset[index][column]
    if isinstance(rec, dict):
        if rec.get("bytes"):
            return Image.open(io.BytesIO(rec["bytes"]))
        if rec.get("path"):
            return Image.open(rec["path"])
    return rec


def lazy_hf_image(dataset, index: int, column: str = "image"):
    """A picklable zero-arg callable that decodes one image cell of a no-decode HF dataset.

    ``dataset`` should have ``column`` cast to ``datasets.Image(decode=False)`` so building
    instances never decodes pixels; the callable decodes exactly one image when called.

    Returns a ``functools.partial`` over a module-level function (not a local closure) so the
    owning ``Instance`` stays picklable across the runner's worker processes — a closure would
    raise ``AttributeError: Can't get local object 'lazy_hf_image.<locals>._load'`` on pickle.
    """
    return functools.partial(_decode_hf_image_cell, dataset, index, column)


def load_instance_image(instance: Instance):
    """Resolve an instance's image to a PIL image (or None if imageless)."""
    image = instance.metadata.get("image")
    if image is not None:
        return image() if callable(image) else image
    path = instance.metadata.get("image_path")
    if path is not None:
        from PIL import Image

        return Image.open(path)
    return None


def _decode_hf_image_list_cell(dataset: Any, index: int, column: str) -> list:
    """Decode one list-of-images cell of a no-decode HF dataset (module-level so it
    is picklable)."""
    import io

    from PIL import Image

    out = []
    for rec in dataset[index][column]:
        if isinstance(rec, dict):
            if rec.get("bytes"):
                rec = Image.open(io.BytesIO(rec["bytes"]))
            elif rec.get("path"):
                rec = Image.open(rec["path"])
        out.append(rec)
    return out


def lazy_hf_image_list(dataset, index: int, column: str):
    """A picklable zero-arg callable that decodes one list-of-images cell.

    ``dataset`` should have ``column`` cast to a sequence of
    ``datasets.Image(decode=False)`` so building instances never decodes pixels.
    """
    return functools.partial(_decode_hf_image_list_cell, dataset, index, column)


def load_instance_images(instance: Instance) -> list:
    """Resolve an instance's image list to PIL images.

    ``instance.metadata["images"]`` may be a zero-arg callable returning the whole
    list, or a sequence whose items are PIL images, zero-arg callables, or
    filesystem paths.
    """
    images = instance.metadata.get("images")
    if images is None:
        return []
    if callable(images):
        images = images()
    resolved = []
    for image in images:
        if callable(image):
            image = image()
        elif isinstance(image, str):
            from PIL import Image

            image = Image.open(image)
        resolved.append(image)
    return resolved

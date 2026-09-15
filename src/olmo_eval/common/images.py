"""Resolution of lazy image references carried on ``LMRequest.images``.

An images entry may be a PIL image, a filesystem path (``str`` / ``Path``), or a
zero-argument callable returning a PIL image. Tasks attach the lazy forms so the
runner never decodes pixels in the parent process or pickles them through the
worker queue; providers resolve entries with :func:`resolve_images` right before
preprocessing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_image(entry: Any) -> Any:
    """Resolve one images entry to a PIL image (returns ``None`` unchanged)."""
    if entry is None:
        return None
    if callable(entry):
        entry = entry()
    if isinstance(entry, (str, Path)):
        from PIL import Image

        entry = Image.open(entry)
    return entry


def resolve_images(images: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
    """Resolve a request's images tuple; ``None`` stays ``None``."""
    if not images:
        return None
    return tuple(resolve_image(entry) for entry in images)

"""Typed lazy image references carried on ``LMRequest.images`` and instance metadata.

An image entry is normally an :class:`ImageReference` — a lightweight, structurally
picklable, JSON-serializable descriptor resolved lazily inside worker processes via
:func:`resolve_images` right before model preprocessing. Raw paths (``str`` /
``Path``), inline PIL images, and legacy zero-argument callables remain accepted for
backward compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "CappedImageList",
    "HfDatasetImage",
    "HfDatasetImageList",
    "ImageList",
    "ImageReference",
    "PathImage",
    "encode_image_entry",
    "resolve_image",
    "resolve_images",
]


def _hf_dataset_metadata(dataset: Any) -> dict[str, str]:
    """Extract safe JSON metadata from a HuggingFace dataset without touching rows."""
    extra: dict[str, str] = {}
    info = getattr(dataset, "info", None)
    dataset_name = getattr(info, "dataset_name", None)
    if isinstance(dataset_name, str) and dataset_name:
        extra["dataset"] = dataset_name
    split = getattr(dataset, "split", None)
    if isinstance(split, str) and split:
        extra["split"] = split
    return extra


def _open_path_image(path: str | Path) -> Any:
    """Open an image path, load its pixel data, and close the underlying file handle."""
    from PIL import Image

    with Image.open(path) as img:
        copied = img.copy()
        copied.format = img.format
        return copied


def _decode_hf_record(rec: Any) -> Any:
    """Decode one cell from a ``datasets.Image(decode=False)`` column."""
    if isinstance(rec, dict):
        import io

        from PIL import Image

        if rec.get("bytes"):
            return Image.open(io.BytesIO(rec["bytes"]))
        if rec.get("path"):
            return _open_path_image(rec["path"])
    return rec


def encode_image_entry(entry: Any) -> Any:
    """Convert an image reference or nested entry into a JSON-serializable summary."""
    if isinstance(entry, ImageReference):
        return entry.to_json()
    if isinstance(entry, Path):
        return str(entry)
    if isinstance(entry, (list, tuple)):
        return [encode_image_entry(item) for item in entry]
    if entry is None or isinstance(entry, (str, int, float, bool)):
        return entry
    return f"<{type(entry).__name__}>"


class ImageReference(ABC):
    """Base contract for lazily resolved, picklable, JSON-serializable image references."""

    @abstractmethod
    def resolve(self) -> Any:
        """Resolve this reference to a PIL image or a sequence of images."""
        ...

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable metadata description of this reference."""
        ...


@dataclass(frozen=True, slots=True)
class PathImage(ImageReference):
    """Reference to an image file on disk opened lazily at inference time."""

    path: str | Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            object.__setattr__(self, "path", str(self.path))

    def resolve(self) -> Any:
        return _open_path_image(self.path)

    def to_json(self) -> dict[str, Any]:
        return {"type": "path", "path": str(self.path)}


@dataclass(frozen=True, slots=True)
class HfDatasetImage(ImageReference):
    """Reference to a single cell in an un-decoded HuggingFace dataset image column."""

    dataset: Any
    index: int
    column: str = "image"

    def resolve(self) -> Any:
        return _decode_hf_record(self.dataset[self.index][self.column])

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "hf_dataset_image",
            "index": self.index,
            "column": self.column,
            **_hf_dataset_metadata(self.dataset),
        }


@dataclass(frozen=True, slots=True)
class HfDatasetImageList(ImageReference):
    """Reference to a sequence cell in an un-decoded HuggingFace dataset image column."""

    dataset: Any
    index: int
    column: str

    def resolve(self, max_items: int | None = None) -> list[Any]:
        cells = self.dataset[self.index][self.column]
        if max_items is not None:
            cells = cells[:max_items]
        return [_decode_hf_record(rec) for rec in cells]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "hf_dataset_image_list",
            "index": self.index,
            "column": self.column,
            **_hf_dataset_metadata(self.dataset),
        }


@dataclass(frozen=True, slots=True)
class ImageList(ImageReference):
    """Ordered tuple of image references resolved lazily as a list."""

    items: tuple[Any, ...]

    def __post_init__(self) -> None:
        if isinstance(self.items, (str, bytes)):
            raise TypeError("ImageList items must be a sequence of image entries, not str/bytes")
        if not isinstance(self.items, tuple):
            object.__setattr__(self, "items", tuple(self.items))

    def resolve(self) -> list[Any]:
        return [resolve_image(item) for item in self.items]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "image_list",
            "items": [encode_image_entry(item) for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class CappedImageList(ImageReference):
    """Reference to a list of image entries truncated to at most ``max_images``."""

    images: Any
    max_images: int

    def __post_init__(self) -> None:
        if self.max_images < 0:
            raise ValueError(f"max_images must be non-negative, got {self.max_images}")
        if (
            isinstance(self.images, list)
            or isinstance(self.images, Iterable)
            and not isinstance(self.images, (tuple, str, bytes, dict, ImageReference))
        ):
            object.__setattr__(self, "images", tuple(self.images))

    def resolve(self) -> list[Any]:
        if self.max_images == 0:
            return []
        if isinstance(self.images, HfDatasetImageList):
            return self.images.resolve(max_items=self.max_images)
        if isinstance(self.images, ImageList):
            return [resolve_image(item) for item in self.images.items[: self.max_images]]
        resolved = self.images.resolve() if isinstance(self.images, ImageReference) else self.images
        if callable(resolved):
            resolved = resolved()
        if resolved is None:
            return []
        if isinstance(resolved, (str, Path, ImageReference)) or not isinstance(resolved, Sequence):
            return [resolve_image(resolved)][: self.max_images]
        return [resolve_image(item) for item in list(resolved)[: self.max_images]]

    def to_json(self) -> dict[str, Any]:
        return {
            "type": "capped_image_list",
            "max_images": self.max_images,
            "images": encode_image_entry(self.images),
        }


def resolve_image(entry: Any) -> Any:
    """Resolve one images entry to a PIL image, or a list of them.

    Handles :class:`ImageReference` instances, recursive lists/tuples, filesystem
    paths (``str`` / ``Path``), and legacy zero-argument callables. Returns ``None``
    unchanged.
    """
    if entry is None:
        return None
    if isinstance(entry, ImageReference):
        resolved = entry.resolve()
        # A custom or nested reference may resolve to further entries (e.g. paths/refs).
        if isinstance(resolved, (ImageReference, list, tuple)):
            return resolve_image(resolved)
        return resolved
    if callable(entry):
        return resolve_image(entry())
    if isinstance(entry, (list, tuple)):
        return [resolve_image(item) for item in entry]
    if isinstance(entry, (str, Path)):
        return PathImage(entry).resolve()
    return entry


def resolve_images(images: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
    """Resolve a request's images tuple, flattening lists and dropping empty slots.

    A benchmark with a variable image count per example (MMMU-Pro interleaves up
    to seven) attaches one lazy reference per slot, and the multi-image tasks
    attach a single lazy loader for the whole list. Unused slots resolve to
    ``None`` and must not reach the provider as holes in the image list.
    """
    if not images:
        return None
    resolved: list[Any] = []
    for entry in images:
        item = resolve_image(entry)
        if isinstance(item, (list, tuple)):
            resolved.extend(image for image in item if image is not None)
        elif item is not None:
            resolved.append(item)
    return tuple(resolved) or None

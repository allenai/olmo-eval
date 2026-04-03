"""Rich repr utilities for dataclasses."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields
from typing import Any, TypeVar

T = TypeVar("T")


def hide_unset(*, skip: frozenset[str] = frozenset()) -> Any:
    """Decorator that adds __rich_repr__ to hide unset fields.

    Hides fields that are None or empty collections (tuple, list, dict, frozenset).

    Args:
        skip: Field names to always skip in repr.

    Example:
        @hide_unset_fields()
        @dataclass
        class MyConfig:
            name: str
            value: int | None = None

        @hide_unset_fields(skip=frozenset({"_cache"}))
        @dataclass
        class CachedConfig:
            data: str
            _cache: dict = field(default_factory=dict)
    """

    def decorator(cls: type[T]) -> type[T]:
        def __rich_repr__(self: T) -> Iterator[tuple[str, Any]]:
            for f in fields(self):  # type: ignore[ty:invalid-argument-type]
                if f.name in skip:
                    continue
                value = getattr(self, f.name)
                if value is None:
                    continue
                if isinstance(value, (tuple, list, dict, frozenset)) and not value:
                    continue
                yield f.name, value

        cls.__rich_repr__ = __rich_repr__  # type: ignore[ty:unresolved-attribute]
        return cls

    return decorator

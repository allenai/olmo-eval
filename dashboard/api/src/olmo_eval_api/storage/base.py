"""Storage protocol for run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class ObjectInfo:
    size: int
    md5_b64: str | None


@dataclass(frozen=True)
class SignedUrl:
    url: str
    headers: dict[str, str]
    expires_at: datetime


class Storage(Protocol):
    bucket: str

    async def list_objects(self, prefix: str) -> dict[str, ObjectInfo]:
        """Map of object key -> info for every object under ``prefix``."""
        ...

    async def sign_upload(self, key: str, content_type: str, md5_b64: str) -> SignedUrl: ...

    async def sign_download(self, key: str, filename: str | None = None) -> SignedUrl: ...

    async def read_range(self, key: str, start: int, length: int) -> bytes:
        """Read ``length`` bytes starting at ``start``. Raises FileNotFoundError."""
        ...

    async def delete_prefix(self, prefix: str) -> int: ...

    async def aclose(self) -> None: ...

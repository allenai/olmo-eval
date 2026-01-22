"""Unified dataset loading from multiple sources."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from olmo_eval.data.backends.base import DataBackend
from olmo_eval.data.backends.gcs import GCSBackend
from olmo_eval.data.backends.huggingface import HuggingFaceBackend
from olmo_eval.data.backends.local import LocalBackend
from olmo_eval.data.backends.s3 import S3Backend
from olmo_eval.data.sources import DataSource, SourceType


class DataLoader:
    """Unified dataset loading from multiple sources.

    Automatically dispatches to the appropriate backend based on the source type.
    Backends are lazily instantiated and cached.

    Examples:
        >>> loader = DataLoader()
        >>> # Load from HuggingFace
        >>> source = DataSource(path="cais/mmlu", subset="abstract_algebra", split="test")
        >>> for doc in loader.load(source):
        ...     print(doc)
        >>>
        >>> # Load from local file
        >>> source = DataSource(path="/path/to/data.jsonl")
        >>> for doc in loader.load(source):
        ...     print(doc)
        >>>
        >>> # Load from S3
        >>> source = DataSource.from_uri("s3://bucket/data.jsonl")
        >>> for doc in loader.load(source):
        ...     print(doc)
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        """Initialize the data loader.

        Args:
            cache_dir: Optional cache directory for downloaded datasets.
        """
        self.cache_dir = cache_dir
        self._backends: dict[SourceType, DataBackend] = {}

    def load(
        self,
        source: DataSource | str,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load dataset documents from any source.

        Args:
            source: The data source to load from. Can be a DataSource object
                or a URI string that will be parsed.
            streaming: Whether to stream the data (if supported by backend).

        Yields:
            Raw document dictionaries from the dataset.

        Raises:
            ValueError: If the source type is unknown or unsupported.
        """
        if isinstance(source, str):
            source = DataSource.from_uri(source)

        if source.source_type is None:
            raise ValueError(f"Could not determine source type for: {source.path}")

        backend = self._get_backend(source.source_type)
        return backend.load(source, streaming=streaming)

    def _get_backend(self, source_type: SourceType) -> DataBackend:
        """Get or create a backend for the given source type."""
        if source_type not in self._backends:
            self._backends[source_type] = self._create_backend(source_type)
        return self._backends[source_type]

    def _create_backend(self, source_type: SourceType) -> DataBackend:
        """Create a new backend instance for the given source type."""
        if source_type == SourceType.HUGGINGFACE:
            return HuggingFaceBackend()
        elif source_type == SourceType.LOCAL:
            return LocalBackend()
        elif source_type == SourceType.S3:
            return S3Backend()
        elif source_type == SourceType.GCS:
            return GCSBackend()
        else:
            raise ValueError(f"Unknown source type: {source_type}")


# Global default loader instance
_default_loader: DataLoader | None = None


def get_default_loader() -> DataLoader:
    """Get the default global DataLoader instance."""
    global _default_loader
    if _default_loader is None:
        _default_loader = DataLoader()
    return _default_loader


def load_dataset(
    source: DataSource | str,
    streaming: bool = False,
) -> Iterator[dict[str, Any]]:
    """Load dataset documents using the default loader.

    Convenience function that uses the global default DataLoader.

    Args:
        source: The data source to load from.
        streaming: Whether to stream the data.

    Yields:
        Raw document dictionaries from the dataset.
    """
    return get_default_loader().load(source, streaming=streaming)

"""Base protocol for data loading backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource


@runtime_checkable
class DataBackend(Protocol):
    """Protocol for dataset loading backends.

    Each backend implementation handles loading from a specific source type
    (HuggingFace, local files, S3, GCS, etc.).
    """

    def load(
        self,
        source: DataSource,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from the source.

        Args:
            source: The data source to load from.
            streaming: Whether to stream the data (if supported).

        Yields:
            Raw document dictionaries from the dataset.
        """
        ...

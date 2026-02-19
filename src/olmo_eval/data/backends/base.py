"""Base protocol for data loading backends."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource

# Common keys for JSON data arrays (used by all backends)
JSON_DATA_KEYS = ("data", "instances", "examples", "items", "records")


def extract_json_data(data: Any, path: str) -> Iterator[dict[str, Any]]:
    """Extract data from parsed JSON.

    Handles both array format and dict with common data keys.

    Args:
        data: Parsed JSON data (list or dict).
        path: File path for error messages.

    Yields:
        Individual data records.

    Raises:
        ValueError: If data format is not recognized.
    """
    if isinstance(data, list):
        yield from data
    elif isinstance(data, dict):
        for key in JSON_DATA_KEYS:
            if key in data and isinstance(data[key], list):
                yield from data[key]
                return
        raise ValueError(
            f"JSON file must contain array or object with one of {JSON_DATA_KEYS} key: {path}"
        )
    else:
        raise ValueError(f"JSON file must contain array or object: {path}")


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

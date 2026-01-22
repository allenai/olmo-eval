"""Google Cloud Storage dataset backend."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource


class GCSBackend:
    """Load datasets from Google Cloud Storage.

    Supports JSONL and Parquet files stored in GCS buckets.
    Requires the `smart_open` and `google-cloud-storage` packages for GCS access.

    Examples:
        >>> backend = GCSBackend()
        >>> source = DataSource(path="gs://my-bucket/datasets/data.jsonl")
        >>> for doc in backend.load(source):
        ...     print(doc)
    """

    def load(
        self,
        source: DataSource,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from GCS.

        Args:
            source: The data source with GCS URI.
            streaming: Ignored (always streams for GCS).

        Yields:
            Raw document dictionaries from the dataset.
        """
        path = source.path

        if path.endswith(".jsonl"):
            yield from self._load_jsonl(path)
        elif path.endswith(".json"):
            yield from self._load_json(path)
        elif path.endswith(".parquet"):
            yield from self._load_parquet(path)
        elif path.endswith(".csv"):
            yield from self._load_csv(path)
        else:
            raise ValueError(
                f"Cannot determine file format from GCS path: {path}. "
                "Please use a path ending in .jsonl, .json, .parquet, or .csv"
            )

    def _load_jsonl(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a JSONL file from GCS."""
        try:
            from smart_open import open as smart_open
        except ImportError:
            raise ImportError(
                "smart_open is required for GCS access: pip install smart_open[gcs]"
            )

        with smart_open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_json(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a JSON file from GCS."""
        try:
            from smart_open import open as smart_open
        except ImportError:
            raise ImportError(
                "smart_open is required for GCS access: pip install smart_open[gcs]"
            )

        with smart_open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and "data" in data:
            yield from data["data"]
        else:
            raise ValueError(f"JSON file must contain array or object with 'data' key: {path}")

    def _load_parquet(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a Parquet file from GCS."""
        try:
            import pyarrow.parquet as pq
            from smart_open import open as smart_open
        except ImportError:
            raise ImportError(
                "pyarrow and smart_open are required for GCS Parquet access: "
                "pip install pyarrow smart_open[gcs]"
            )

        with smart_open(path, "rb") as f:
            table = pq.read_table(f)

        for batch in table.to_batches():
            for row in batch.to_pylist():
                yield row

    def _load_csv(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a CSV file from GCS."""
        import csv
        import io

        try:
            from smart_open import open as smart_open
        except ImportError:
            raise ImportError(
                "smart_open is required for GCS access: pip install smart_open[gcs]"
            )

        with smart_open(path, "r", encoding="utf-8") as f:
            content = f.read()

        reader = csv.DictReader(io.StringIO(content))
        yield from reader

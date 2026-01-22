"""Local file system dataset backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource


class LocalBackend:
    """Load datasets from local files.

    Supports JSONL, Parquet, CSV files, and directories containing these files.

    Examples:
        >>> backend = LocalBackend()
        >>> source = DataSource(path="/path/to/dataset.jsonl")
        >>> for doc in backend.load(source):
        ...     print(doc)
    """

    def load(
        self,
        source: DataSource,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from local files.

        Args:
            source: The data source with local file path.
            streaming: Ignored for local files (always streams).

        Yields:
            Raw document dictionaries from the dataset.
        """
        # Remove file:// prefix if present
        path_str = source.path.removeprefix("file://")
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        if path.is_dir():
            yield from self._load_directory(path, source.split)
        elif path.suffix == ".jsonl":
            yield from self._load_jsonl(path)
        elif path.suffix == ".json":
            yield from self._load_json(path)
        elif path.suffix == ".parquet":
            yield from self._load_parquet(path)
        elif path.suffix == ".csv":
            yield from self._load_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_jsonl(self, path: Path) -> Iterator[dict[str, Any]]:
        """Load a JSONL file."""
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _load_json(self, path: Path) -> Iterator[dict[str, Any]]:
        """Load a JSON file (expects array of objects or object with 'data' key)."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict) and "data" in data:
            yield from data["data"]
        else:
            raise ValueError(f"JSON file must contain array or object with 'data' key: {path}")

    def _load_parquet(self, path: Path) -> Iterator[dict[str, Any]]:
        """Load a Parquet file."""
        try:
            import pyarrow.parquet as pq
        except ImportError as err:
            raise ImportError(
                "pyarrow is required to load Parquet files: pip install pyarrow"
            ) from err

        table = pq.read_table(path)
        for batch in table.to_batches():
            yield from batch.to_pylist()

    def _load_csv(self, path: Path) -> Iterator[dict[str, Any]]:
        """Load a CSV file."""
        import csv

        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            yield from reader

    def _load_directory(self, path: Path, split: str | None = None) -> Iterator[dict[str, Any]]:
        """Load all supported files from a directory.

        If split is specified, looks for files matching the split name.
        Otherwise, loads all supported files in the directory.
        """
        supported_suffixes = {".jsonl", ".json", ".parquet", ".csv"}

        # If split specified, look for split-specific files first
        if split:
            from olmo_eval.data.sources import DataSource

            for suffix in supported_suffixes:
                split_file = path / f"{split}{suffix}"
                if split_file.exists():
                    yield from self.load(
                        DataSource(path=str(split_file), split=split),
                        streaming=False,
                    )
                    return

        # Load all supported files in directory
        files = sorted(f for f in path.iterdir() if f.suffix in supported_suffixes)
        if not files:
            raise ValueError(f"No supported files found in directory: {path}")

        for file_path in files:
            if file_path.suffix == ".jsonl":
                yield from self._load_jsonl(file_path)
            elif file_path.suffix == ".json":
                yield from self._load_json(file_path)
            elif file_path.suffix == ".parquet":
                yield from self._load_parquet(file_path)
            elif file_path.suffix == ".csv":
                yield from self._load_csv(file_path)

"""Google Cloud Storage dataset backend."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource

logger = logging.getLogger(__name__)


class GCSBackend:
    """Load datasets from Google Cloud Storage.

    Supports JSONL, JSON, Parquet, and CSV files stored in GCS buckets.
    Also supports loading all files from a GCS prefix (directory).

    Requires the `smart_open` and `google-cloud-storage` packages for GCS access.

    Examples:
        >>> backend = GCSBackend()
        >>> # Load a single file
        >>> source = DataSource(path="gs://my-bucket/datasets/data.jsonl")
        >>> for doc in backend.load(source):
        ...     print(doc)
        >>> # Load all files from a prefix
        >>> source = DataSource(path="gs://my-bucket/datasets/mmlu/")
        >>> for doc in backend.load(source):
        ...     print(doc)
    """

    SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".parquet", ".csv"}

    def __init__(self, project: str | None = None):
        """Initialize the GCS backend.

        Args:
            project: Google Cloud project ID. If not specified, uses application defaults.
        """
        self.project = project
        self._gcs_client = None

    @property
    def gcs_client(self):
        """Lazily initialize the GCS client."""
        if self._gcs_client is None:
            try:
                from google.cloud import storage  # type: ignore[import-not-found]
            except ImportError as err:
                raise ImportError(
                    "google-cloud-storage is required for GCS access: "
                    "pip install google-cloud-storage"
                ) from err

            self._gcs_client = storage.Client(project=self.project)
        return self._gcs_client

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

        # Check if path is a prefix (directory) or a file
        if self._is_prefix(path):
            yield from self._load_prefix(path, source.split)
        elif path.endswith(".jsonl"):
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
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}. "
                "For directories, ensure the path ends with '/'."
            )

    def _is_prefix(self, path: str) -> bool:
        """Check if a GCS path is a prefix (directory) rather than a file."""
        if path.endswith("/"):
            return True
        # Check if path has a known extension
        for ext in self.SUPPORTED_EXTENSIONS:
            if path.endswith(ext):
                return False
        # No extension - could be a prefix, try to list objects
        return self._prefix_has_objects(path + "/")

    def _prefix_has_objects(self, prefix: str) -> bool:
        """Check if a GCS prefix contains any objects."""
        bucket_name, blob_prefix = self._parse_gcs_uri(prefix)
        try:
            bucket = self.gcs_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=blob_prefix, max_results=1)
            return any(True for _ in blobs)
        except Exception:
            return False

    def _parse_gcs_uri(self, uri: str) -> tuple[str, str]:
        """Parse a GCS URI into bucket and blob prefix."""
        parsed = urlparse(uri)
        if parsed.scheme != "gs":
            raise ValueError(f"Expected gs:// URI, got: {uri}")
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        return bucket, prefix

    def _list_objects(self, prefix: str) -> Iterator[str]:
        """List all objects under a GCS prefix.

        Args:
            prefix: The GCS URI prefix (e.g., gs://bucket/path/to/dir/)

        Yields:
            Full GCS URIs for each object.
        """
        bucket_name, blob_prefix = self._parse_gcs_uri(prefix)
        bucket = self.gcs_client.bucket(bucket_name)

        for blob in bucket.list_blobs(prefix=blob_prefix):
            # Skip "directory" markers
            if not blob.name.endswith("/"):
                yield f"gs://{bucket_name}/{blob.name}"

    def _load_prefix(
        self,
        prefix: str,
        split: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Load all supported files from a GCS prefix.

        Args:
            prefix: The GCS URI prefix.
            split: Optional split name to filter files by.

        Yields:
            Documents from all files in the prefix.
        """
        # Ensure prefix ends with /
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        # List all objects under the prefix
        objects = list(self._list_objects(prefix))

        if not objects:
            raise ValueError(f"No objects found under GCS prefix: {prefix}")

        # Filter for supported file types
        supported_files = [
            obj for obj in objects if any(obj.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS)
        ]

        if not supported_files:
            raise ValueError(
                f"No supported files found under GCS prefix: {prefix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # If split is specified, try to find split-specific files
        if split:
            split_files = [
                f
                for f in supported_files
                if f"/{split}." in f or f"/{split}/" in f or f.endswith(f"/{split}")
            ]
            if split_files:
                supported_files = split_files
                logger.debug(f"Filtered to split '{split}': {len(supported_files)} files")

        logger.info(f"Loading {len(supported_files)} files from {prefix}")

        # Load each file
        for file_uri in sorted(supported_files):
            logger.debug(f"Loading {file_uri}")
            if file_uri.endswith(".jsonl"):
                yield from self._load_jsonl(file_uri)
            elif file_uri.endswith(".json"):
                yield from self._load_json(file_uri)
            elif file_uri.endswith(".parquet"):
                yield from self._load_parquet(file_uri)
            elif file_uri.endswith(".csv"):
                yield from self._load_csv(file_uri)

    def _get_smart_open(self):
        """Get smart_open configured for GCS."""
        try:
            from smart_open import open as smart_open  # type: ignore[import-not-found]
        except ImportError as err:
            raise ImportError(
                "smart_open is required for GCS access: pip install smart_open[gcs]"
            ) from err

        # Configure transport params if using custom project
        transport_params = {}
        if self.project:
            transport_params["client"] = self.gcs_client

        return smart_open, transport_params

    def _load_jsonl(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a JSONL file from GCS."""
        smart_open, transport_params = self._get_smart_open()

        with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at {path}:{line_num}: {e}")

    def _load_json(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a JSON file from GCS."""
        smart_open, transport_params = self._get_smart_open()

        with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as f:
            data = json.load(f)

        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            # Try common keys for data arrays
            for key in ("data", "instances", "examples", "items", "records"):
                if key in data and isinstance(data[key], list):
                    yield from data[key]
                    return
            raise ValueError(
                f"JSON file must contain array or object with "
                f"'data'/'instances'/'examples' key: {path}"
            )
        else:
            raise ValueError(f"JSON file must contain array or object: {path}")

    def _load_parquet(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a Parquet file from GCS."""
        try:
            import pyarrow.parquet as pq
        except ImportError as err:
            raise ImportError(
                "pyarrow is required for Parquet access: pip install pyarrow"
            ) from err

        smart_open, transport_params = self._get_smart_open()

        with smart_open(path, "rb", transport_params=transport_params) as f:
            table = pq.read_table(f)

        # Yield rows in batches for memory efficiency
        for batch in table.to_batches():
            yield from batch.to_pylist()

    def _load_csv(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a CSV file from GCS."""
        import csv
        import io

        smart_open, transport_params = self._get_smart_open()

        with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as f:
            content = f.read()

        reader = csv.DictReader(io.StringIO(content))
        yield from reader

    def exists(self, path: str) -> bool:
        """Check if a GCS path exists.

        Args:
            path: GCS URI to check.

        Returns:
            True if the path exists (as file or prefix with objects).
        """
        bucket_name, blob_name = self._parse_gcs_uri(path)
        bucket = self.gcs_client.bucket(bucket_name)

        # Check if it's a direct blob
        blob = bucket.blob(blob_name)
        if blob.exists():
            return True

        # Check if it's a prefix with objects
        return self._prefix_has_objects(path if path.endswith("/") else path + "/")

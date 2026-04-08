"""Amazon S3 dataset backend."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from olmo_eval.data.sources import DataSource

logger = logging.getLogger(__name__)


class S3Backend:
    """Load datasets from Amazon S3.

    Supports JSONL, JSON, Parquet, and CSV files stored in S3 buckets.
    Also supports loading all files from an S3 prefix (directory).

    Requires the `smart_open` and `boto3` packages for S3 access.

    Examples:
        >>> backend = S3Backend()
        >>> # Load a single file
        >>> source = DataSource(path="s3://my-bucket/datasets/data.jsonl")
        >>> for doc in backend.load(source):
        ...     print(doc)
        >>> # Load all files from a prefix
        >>> source = DataSource(path="s3://my-bucket/datasets/mmlu/")
        >>> for doc in backend.load(source):
        ...     print(doc)
    """

    SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".parquet", ".csv"}

    def __init__(
        self,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ):
        """Initialize the S3 backend.

        Args:
            endpoint_url: Custom S3 endpoint URL (e.g., for LocalStack or MinIO).
            region_name: AWS region name. If not specified, uses boto3 defaults.
        """
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self._s3_client = None

    @property
    def s3_client(self) -> Any:
        """Lazily initialize the S3 client."""
        if self._s3_client is None:
            try:
                import boto3
            except ImportError as err:
                raise ImportError("boto3 is required for S3 access: pip install boto3") from err

            self._s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
            )
        return self._s3_client

    def load(
        self,
        source: DataSource,
        streaming: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Load documents from S3.

        Args:
            source: The data source with S3 URI.
            streaming: Ignored (always streams for S3).

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
                f"Cannot determine file format from S3 path: {path}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}. "
                "For directories, ensure the path ends with '/'."
            )

    def _is_prefix(self, path: str) -> bool:
        """Check if an S3 path is a prefix (directory) rather than a file."""
        if path.endswith("/"):
            return True
        # Check if path has a known extension
        for ext in self.SUPPORTED_EXTENSIONS:
            if path.endswith(ext):
                return False
        # No extension - could be a prefix, try to list objects
        return self._prefix_has_objects(path + "/")

    def _prefix_has_objects(self, prefix: str) -> bool:
        """Check if an S3 prefix contains any objects."""
        from botocore.exceptions import ClientError

        bucket, key = self._parse_s3_uri(prefix)
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=key,
                MaxKeys=1,
            )
            return response.get("KeyCount", 0) > 0
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            # Only treat "not found" errors as prefix not existing
            if error_code in ("NoSuchBucket", "NoSuchKey", "404"):
                return False
            # Log and re-raise permission/transient errors
            logger.error(f"S3 error checking prefix {prefix}: {error_code} - {e}")
            raise

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """Parse an S3 URI into bucket and key."""
        parsed = urlparse(uri)
        if parsed.scheme != "s3":
            raise ValueError(f"Expected s3:// URI, got: {uri}")
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return bucket, key

    def _list_objects(self, prefix: str) -> Iterator[str]:
        """List all objects under an S3 prefix.

        Args:
            prefix: The S3 URI prefix (e.g., s3://bucket/path/to/dir/)

        Yields:
            Full S3 URIs for each object.
        """
        bucket, key = self._parse_s3_uri(prefix)

        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=key)

        for page in pages:
            for obj in page.get("Contents", []):
                object_key = obj["Key"]
                # Skip "directory" markers
                if not object_key.endswith("/"):
                    yield f"s3://{bucket}/{object_key}"

    def _load_prefix(
        self,
        prefix: str,
        split: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Load all supported files from an S3 prefix.

        Args:
            prefix: The S3 URI prefix.
            split: Optional split name to filter files by.

        Yields:
            Documents from all files in the prefix.
        """
        # Ensure prefix ends with /
        if not prefix.endswith("/"):
            prefix = prefix + "/"

        # Collect supported files lazily (only materializes supported files, not all objects)
        supported_files = []
        has_any_objects = False
        for obj in self._list_objects(prefix):
            has_any_objects = True
            if any(obj.endswith(ext) for ext in self.SUPPORTED_EXTENSIONS):
                supported_files.append(obj)

        if not has_any_objects:
            raise ValueError(f"No objects found under S3 prefix: {prefix}")

        if not supported_files:
            raise ValueError(
                f"No supported files found under S3 prefix: {prefix}. "
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
        """Get smart_open with proper transport params."""
        try:
            from smart_open import open as smart_open
        except ImportError as err:
            raise ImportError(
                "smart_open is required for S3 access: pip install smart_open[s3]"
            ) from err

        # Configure transport params if using custom endpoint
        transport_params = {}
        if self.endpoint_url or self.region_name:
            transport_params["client"] = self.s3_client

        return smart_open, transport_params

    def _load_jsonl(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a JSONL file from S3."""
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
        """Load a JSON file from S3."""
        from olmo_eval.data.backends.base import extract_json_data

        smart_open, transport_params = self._get_smart_open()

        with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as f:
            data = json.load(f)

        yield from extract_json_data(data, path)

    def _load_parquet(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a Parquet file from S3."""
        try:
            import pyarrow.parquet as pq
        except ImportError as err:
            raise ImportError(
                "pyarrow is required for Parquet access: pip install pyarrow"
            ) from err

        smart_open, transport_params = self._get_smart_open()

        # Stream parquet file in batches to avoid loading entire table into memory
        with smart_open(path, "rb", transport_params=transport_params) as f:
            parquet_file = pq.ParquetFile(f)
            for batch in parquet_file.iter_batches():
                yield from batch.to_pylist()

    def _load_csv(self, path: str) -> Iterator[dict[str, Any]]:
        """Load a CSV file from S3."""
        import csv

        smart_open, transport_params = self._get_smart_open()

        # Stream CSV directly from the file handle
        with smart_open(path, "r", encoding="utf-8", transport_params=transport_params) as f:
            reader = csv.DictReader(f)
            yield from reader

    def exists(self, path: str) -> bool:
        """Check if an S3 path exists.

        Args:
            path: S3 URI to check.

        Returns:
            True if the path exists (as file or prefix with objects).
        """
        from botocore.exceptions import ClientError

        bucket, key = self._parse_s3_uri(path)

        # Check if it's a direct object
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            # Only treat "not found" errors as object not existing
            if error_code not in ("NoSuchBucket", "NoSuchKey", "404", "NotFound"):
                # Log and re-raise permission/transient errors
                logger.error(f"S3 error checking path {path}: {error_code} - {e}")
                raise

        # Check if it's a prefix with objects
        return self._prefix_has_objects(path if path.endswith("/") else path + "/")

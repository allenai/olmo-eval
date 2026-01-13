"""S3-based storage backend for evaluation results."""

from __future__ import annotations

import contextlib
import json
from datetime import datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError

from olmo_eval.storage.base import EvalResult, StorageBackend


class S3Backend(StorageBackend):
    """S3-based storage backend that saves results as JSON files in S3.

    Results are organized as:
        s3://{bucket}/{prefix}{model_slug}/{date}/{run_id}.json

    An index is maintained per model:
        s3://{bucket}/{prefix}_index/{model_slug}.json
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "results/",
        region: str | None = None,
        endpoint_url: str | None = None,
    ):
        """Initialize the S3 backend.

        Args:
            bucket: S3 bucket name.
            prefix: Prefix for all keys (default: "results/").
            region: AWS region (optional).
            endpoint_url: Custom endpoint URL (for LocalStack/MinIO).
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        self.region = region
        self.endpoint_url = endpoint_url

        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        self._s3 = boto3.client("s3", **kwargs)
        self._index_cache: dict[str, dict[str, dict[str, Any]]] = {}

    def initialize(self) -> None:
        """Create the bucket if it doesn't exist (for testing)."""
        try:
            self._s3.head_bucket(Bucket=self.bucket)
        except ClientError:
            kwargs: dict[str, Any] = {}
            if self.region and self.region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
            self._s3.create_bucket(Bucket=self.bucket, **kwargs)

    def cleanup(self) -> None:
        """Delete all objects and the bucket (for testing)."""
        # List and delete all objects
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                self._s3.delete_object(Bucket=self.bucket, Key=obj["Key"])

        # Delete index files
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}_index/"):
            for obj in page.get("Contents", []):
                self._s3.delete_object(Bucket=self.bucket, Key=obj["Key"])

    def _model_slug(self, model_name: str) -> str:
        """Convert model name to key-safe slug."""
        return model_name.replace("/", "_").replace(".", "-")

    def _result_key(self, result: EvalResult) -> str:
        """Get the S3 key for a result."""
        model_slug = self._model_slug(result.model_name)
        date_str = result.timestamp.strftime("%Y-%m-%d")
        return f"{self.prefix}{model_slug}/{date_str}/{result.run_id}.json"

    def _index_key(self, model_slug: str) -> str:
        """Get the S3 key for a model's index file."""
        return f"{self.prefix}_index/{model_slug}.json"

    def _load_model_index(self, model_slug: str) -> dict[str, dict[str, Any]]:
        """Load the index for a model."""
        if model_slug in self._index_cache:
            return self._index_cache[model_slug]

        key = self._index_key(model_slug)
        try:
            response = self._s3.get_object(Bucket=self.bucket, Key=key)
            index = json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                index = {}
            else:
                raise

        self._index_cache[model_slug] = index
        return index

    def _save_model_index(self, model_slug: str, index: dict[str, dict[str, Any]]) -> None:
        """Save the index for a model."""
        key = self._index_key(model_slug)
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(index, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        self._index_cache[model_slug] = index

    def save(self, result: EvalResult) -> str:
        """Save an evaluation result to S3."""
        key = self._result_key(result)

        # Save the result
        self._s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(result.to_dict(), indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        # Update index with queryable fields
        model_slug = self._model_slug(result.model_name)
        index = self._load_model_index(model_slug)
        index[result.run_id] = {
            "key": key,
            "model_name": result.model_name,
            "backend_name": result.backend_name,
            "timestamp": result.timestamp.isoformat(),
            "task_names": [t.task_name for t in result.tasks],
            # Additional queryable metadata
            "experiment_name": result.experiment_name,
            "workspace": result.workspace,
            "author": result.author,
            "model_hash": result.model_hash,
            "s3_location": result.s3_location,
        }
        self._save_model_index(model_slug, index)

        return result.run_id

    def get(self, run_id: str) -> EvalResult | None:
        """Retrieve an evaluation result by run_id."""
        # Search all model indexes
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}_index/"):
            for obj in page.get("Contents", []):
                index_key = obj["Key"]
                model_slug = index_key.split("/")[-1].replace(".json", "")
                index = self._load_model_index(model_slug)

                if run_id in index:
                    key = index[run_id]["key"]
                    try:
                        response = self._s3.get_object(Bucket=self.bucket, Key=key)
                        data = json.loads(response["Body"].read().decode("utf-8"))
                        return EvalResult.from_dict(data)
                    except ClientError:
                        return None

        return None

    def query(
        self,
        model_name: str | None = None,
        task_name: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[EvalResult]:
        """Query evaluation results by filters."""
        results: list[EvalResult] = []

        # Determine which model indexes to search
        if model_name:
            model_slugs = [self._model_slug(model_name)]
        else:
            # List all model indexes
            model_slugs = []
            paginator = self._s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}_index/"):
                for obj in page.get("Contents", []):
                    slug = obj["Key"].split("/")[-1].replace(".json", "")
                    model_slugs.append(slug)

        for model_slug in model_slugs:
            index = self._load_model_index(model_slug)

            for _run_id, entry in index.items():
                # Apply filters
                if model_name and entry["model_name"] != model_name:
                    continue

                if task_name and task_name not in entry.get("task_names", []):
                    continue

                entry_time = datetime.fromisoformat(entry["timestamp"])
                if start_time and entry_time < start_time:
                    continue
                if end_time and entry_time > end_time:
                    continue

                # Load full result
                try:
                    response = self._s3.get_object(Bucket=self.bucket, Key=entry["key"])
                    data = json.loads(response["Body"].read().decode("utf-8"))
                    results.append(EvalResult.from_dict(data))
                except ClientError:
                    continue

                if len(results) >= limit:
                    break

            if len(results) >= limit:
                break

        # Sort by timestamp descending
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]

    def delete(self, run_id: str) -> bool:
        """Delete an evaluation result from S3."""
        # Search all model indexes
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prefix}_index/"):
            for obj in page.get("Contents", []):
                model_slug = obj["Key"].split("/")[-1].replace(".json", "")
                index = self._load_model_index(model_slug)

                if run_id in index:
                    key = index[run_id]["key"]

                    # Delete the object
                    with contextlib.suppress(ClientError):
                        self._s3.delete_object(Bucket=self.bucket, Key=key)

                    # Update index
                    del index[run_id]
                    self._save_model_index(model_slug, index)
                    return True

        return False

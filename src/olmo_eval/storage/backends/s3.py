"""S3-based storage backend for evaluation results."""

from __future__ import annotations

import json
from typing import Any

import boto3
from botocore.exceptions import ClientError

from olmo_eval.core.types import EvalResult


class S3Backend:
    """Simple S3 storage backend for evaluation results.

    Saves and retrieves EvalResult objects as JSON files in S3.
    The caller provides the complete S3 path for each operation.
    """

    def __init__(
        self,
        region: str | None = None,
        endpoint_url: str | None = None,
    ):
        """Initialize the S3 backend.

        Args:
            region: AWS region (optional).
            endpoint_url: Custom endpoint URL (for LocalStack/MinIO).
        """
        self.region = region
        self.endpoint_url = endpoint_url

        kwargs: dict[str, Any] = {}
        if region:
            kwargs["region_name"] = region
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        self._s3 = boto3.client("s3", **kwargs)

    def save(self, bucket: str, key: str, result: EvalResult) -> str:
        """Save an evaluation result to S3.

        Args:
            bucket: S3 bucket name.
            key: Full S3 key path for the result.
            result: EvalResult to save.

        Returns:
            The S3 URI of the saved result (s3://bucket/key).
        """
        self._s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(result.to_dict(), indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        return f"s3://{bucket}/{key}"

    def get(self, bucket: str, key: str) -> EvalResult | None:
        """Retrieve an evaluation result from S3.

        Args:
            bucket: S3 bucket name.
            key: Full S3 key path for the result.

        Returns:
            EvalResult if found, None otherwise.
        """
        try:
            response = self._s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(response["Body"].read().decode("utf-8"))
            return EvalResult.from_dict(data)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise
